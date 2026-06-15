"""Model loading, prompt formatting, and tokenization utilities for TATM."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import torch
from transformer_lens import HookedTransformer

YEAR_PAT = re.compile(r"\b(19|20)\d{2}\b")

# Lexical placeholder inserted by `fact_timeline.eval_builder._strip_years`.
# Must stay in lock-step with that constant — see methodology Step 2(c).
YEAR_PLACEHOLDER = "<YEAR>"


# ── Phi-3 compatibility patch ────────────────────────────────────────────────

def _patch_phi3_rope_scaling(model_name: str) -> None:
    """Pre-download and patch Phi-3 modeling file before TransformerLens imports it.

    Timing fix: we call get_cached_module_file() to pull modeling_phi3.py to
    disk FIRST, then patch it, then clear sys.modules so the next import picks
    up the patched version.  Without this step the file is absent when we look
    for it and TL downloads+imports the unpatched version on its own.

    Two bugs fixed in the cached file:
      Bug 1 – KeyError 'type': old code uses rope_scaling["type"] but newer
              configs store the key as "rope_type".
      Bug 2 – ValueError for 'longrope'/'su': old code only handles "linear"
              and "dynamic".  We fall back to standard Phi3RotaryEmbedding
              (correct for the 4K context window, harmless for MI experiments).
              A regex captures the exact indentation of the raise line so the
              replacement is always syntactically valid.
    """
    import importlib
    import sys

    cache_root = (
        Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    )

    # ── Step 1: ensure the file is on disk ────────────────────────────────
    targets = list(cache_root.glob("**/modeling_phi3.py"))
    if not targets:
        try:
            from transformers.dynamic_module_utils import get_cached_module_file
            local = get_cached_module_file(
                model_name,
                "modeling_phi3.py",
                trust_remote_code=True,
            )
            targets = [Path(local)]
        except Exception as exc:
            print(f"  [patch] Warning: could not pre-download modeling_phi3.py: {exc}")
            return

    # ── Step 2: apply fixes to every copy found ───────────────────────────
    raise_pat = re.compile(
        r'^( *)raise ValueError\(f"Unknown RoPE scaling type \{scaling_type\}"\)',
        re.MULTILINE,
    )

    for p in targets:
        text = p.read_text(encoding="utf-8")
        changed = False

        # Fix 1: key name lookup
        old1 = 'scaling_type = self.config.rope_scaling["type"]'
        new1 = (
            'scaling_type = (self.config.rope_scaling.get("type") or '
            'self.config.rope_scaling.get("rope_type") or "unknown")'
        )
        if old1 in text:
            text = text.replace(old1, new1)
            changed = True

        # Fix 2: unknown type → standard RoPE (regex preserves actual indentation)
        if raise_pat.search(text):
            def _repl(m: re.Match) -> str:
                ind = m.group(1)      # exact whitespace of the raise line
                sub = ind + "    "    # one extra indent level for arguments
                return (
                    f"{ind}# patched: fall back to standard RoPE for unknown types\n"
                    f"{ind}self.rotary_emb = Phi3RotaryEmbedding(\n"
                    f"{sub}self.head_dim,\n"
                    f"{sub}max_position_embeddings=self.max_position_embeddings,\n"
                    f"{sub}base=self.rope_theta,\n"
                    f"{ind})"
                )
            text = raise_pat.sub(_repl, text)
            changed = True

        if changed:
            p.write_text(text, encoding="utf-8")
            print(f"  [patch] {p.name} in …/{p.parent.name[:40]}")

    # ── Step 3: clear any already-imported version from memory ───────────
    for key in list(sys.modules.keys()):
        if "modeling_phi3" in key:
            del sys.modules[key]
    importlib.invalidate_caches()


# ── Model loading ────────────────────────────────────────────────────────────

# Models natively supported by TransformerLens — do NOT pass trust_remote_code.
# TL has its own weight mapping for these; trust_remote_code makes HF use the
# downloaded custom code which produces a different state-dict structure and
# causes wrong weight mapping (garbage outputs).
_TL_NATIVE_PATTERNS = (
    "phi-1", "phi-2", "phi-3", "phi-4",   # all in TL's OFFICIAL_MODEL_NAMES
    "llama", "mistral", "gemma", "qwen",
)

# Models NOT in TL's official list that still need trust_remote_code
_TRUST_REMOTE_CODE_PATTERNS = ("falcon", "starcoder")


def _needs_trust_remote_code(model_name: str) -> bool:
    name_lower = model_name.lower()
    if any(p in name_lower for p in _TL_NATIVE_PATTERNS):
        return False   # TL-native: never use custom remote code
    return any(p in name_lower for p in _TRUST_REMOTE_CODE_PATTERNS)


def load_model(
    model_name: str,
    device: str = "auto",
    dtype: torch.dtype = torch.float32,
) -> HookedTransformer:
    """Load a HookedTransformer with sensible defaults for TATM experiments.

    device="auto" selects CUDA → MPS → CPU in order of availability.
    dtype defaults to float32 for MPS/CPU (float16 is unstable on MPS).
    Handles Phi-3 rope_scaling compatibility automatically.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # MPS does not support float16 reliably; fall back to float32
    if device == "mps" and dtype == torch.float16:
        dtype = torch.float32

    print(f"  → device={device}, dtype={dtype}")

    extra_kwargs: dict = {}
    if _needs_trust_remote_code(model_name):
        extra_kwargs["trust_remote_code"] = True

    # TL's default weight post-processing (fold_ln, centering) is numerically
    # unstable in float16 and produces garbage outputs for models like Phi-3.
    # The official TL recommendation is to use from_pretrained_no_processing,
    # which sets all five processing flags to False.
    no_proc_kwargs: dict = dict(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )

    def _is_transient_cuda_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in (
            "busy or unavailable",
            "cudaerrordevicesunavailable",
            "all cuda-capable devices are busy",
            "no cuda-capable device",
            "cuda error: out of memory",
            "cuda-capable device(s) is/are busy",
        ))

    # Retry config (overridable via env on the sbatch call).
    max_retries = int(os.environ.get("LOAD_MODEL_RETRIES", "4"))
    retry_wait = float(os.environ.get("LOAD_MODEL_RETRY_WAIT", "45"))

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                dtype=dtype,
                **no_proc_kwargs,
                **extra_kwargs,
            )
            model.eval()
            return model
        except Exception as exc:  # noqa: BLE001 — need broad catch for CUDA driver errors
            last_exc = exc
            transient = device == "cuda" and _is_transient_cuda_error(exc)
            if not transient or attempt == max_retries:
                break
            print(f"  [load_model] CUDA transiently unavailable "
                  f"(attempt {attempt}/{max_retries}): {exc}")
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            print(f"  [load_model] retrying in {retry_wait:.0f}s …")
            time.sleep(retry_wait)

    # Last resort: build on CPU (avoids the on-CUDA weight conversion), then move.
    if device == "cuda" and last_exc is not None and _is_transient_cuda_error(last_exc):
        print("  [load_model] GPU still unavailable — building on CPU then moving "
              "to CUDA (one allocation pass instead of many).")
        try:
            model = HookedTransformer.from_pretrained(
                model_name,
                device="cpu",
                dtype=dtype,
                **no_proc_kwargs,
                **extra_kwargs,
            )
            model = model.to("cuda")
            model.eval()
            return model
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    raise RuntimeError(
        f"Failed to load {model_name} on device={device} after {max_retries} "
        f"attempts. Last error: {last_exc}. If this persists the assigned GPU is "
        f"wedged — resubmit excluding that node (see sacct NodeList)."
    ) from last_exc


# ── Prompt formatting ────────────────────────────────────────────────────────

_TEMPLATES: dict[str, dict[str, str]] = {
    "plain": {
        "with_ctx": "Context: {context}\n\nQuestion: {question}\nAnswer:",
        "no_ctx":   "Question: {question}\nAnswer:",
    },
    "llama3": {
        "with_ctx": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Answer the question based on the provided context. "
            "Give a short, direct answer.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Context: {context}\n\nQuestion: {question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "no_ctx": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Answer the question. Give a short, direct answer.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Question: {question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    "llama2": {
        "with_ctx": (
            "[INST] <<SYS>>\nAnswer the question based on the provided context. "
            "Give a short, direct answer.\n<</SYS>>\n\n"
            "Context: {context}\n\nQuestion: {question} [/INST]"
        ),
        "no_ctx": (
            "[INST] <<SYS>>\nAnswer the question. "
            "Give a short, direct answer.\n<</SYS>>\n\n"
            "Question: {question} [/INST]"
        ),
    },
    # Phi-3 chat template  (<|user|> ... <|end|> \n <|assistant|>)
    "phi3": {
        "with_ctx": (
            "<|system|>\nAnswer the question based on the provided context. "
            "Give a short, direct answer.<|end|>\n"
            "<|user|>\nContext: {context}\n\nQuestion: {question}<|end|>\n"
            "<|assistant|>\n"
        ),
        "no_ctx": (
            "<|system|>\nAnswer the question. "
            "Give a short, direct answer.<|end|>\n"
            "<|user|>\nQuestion: {question}<|end|>\n"
            "<|assistant|>\n"
        ),
    },
    # Qwen2 / Qwen2.5 / Qwen3 chat template  (<|im_start|> ... <|im_end|>)
    "qwen": {
        "with_ctx": (
            "<|im_start|>system\nAnswer the question based on the provided context. "
            "Give a short, direct answer.<|im_end|>\n"
            "<|im_start|>user\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "no_ctx": (
            "<|im_start|>system\nAnswer the question. "
            "Give a short, direct answer.<|im_end|>\n"
            "<|im_start|>user\nQuestion: {question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
}


def build_prompt(
    context: str,
    question: str,
    template: str = "plain",
) -> str:
    """Format context + question into a model prompt."""
    tpl = _TEMPLATES.get(template, _TEMPLATES["plain"])
    key = "with_ctx" if context.strip() else "no_ctx"
    return tpl[key].format(context=context, question=question)


# ── Year-token identification ────────────────────────────────────────────────

_YEAR_EXACT = re.compile(r"^(19|20)\d{2}$")


def find_year_positions(
    token_ids: torch.Tensor,
    tokenizer,
    *,
    target_year: Optional[int] = None,
) -> list[int]:
    """Find token positions that encode 4-digit years.

    Works across tokenizer families:
    - SentencePiece (▁ prefix, used by Phi-3 / LLaMA-2)
    - BPE with Ġ prefix (GPT-2 / RoBERTa style)
    - Plain decode (LLaMA-3 tiktoken style)

    SentencePiece tokenizers (e.g. Phi-3) split "2021" into up to 5 tokens:
    ▁(space), 2, 0, 2, 1.  We use a sliding-window decode over windows of
    1–6 consecutive tokens to catch this case reliably.

    Parameters
    ----------
    token_ids : 1-D tensor of token IDs
    tokenizer : HuggingFace-compatible tokenizer
    target_year : if set, only return positions for this specific year

    Returns
    -------
    Sorted list of 0-indexed token positions.
    """
    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
    positions: set[int] = set()

    # ── Pass 1: raw subword strings (handles "▁2021" as a single token) ───────
    try:
        raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        raw_tokens = None

    if raw_tokens:
        for i, raw in enumerate(raw_tokens):
            if not raw:
                continue
            # strip SentencePiece (▁ U+2581) and GPT-2 BPE (Ġ U+0120) prefixes
            clean = raw.lstrip("\u2581\u0120").strip()
            if _YEAR_EXACT.match(clean):
                if target_year is None or int(clean) == target_year:
                    positions.add(i)

    # ── Pass 2: sliding window over 1-6 consecutive tokens ────────────────────
    # Phi-3 splits "2021" as: ▁(id=29871, decoded='') + '2' + '0' + '2' + '1'
    # so we need windows up to length 6 (space marker + 5 digit tokens for a
    # year like "20xx" or "19xx").
    for window in range(1, 7):
        for i in range(len(ids) - window + 1):
            # skip if all positions in this window are already known year tokens
            window_set = set(range(i, i + window))
            if window_set <= positions:
                continue
            combined = tokenizer.decode(ids[i : i + window]).strip()
            if _YEAR_EXACT.match(combined):
                year_val = int(combined)
                if target_year is None or year_val == target_year:
                    positions.update(window_set)

    return sorted(positions)


def find_year_placeholder_positions(
    token_ids: torch.Tensor,
    tokenizer,
    *,
    placeholder: str = YEAR_PLACEHOLDER,
    max_window: int = 6,
) -> list[int]:
    """Find BPE-token positions occupied by the ``<YEAR>`` placeholder.

    Used for F1-b's B3 / B6 attention measurement: the placeholder inserted
    by ``fact_timeline.eval_builder._strip_years`` is position-preserving,
    so its BPE span is the same residual-stream slot the year occupied in
    B1 / B5.  Per methodology Step 2(c)/(d), Phi-3-mini's tokenizer splits
    ``"<YEAR>"`` into 3 ordinary sub-words (``<``, ``YEAR``, ``>``); other
    families may use different decompositions, so we use a sliding window
    over 1–``max_window`` consecutive tokens.

    Returns the union of all positions whose decoded sub-string contains
    the placeholder (deduplicated, sorted).
    """
    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
    positions: set[int] = set()
    target = placeholder.strip()
    if not target:
        return []

    for window in range(1, max_window + 1):
        for i in range(len(ids) - window + 1):
            window_set = set(range(i, i + window))
            if window_set <= positions:
                continue
            combined = tokenizer.decode(ids[i : i + window])
            if target in combined:
                positions.update(window_set)

    return sorted(positions)


# ── Answer token lookup ──────────────────────────────────────────────────────

def get_first_answer_token(model: "HookedTransformer", answer: str) -> int:
    """Return the token ID of the first *meaningful* subword of *answer*.

    SentencePiece tokenizers (Phi-3, LLaMA-2) may split ``" Mario"`` into
    ``[▁, M, ario]`` where ``▁`` (id=29871) is a whitespace-only marker
    shared by all answers.  Using that marker makes every logit-diff
    identically zero and all Logit Lens trajectories identical.
    We skip any leading token that decodes to pure whitespace.

    Returns ``-1`` if no non-whitespace token can be found.
    """
    for prefix in (" ", ""):
        ids = model.tokenizer.encode(f"{prefix}{answer}", add_special_tokens=False)
        if not ids:
            continue
        for tid in ids:
            if model.tokenizer.decode([tid]).strip():
                return tid
    return -1


# ── Answer matching ──────────────────────────────────────────────────────────

def check_match(generated: str, expected: str) -> bool:
    """Case-insensitive answer matching with three-tier logic.

    Tier 1 (strict):  expected is a substring of generated.
    Tier 2 (loose):   the first token of expected (≥4 chars) appears in
                      generated.  Handles "Baron McFall" vs "Lord McFall".
    Tier 3 (variant): any word ≥6 chars from expected appears in generated,
                      split on whitespace, hyphens, and commas.  Handles
                      transliteration variants ("Yevgeny" vs "Yevgeni"),
                      title abbreviations ("Lord Neuberger of Abbotsbury" vs
                      "David Neuberger, Baron Neuberger of Abbotsbury"), and
                      script-romanisation pairs ("Tokayev" vs "Toqaev" share
                      the middle name "Jomart").

    Special case: if expected looks like an unresolved Wikidata QID (starts
    with "Q" followed only by digits) the match always returns False so the
    instance is visibly flagged as bad data rather than silently correct.
    """
    import re as _re
    # Guard against unresolved QIDs in the dataset
    if _re.fullmatch(r"Q\d+", expected.strip()):
        return False

    gen_lower = generated.lower().strip()
    exp_lower = expected.lower().strip()

    # Tier 1 – substring
    if exp_lower in gen_lower:
        return True

    # Tier 2 – first distinctive word
    first_word = next((w for w in exp_lower.split() if len(w) >= 4), "")
    if first_word and first_word in gen_lower:
        return True

    # Tier 3 – any word ≥6 chars shared between expected and generated
    def _words(s: str) -> list[str]:
        return [w for w in _re.split(r"[\s\-,]+", s) if len(w) >= 6]

    for word in _words(exp_lower):
        if word in gen_lower:
            return True

    return False


def _collect_eos_ids(tokenizer) -> list[int]:
    """Return all token IDs that should stop generation for this model."""
    eos: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    for marker in ("<|end|>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>"):
        vocab = tokenizer.get_vocab()
        if marker in vocab:
            tid = vocab[marker]
            if tid not in eos:
                eos.append(tid)
    return eos


def _clean_generated(text: str) -> str:
    """Extract the first clean answer phrase from raw model output."""
    text = text.strip()
    # If model echoed the "Answer:" prefix, extract what follows
    m = re.search(r"(?i)Answer\s*:\s*([^\n\r]+)", text)
    if m:
        text = m.group(1)
    else:
        text = text.split("\n")[0]
    # Truncate at special-token remnants, sentence boundary, or prompt echoes
    for stop in ("<|", "[", "Instruction"):
        text = text.split(stop)[0]
    text = text.split(".")[0]
    return text.strip()


def generate_answer(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    """Greedy-decode a short answer from *prompt*.

    We bypass TL's model.generate() entirely because TL may prepend a BOS
    token to an already-formatted tensor, corrupting the sequence and causing
    degenerate repetition (e.g. "Has Has Has…").

    Instead we run a manual token-by-token loop:
      1. Encode the full prompt with the HF tokenizer (handles Phi-3 special
         tokens like <|end|> correctly).
      2. Call model(ids, prepend_bos=False) to get logits for each step.
      3. Argmax → append → repeat until EOS or max_new_tokens.
    """
    tok = model.tokenizer
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    current_ids = enc["input_ids"].to(model.cfg.device)

    eos_ids = set(_collect_eos_ids(tok))
    generated: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # prepend_bos=False: the prompt is already fully formatted
            logits = model(current_ids, prepend_bos=False)   # [1, seq, vocab]
            next_id = int(logits[0, -1, :].argmax())
            del logits   # free GPU memory immediately; we only need next_id
            if next_id in eos_ids:
                break
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=current_ids.device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)
            del next_tensor

    raw = tok.decode(generated, skip_special_tokens=True)
    return _clean_generated(raw)
