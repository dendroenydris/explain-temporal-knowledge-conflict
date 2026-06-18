#!/usr/bin/env python3
"""Build Layer-3 parametric answer cache from Layer-2 EvalInstances.

Layer 3 stores model-dependent answers for selected Layer-2 questions.  Unlike
Layer 2, this file is tied to a specific model, template, and decoding setup.

Default usage builds parametric answers from B1 questions without using their
Layer-2 context.  Prompts include a one-shot example to encourage concise
answers.  Pass --use-context only when you intentionally want to cache
context-conditioned answers.

    python scripts/build_wikidata_layer3.py \
        --layer2 data/processed/wikidata_layer2_1000.jsonl \
        --model microsoft/phi-3-mini-4k-instruct \
        --template phi3 \
        --out data/processed/wikidata_layer3_phi3_1000.jsonl
"""
from __future__ import annotations

import argparse
import json
import random as _random
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "source"))

from tatm.model import check_match, load_model  # noqa: E402

DEFAULT_LAYER2 = REPO_ROOT / "data/processed/wikidata_layer2.jsonl"
DEFAULT_OUT = REPO_ROOT / "data/processed/wikidata_layer3.jsonl"
ONE_SHOT_QUESTION = "As of 2017, who was the head of government of United Kingdom?"
ONE_SHOT_ANSWER = "Theresa May"


def load_layer2(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_instances(
    records: list[dict],
    *,
    layers: list[str],
    max_instances: int | None,
    number: int | None,
    seed: int,
) -> list[dict]:
    if layers != ["all"]:
        keep_layers = set(layers)
        records = [
            record for record in records
            if str(record.get("instance_id", ""))[:2] in keep_layers
        ]

    if max_instances is not None:
        records = records[:max_instances]

    if number is None:
        return records
    if number < 1:
        raise ValueError("--number must be >= 1")

    fact_ids = sorted({
        str(record.get("fact_id") or record.get("instance_id"))
        for record in records
    })
    if number >= len(fact_ids):
        return records

    selected_fact_ids = set(_random.Random(seed).sample(fact_ids, number))
    return [
        record for record in records
        if str(record.get("fact_id") or record.get("instance_id")) in selected_fact_ids
    ]


def _collect_eos_ids(tokenizer) -> list[int]:
    eos: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for marker in ("<|end|>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>"):
        if marker in vocab and vocab[marker] not in eos:
            eos.append(vocab[marker])
    return eos


def build_concise_prompt(
    *,
    context: str,
    question: str,
    template: str,
) -> str:
    """Build a one-shot prompt that asks for only the answer string."""
    instruction = (
        "Answer each question with only the person's or entity's name. "
        "Do not explain. "
        "Output exactly one line in this format: Answer: <name>."
    )
    if context.strip():
        current = f"Context: {context}\n\nQuestion: {question}"
    else:
        current = f"Question: {question}"

    if template == "phi3":
        return (
            f"<|system|>\n{instruction}<|end|>\n"
            f"<|user|>\nQuestion: {ONE_SHOT_QUESTION}<|end|>\n"
            f"<|assistant|>\n{ONE_SHOT_ANSWER}<|end|>\n"
            f"<|user|>\n{current}<|end|>\n"
            "<|assistant|>\n"
        )
    if template == "qwen":
        return (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {ONE_SHOT_QUESTION}<|im_end|>\n"
            f"<|im_start|>assistant\n{ONE_SHOT_ANSWER}<|im_end|>\n"
            f"<|im_start|>user\n{current}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if template == "llama2":
        return (
            "[INST] <<SYS>>\n"
            f"{instruction}\n"
            "<</SYS>>\n\n"
            f"Question: {ONE_SHOT_QUESTION}\n"
            f"Answer: {ONE_SHOT_ANSWER}\n\n"
            f"{current}\n"
            "Answer: [/INST]"
        )
    if template == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Question: {ONE_SHOT_QUESTION}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{ONE_SHOT_ANSWER}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{current}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if template == "mistral":
        return (
            f"[INST] {instruction}\n\n"
            f"Question: {ONE_SHOT_QUESTION}\n"
            f"Answer: {ONE_SHOT_ANSWER}\n\n"
            f"{current}\n"
            "Answer: [/INST]"
        )

    return (
        f"{instruction}\n\n"
        f"Question: {ONE_SHOT_QUESTION}\n"
        f"Answer: {ONE_SHOT_ANSWER}\n\n"
        f"{current}\n"
        "Answer:"
    )


def _strip_answer_text(text: str) -> str:
    """Normalize raw model output before extracting a short answer."""
    text = text.strip()

    # Prefer the LAST "Answer: ..." marker anywhere in the output so leading
    # chatter / prompt-echo ("Sure ...", "Question: ...") does not interfere.
    markers = list(re.finditer(r"(?im)\b(?:final\s+)?answer\s*:\s*", text))
    if markers:
        extracted = ""
        for marker in reversed(markers):
            tail = text[marker.end():]
            candidate = tail.split("\n", 1)[0]
            for stop in ("<|", "[/INST]", "[INST]", "Instruction"):
                candidate = candidate.split(stop)[0]
            candidate = candidate.strip(" \t\r\n\"'")
            if candidate:
                extracted = candidate
                break
        text = extracted
    else:
        text = text.split("\n")[0]
    for stop in ("<|", "[", "Instruction"):
        text = text.split(stop)[0]
    return text.strip(" \t\r\n\"'")


def _candidate_match(text: str, candidates: list[str]) -> str:
    """Return a canonical candidate if the model output clearly mentions it."""
    clean = text.lower()
    sorted_candidates = sorted(
        [candidate.strip() for candidate in candidates if candidate.strip()],
        key=len,
        reverse=True,
    )
    for candidate in sorted_candidates:
        if candidate.lower() in clean:
            return candidate
    return ""


def extract_answer(text: str, candidates: list[str] | None = None) -> str:
    """Extract the first short answer phrase from raw generated text."""
    raw_text = text
    text = _strip_answer_text(text)
    if candidates:
        matched = _candidate_match(text, candidates)
        if matched:
            return matched

    # Llama-2 can start with chat boilerplate ("Sure! Here are the answers...")
    # and get truncated before any actual entity appears. Treat this as empty
    # extraction instead of a misleading phrase.
    lower = text.lower().strip()
    if (
        "here are the answers to your questions" in lower
        or lower.startswith("sure")
        or lower.startswith("here are")
        or lower.startswith("question:")
    ):
        # Last chance: sometimes the raw output still contains a candidate name.
        if candidates:
            matched_raw = _candidate_match(raw_text, candidates)
            if matched_raw:
                return matched_raw
        return ""

    # Phi-3 often answers with "As of 2018, the ... was X"; keep only X.
    text = re.sub(r"(?i)^as of\s+(?:19|20)\d{2},?\s*", "", text).strip()
    text = re.sub(r"(?i)^the answer is\s+", "", text).strip()
    text = re.sub(r"(?i)^answer\s*:\s*", "", text).strip()

    # Guard against non-answer templates / uncertainty replies.
    invalid_prefixes = (
        "question:",
        "not specified",
        "unknown",
        "the director or manager of",
        "the officeholder associated",
        "the head coach of",
        "the ceo of",
        "the head of government of",
        "the head of state of",
    )
    lower_clean = text.lower()
    if any(lower_clean.startswith(p) for p in invalid_prefixes):
        if candidates:
            matched_raw = _candidate_match(raw_text, candidates)
            if matched_raw:
                return matched_raw
        return ""

    subject_answer_match = re.match(
        r"^(.+?)\s+(?:was|is|were|are)\s+(?:the\s+)?"
        r"(?:head|officeholder|chairperson|ceo|chief executive officer|"
        r"director|manager|president|prime minister)\b",
        text,
        flags=re.IGNORECASE,
    )
    if subject_answer_match:
        candidate = subject_answer_match.group(1).strip(" ,")
        if not re.match(
            r"(?i)^(?:the\s+)?(?:current\s+)?"
            r"(?:officeholder|chairperson|head|ceo|chief executive officer|"
            r"director|manager|president|prime minister)\b",
            candidate,
        ):
            return _trim_answer_tail(candidate)

    predicate_match = re.search(
        r"\b(?:was|is|were|are)\s+(?!associated\b)(?:named\s+|called\s+)?(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if predicate_match:
        text = predicate_match.group(1).strip(" ,")
        text = re.sub(
            r"(?i)^(?:the\s+)?(?:current\s+)?"
            r"(?:officeholder|chairperson|head(?:\s+of\s+(?:government|state))?|"
            r"ceo|chief executive officer|director|manager|president|prime minister)\s+"
            r"(?:of|for|associated with|was|is)?\s*",
            "",
            text,
        ).strip(" ,")

    return _trim_answer_tail(text)


def _trim_answer_tail(text: str) -> str:
    """Trim explanation tails without cutting honorifics like ``Rt Hon.``."""
    text = text.strip(" \t\r\n\"'")
    text = re.split(r",\s+(?:who|which|the|a)\b", text, maxsplit=1)[0]
    text = re.split(r"\s+-\s+", text, maxsplit=1)[0]
    text = re.split(
        r"\s+(?:who|which)\s+(?:is|was|were|are)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = re.split(
        r"\s+(?:as|because|since)\s+(?:of|the|a|an)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # Only treat a period as a sentence boundary when it follows lowercase text
    # and starts a new sentence.  This preserves initials like "J.C.R. Licklider"
    # and "Jane M. Xiang".
    sentence_boundary = re.search(
        r"(?<=[a-z])\.\s+[A-Z]",
        text,
    )
    if sentence_boundary:
        text = text[: sentence_boundary.start() + 1]

    text = text.strip(" ,")
    if text.endswith(".") and len(text) >= 2 and text[-2].islower():
        text = text[:-1]
    return text


def generate_raw_answer(model, prompt: str, max_new_tokens: int) -> str:
    tok = model.tokenizer
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    current_ids = enc["input_ids"].to(model.cfg.device)
    eos_ids = set(_collect_eos_ids(tok))
    generated: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(current_ids, prepend_bos=False)
            next_id = int(logits[0, -1, :].argmax())
            del logits
            if next_id in eos_ids:
                break
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=current_ids.device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)
            del next_tensor

    return tok.decode(generated, skip_special_tokens=True).strip()


def model_tag(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1].replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--layer2", default=str(DEFAULT_LAYER2), help="Layer-2 JSONL input")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Layer-3 JSONL output")
    parser.add_argument(
        "--model",
        default="microsoft/phi-3-mini-4k-instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--template",
        default="phi3",
        choices=["plain", "llama2", "llama3", "mistral", "phi3", "qwen"],
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["B1"],
        help="Layer-2 instance prefixes to answer, e.g. B1 A1, or 'all' (default: B1)",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "-n", "--number", type=int, default=None,
        help="Randomly sample N Layer-1 facts and keep all selected Layer-2 instances",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="Use each Layer-2 instance context. Default is question-only parametric answering.",
    )
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"],
    )
    args = parser.parse_args()

    # Llama-2/Mistral frequently emit a short preamble before the actual entity.
    # Keep a larger generation budget by default unless explicitly overridden.
    if (
        args.max_new_tokens == 16
        and args.template in {"llama2", "mistral"}
        and "--max-new-tokens" not in sys.argv
    ):
        args.max_new_tokens = 48

    layer2_path = Path(args.layer2)
    out_path = Path(args.out)
    if not layer2_path.exists():
        raise SystemExit(f"[ERROR] Layer-2 not found: {layer2_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if args.dtype == "auto":
        dev = args.device
        if dev == "auto":
            dev = (
                "cuda" if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        resolved_dtype = torch.float16 if dev == "cuda" else torch.float32
    else:
        resolved_dtype = dtype_map[args.dtype]

    records = select_instances(
        load_layer2(layer2_path),
        layers=args.layers,
        max_instances=args.max_instances,
        number=args.number,
        seed=args.sample_seed,
    )
    print(f"Layer-2 input : {layer2_path}")
    print(f"Layer-3 output: {out_path}")
    print(f"Model         : {args.model}")
    print(f"Template      : {args.template}")
    print(f"Use context   : {args.use_context}")
    print("Prompt style  : one-shot concise answer")
    print(f"Selected rows : {len(records)}")

    print(f"\nLoading model {args.model} ...")
    model = load_model(args.model, device=args.device, dtype=resolved_dtype)

    tag = model_tag(args.model)
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in tqdm(records, desc="Generating Layer-3", unit="inst", dynamic_ncols=True):
            prompt_context = record.get("context", "") if args.use_context else ""
            prompt = build_concise_prompt(
                context=prompt_context,
                question=record.get("question", ""),
                template=args.template,
            )
            raw = generate_raw_answer(model, prompt, max_new_tokens=args.max_new_tokens)
            extracted = extract_answer(
                raw,
                candidates=[
                    record.get("answer_old", ""),
                    record.get("answer_new", ""),
                ],
            )
            output = {
                "layer3_id": f"L3_{tag}_{record.get('instance_id', '')}",
                "instance_id": record.get("instance_id", ""),
                "fact_id": record.get("fact_id", ""),
                "layer2_type": str(record.get("instance_id", ""))[:2],
                "model": args.model,
                "template": args.template,
                "question": record.get("question", ""),
                "uses_context": args.use_context,
                "prompt_style": "one_shot_concise_answer",
                "context": prompt_context,
                "prompt": prompt,
                "model_output_raw": raw,
                "extracted_answer": extracted,
                "answer_old": record.get("answer_old", ""),
                "answer_new": record.get("answer_new", ""),
                "matches_answer_old": check_match(extracted, record.get("answer_old", "")),
                "matches_answer_new": check_match(extracted, record.get("answer_new", "")),
                "subject_qid": record.get("subject_qid", ""),
                "subject_label": record.get("subject_label", ""),
                "property_pid": record.get("property_pid", ""),
                "property_label": record.get("property_label", ""),
                "t_old": record.get("t_old"),
                "t_new": record.get("t_new"),
                "decoding": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": 0.0,
                    "method": "greedy_argmax",
                },
            }
            fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            fh.flush()

    print(f"\n[OK] Layer-3 written to {out_path}")


if __name__ == "__main__":
    main()
