"""Logit Lens — layer-by-layer vocabulary projection.

Shared infrastructure for F2-b (RouteScore) and F3-a (trajectory analysis).
Projects the residual stream at each layer through ln_final + W_U to track
how answer probabilities evolve across the network.

Memory-efficient: hooks extract only scalar values during the forward pass;
no full residual stream is ever cached on GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformer_lens import HookedTransformer

from tatm.model import get_first_answer_token  # re-exported for callers


@dataclass
class LogitTrajectory:
    """Per-instance layer-by-layer probability trajectory.

    All arrays have shape ``[n_layers]`` — index *l* corresponds to the
    residual stream **after** layer *l* (i.e. ``hook_resid_post`` of block *l*).

    ``ranks_new`` / ``ranks_old`` give the **full-vocabulary rank** of the
    candidate token at each layer (0 = top-1).  Rank is calibration-free and
    robust to late crystallization, where absolute probabilities are
    uninformative mid-stream (methodology F3-a, "measurable instruments").
    """
    probs_new: np.ndarray
    probs_old: np.ndarray
    logits_new: np.ndarray
    logits_old: np.ndarray
    ranks_new: np.ndarray | None = None
    ranks_old: np.ndarray | None = None


# ── Core ──────────────────────────────────────────────────────────────────────

# Module-level flag so the "tuned not implemented" warning only fires once
# per process even if many trajectories are computed in a loop.
_TUNED_WARN_ISSUED = False


def run_logit_lens(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answer_new_token: int,
    answer_old_token: int,
    *,
    lens_kind: str = "raw",
) -> LogitTrajectory:
    """Run Logit Lens on a single prompt.

    At each layer *l*, takes the residual stream at the **last** token
    position, applies ``ln_final`` + ``W_U`` (+ ``b_U``), and records
    the probability / logit for both candidate answers.

    Parameters
    ----------
    model : HookedTransformer
    tokens : ``[1, seq_len]`` token IDs (already formatted, no BOS needed)
    answer_new_token : vocabulary index of the first token of ``answer_new``
    answer_old_token : vocabulary index of the first token of ``answer_old``
    lens_kind : ``"raw"`` (default) or ``"tuned"``.  Methodology F3-a Step 3
        specifies **Tuned Lens** (Belrose 2023) as the primary lens with
        the raw RMSNorm-scaled lens reported as appendix robustness.  The
        ``"tuned"`` path requires per-layer affine probes that are *not
        yet trained* in this repository; selecting it currently falls
        back to raw lens and emits a one-time warning so downstream
        callers see that the lens choice was not honoured.
    """
    global _TUNED_WARN_ISSUED
    if lens_kind == "tuned":
        if not _TUNED_WARN_ISSUED:
            import warnings
            warnings.warn(
                "Tuned Lens (Belrose 2023) was requested but is not yet "
                "implemented in tatm.logit_lens; falling back to raw lens. "
                "Methodology F3-a Step 3 specifies Tuned Lens as the "
                "primary lens — results downstream are appendix-grade "
                "until per-layer affine Tuned-Lens probes are trained "
                "and shipped under tatm.tuned_lens.",
                RuntimeWarning,
                stacklevel=2,
            )
            _TUNED_WARN_ISSUED = True
        lens_kind = "raw"
    elif lens_kind != "raw":
        raise ValueError(
            f"lens_kind must be 'raw' or 'tuned', got {lens_kind!r}"
        )
    n_layers = model.cfg.n_layers
    W_U = model.W_U                       # [d_model, d_vocab]
    has_b_U = model.b_U is not None and model.b_U.abs().sum() > 0

    probs_new = np.zeros(n_layers)
    probs_old = np.zeros(n_layers)
    logits_new = np.zeros(n_layers)
    logits_old = np.zeros(n_layers)
    ranks_new = np.zeros(n_layers, dtype=np.int64)
    ranks_old = np.zeros(n_layers, dtype=np.int64)

    def _make_hook(layer: int):
        def _hook(resid: torch.Tensor, hook) -> torch.Tensor:
            # resid: [batch, seq, d_model]
            last = resid[0, -1, :]                          # [d_model]
            normed = model.ln_final(last.unsqueeze(0))[0]   # apply final LN
            proj = normed @ W_U                              # [d_vocab]
            if has_b_U:
                proj = proj + model.b_U
            probs = torch.softmax(proj, dim=-1)

            probs_new[layer] = probs[answer_new_token].item()
            probs_old[layer] = probs[answer_old_token].item()
            logits_new[layer] = proj[answer_new_token].item()
            logits_old[layer] = proj[answer_old_token].item()
            # Full-vocab rank (0 = top-1): count tokens that strictly outrank.
            ranks_new[layer] = int((proj > proj[answer_new_token]).sum().item())
            ranks_old[layer] = int((proj > proj[answer_old_token]).sum().item())
            return resid
        return _hook

    hooks = [
        (f"blocks.{l}.hook_resid_post", _make_hook(l))
        for l in range(n_layers)
    ]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks, prepend_bos=False)

    return LogitTrajectory(
        probs_new=probs_new,
        probs_old=probs_old,
        logits_new=logits_new,
        logits_old=logits_old,
        ranks_new=ranks_new,
        ranks_old=ranks_old,
    )


# ── Teacher-forced position-aware lens (methodology F3-a position fix) ──────────


@dataclass
class TeacherForcedTrajectory:
    """Layer-by-layer probability / rank of a *teacher-forced* answer.

    The plain :func:`run_logit_lens` reads the **last prompt token** (i.e. the
    distribution over the *first generated token*).  On multi-token answers the
    parametric answer ``a_param`` is emitted *later* in the generation, so its
    first token is invisible at the last-prompt position even when the model
    actually outputs it (see methodology F3-a, instance ``B1_6d98ad7373``).

    This reader appends the answer's own tokens to the prompt and, at each
    answer-token position, records the probability / rank that the model
    assigns to the *correct continuation token* at that position, across all
    layers.  ``per_token_prob`` / ``per_token_rank`` have shape
    ``[n_answer_tokens, n_layers]``; ``prob`` / ``rank`` are the per-layer
    mean over answer tokens (a position-corrected summary).
    """
    answer_tokens: list[int]
    per_token_prob: np.ndarray   # [n_answer_tokens, n_layers]
    per_token_rank: np.ndarray   # [n_answer_tokens, n_layers]
    prob: np.ndarray             # [n_layers] mean over answer tokens
    rank: np.ndarray             # [n_layers] mean over answer tokens


def run_logit_lens_teacher_forced(
    model: HookedTransformer,
    prompt_tokens: torch.Tensor,
    answer: str,
    *,
    lens_kind: str = "raw",
    max_answer_tokens: int = 6,
) -> TeacherForcedTrajectory | None:
    """Position-corrected logit lens over a teacher-forced answer continuation.

    Parameters
    ----------
    prompt_tokens : ``[1, seq_len]`` already-formatted prompt token IDs.
    answer : the answer string to teacher-force (e.g. ``a_param`` or
        ``answer_new``).  Tokenised with a leading space so SentencePiece
        models reproduce mid-sentence tokenisation.
    max_answer_tokens : cap on the number of answer tokens scored (keeps
        cost bounded for long entity names).

    Returns ``None`` if the answer tokenises to nothing usable.
    """
    if lens_kind == "tuned":
        # Honour the same one-time warning + raw fallback as run_logit_lens.
        run_logit_lens(model, prompt_tokens, 0, 0, lens_kind="tuned")
        lens_kind = "raw"

    ans_ids = model.tokenizer.encode(f" {answer}", add_special_tokens=False)
    ans_ids = [t for t in ans_ids if model.tokenizer.decode([t]).strip()]
    ans_ids = ans_ids[:max_answer_tokens]
    if not ans_ids:
        return None

    n_layers = model.cfg.n_layers
    W_U = model.W_U
    has_b_U = model.b_U is not None and model.b_U.abs().sum() > 0

    device = prompt_tokens.device
    ans_tensor = torch.tensor([ans_ids], device=device)
    full = torch.cat([prompt_tokens, ans_tensor], dim=1)

    prompt_len = prompt_tokens.shape[1]
    n_ans = len(ans_ids)
    # Position p predicts token at p+1.  The answer token ``ans_ids[j]`` is the
    # target of the residual stream at position ``prompt_len - 1 + j``.
    target_positions = [prompt_len - 1 + j for j in range(n_ans)]

    per_token_prob = np.zeros((n_ans, n_layers))
    per_token_rank = np.zeros((n_ans, n_layers), dtype=np.int64)

    def _make_hook(layer: int):
        def _hook(resid: torch.Tensor, hook) -> torch.Tensor:
            for j, (pos, tid) in enumerate(zip(target_positions, ans_ids)):
                vec = resid[0, pos, :]
                normed = model.ln_final(vec.unsqueeze(0))[0]
                proj = normed @ W_U
                if has_b_U:
                    proj = proj + model.b_U
                probs = torch.softmax(proj, dim=-1)
                per_token_prob[j, layer] = probs[tid].item()
                per_token_rank[j, layer] = int((proj > proj[tid]).sum().item())
            return resid
        return _hook

    hooks = [
        (f"blocks.{l}.hook_resid_post", _make_hook(l))
        for l in range(n_layers)
    ]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        model.run_with_hooks(full, fwd_hooks=hooks, prepend_bos=False)

    return TeacherForcedTrajectory(
        answer_tokens=ans_ids,
        per_token_prob=per_token_prob,
        per_token_rank=per_token_rank,
        prob=per_token_prob.mean(axis=0),
        rank=per_token_rank.mean(axis=0),
    )
