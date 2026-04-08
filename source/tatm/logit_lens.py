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
    """
    probs_new: np.ndarray
    probs_old: np.ndarray
    logits_new: np.ndarray
    logits_old: np.ndarray


# ── Core ──────────────────────────────────────────────────────────────────────

def run_logit_lens(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answer_new_token: int,
    answer_old_token: int,
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
    """
    n_layers = model.cfg.n_layers
    W_U = model.W_U                       # [d_model, d_vocab]
    has_b_U = model.b_U is not None and model.b_U.abs().sum() > 0

    probs_new = np.zeros(n_layers)
    probs_old = np.zeros(n_layers)
    logits_new = np.zeros(n_layers)
    logits_old = np.zeros(n_layers)

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
    )
