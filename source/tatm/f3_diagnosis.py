"""F3 Diagnosis — "Time Routed but Overridden" (verdict-conditional title).

Implements the methodology section *F3 Diagnosis* (methodology.md L364–L728).
F3 has four sub-experiments and one final verdict assembly:

* **F3-a**  — Logit Lens Trajectory (descriptive).  Layer-agnostic mid-rise
  filter ``max_{l ∈ [L/4, 2L/3]} P^l(answer_new) − P^L(answer_new) > τ``,
  argmax-alignment to ``ℓ_{H_T}`` and ``ℓ_R`` (independent), late-layer
  drop, sublayer Δ decomposition, asymmetric PARAM_OLD vs PARAM_OTHER
  reading.
* **F3-0.5** — Bridge experiment.  Attribution-patching on the B1-success
  Partition A and the B1-failure F3-trajectory subset (stratified by
  PARAM_OLD/PARAM_OTHER); selects the routing set ``R`` for downstream
  F3-b/c plus the panel-asymmetry verdict.
* **F3-b**  — Routing causality.  Dual (M)/(Z) ablation of ``R``,
  ``H_T``-only, random-same-layer, random-depth-matched.  Specificity
  ratios with bootstrap 95% CIs and the per-head-normalised temporal
  specificity ``ρ_{H_T} = (δ_{H_T}/|H_T|) / (δ_R/|R|)``.
* **F3-c**  — Override + Chain + Content.  2×2 (Routing-KO × Late-KO at
  ``L^*_σ``) with logit-space Chain interaction, Step-4 raw-``W_U``
  projection of sublayer updates plus closed-book / random-late /
  random-mid baselines.

Tuned Lens (Belrose 2023, methodology line 376) is *not* trained in this
repository; F3-a / F3-b / F3-c residual measurements therefore fall
back to the raw lens via :func:`tatm.logit_lens.run_logit_lens` and
emit a one-time warning.  Step 4 Sub-test 1 explicitly mandates raw
``W_U`` projection of *update* vectors so the Tuned-Lens departure
does not contaminate it.

All public phase entries (``run_f3a_trajectory``, ``run_f3_half_bridge``,
``run_f3b_ablation``, ``run_f3c_step1_l_star``, ``run_f3c_step2_3``,
``run_f3c_step4_content``) return JSON-serialisable dataclasses (via
``asdict``) so the driver script can dump per-instance + summary
artefacts without bespoke encoders.
"""
from __future__ import annotations

import gc
import math
import random as _random
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from tatm.logit_lens import LogitTrajectory, run_logit_lens
from tatm.model import (
    build_prompt,
    check_match,
    find_year_positions,
    get_first_answer_token,
)

# ═════════════════════════════════════════════════════════════════════════════
# Methodology constants (defaults; sweep / override at call site)
# ═════════════════════════════════════════════════════════════════════════════

#: F3-a default τ (methodology line 374).
F3A_DEFAULT_TAU            = 0.10
F3A_TAU_SWEEP              = (0.05, 0.10, 0.15)
#: F3-a alignment window half-width k = ⌊L/8⌋ (methodology line 379).
F3A_ALIGNMENT_K_DIV        = 8
#: Mid-rise window [L/4, 2L/3] (methodology line 371) and late window
#: (2L/3, L] (methodology line 562 — disjoint from F3-c late-KO window).
F3_MID_WINDOW_LOW_DIV      = 4
F3_MID_WINDOW_HIGH_FRAC    = (2, 3)   # 2L/3
F3_LATE_WINDOW_LOW_FRAC    = (2, 3)   # (2L/3, L]
#: F3-a positive-control thresholds (methodology table lines 405–411).
F3A_PARAM_OLD_TRAJ_RATE    = 0.40
F3A_PARAM_NEW_TRAJ_RATE_MAX = 0.20
F3A_ALIGN_RATE_THR         = 0.60
F3A_MEDIAN_LATE_DROP_PARAM_OLD = 0.15
F3A_MEDIAN_LATE_DROP_CONTROL_MAX = 0.05
F3A_PARAM_OLD_OVER_OTHER_PP = 0.10
F3A_CRAMER_V_THR           = 0.30

#: F3-0.5 panel-asymmetry threshold |ω_OLD − ω_OTHER| > 0.3 (line 458).
F3HALF_PANEL_ASYMMETRY_THR = 0.30
#: F3-0.5 routing-set selection thresholds (lines 463–467).
F3HALF_OMEGA_HIGH          = 0.70
F3HALF_OMEGA_LOW           = 0.30
F3HALF_MIN_INTERSECT_SIZE  = 3
#: Top-decile head fraction for routing-set definition (line 449).
F3HALF_TOP_DECILE_FRAC     = 0.10

#: F3-b ρ_{H_T} anchors (methodology line 519).
F3B_RHO_HIGH               = 1.50
F3B_RHO_LOW                = 0.70
F3B_BOOTSTRAP_N            = 10000

#: F3-c floor-effect threshold P^L_(2)(answer_new) < 1e-3 (line 612).
F3C_FLOOR_PROB             = 1e-3
F3C_FLOOR_FLAG_LIMIT       = 0.25
#: F3-c late-window selection l > 2L/3 (line 567).
F3C_LATE_FRAC              = (2, 3)
#: F3-c top-3 L^*_σ (line 567).
F3C_TOP_K_LATE             = 3
#: F3-c random baselines: 20 paired draws per instance per arm (line 581).
F3C_N_RANDOM_LATE          = 20
#: F3-c Step 4 r_random sample count (line 661).
F3C_N_RANDOM_CONTENT       = 20
#: F3-c Step 4 baseline anchor (line 671); robustness sweep at appendix.
F3C_CLOSED_BOOK_ANCHOR     = 0.80
F3C_CLOSED_BOOK_ANCHOR_SWEEP = (0.70, 0.80, 0.90)
#: F3-c Step 4 stable-parametric floor (line 651).
F3C_A_PARAM_PROB_FLOOR     = 0.30
F3C_A_PARAM_DROP_LIMIT     = 0.30
#: F3-c Step 4 closed-book detector-quality threshold (line 660).
F3C_CLOSED_BOOK_DETECT_MIN = 0.50
#: F3-c Step 4 top-k for vocab projection (line 643).
F3C_TOP_K_VOCAB            = 10

# ═════════════════════════════════════════════════════════════════════════════
# Prepared instance + Layer-3 / Layer-4 IO
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class F3PreparedInstance:
    """B1 instance enriched with the F3 fields required by methodology L364+.

    ``a_param`` is **the A1(t_new) parametric answer** (methodology line 365).
    ``param_class`` is one of ``"PARAM_OLD"`` (``a_param == answer_old``),
    ``"PARAM_OTHER"`` (``a_param`` neither old nor new), or
    ``"PARAM_NEW"`` (``a_param == answer_new``).  PARAM_NEW is the
    closed-book control population that F3-a / F3-c use as the
    matched-stratum success comparator.
    """

    instance_id: str
    fact_id: str
    row: dict                       # full Layer-2 record
    answer_new: str
    answer_old: str
    a_param: str
    answer_new_tid: int
    a_param_tid: int
    param_class: str                # PARAM_OLD / PARAM_OTHER / PARAM_NEW

    # B1 behavioural label (filled in by classify_b1_behavior or Layer-4)
    b1_success: Optional[bool] = None        # True ↔ B1-success (output = answer_new)
    b1_outputs_param: Optional[bool] = None  # True ↔ output = a_param
    b1_ambiguous: bool = False
    b1_rouge_new: Optional[float] = None
    b1_rouge_param: Optional[float] = None

    # Partition assignment (filled by partition_b1_success_pool)
    partition: Optional[str] = None  # "A" / "B" / "C" / None (B1-failure / control)
    # F3-a per-instance result (filled after run_f3a_trajectory)
    is_f3_trajectory: Optional[bool] = None


# ── Layer-3 IO ──────────────────────────────────────────────────────────────


def _layer3_key(row: dict) -> tuple[str, object, object]:
    return (str(row.get("fact_id", "")), row.get("t_old"), row.get("t_new"))


def load_layer3_by_key(path: str) -> tuple[dict[str, dict], dict[tuple, dict]]:
    """Load Layer-3 (cached parametric answers) keyed by ``layer2_type == 'A1'``.

    Returns ``(by_instance_id_of_B1, by_(fact, t_old, t_new))``.  The
    instance-id map matches B1 ↔ A1 by ``fact_id + t_old + t_new``
    sharing across A1/B1 — see :mod:`fact_timeline.eval_builder`.
    """
    import json

    by_id: dict[str, dict] = {}
    by_key: dict[tuple, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("layer2_type", "")) != "A1":
                continue
            iid = str(row.get("instance_id", ""))
            if iid:
                by_id[iid] = row
            key = _layer3_key(row)
            if key[0]:
                by_key[key] = row
    return by_id, by_key


def _resolve_a_param(model: HookedTransformer, layer3_row: dict) -> tuple[str, int]:
    """Extract a_param string + first-meaningful-token from Layer-3 row."""
    a_param = str(layer3_row.get("extracted_answer", "")).strip()
    if not a_param:
        a_param = str(layer3_row.get("model_output_raw", "")).strip()
    tid = get_first_answer_token(model, a_param)
    return a_param, tid


def _classify_param(
    a_param: str,
    answer_new: str,
    answer_old: str,
) -> str:
    if check_match(a_param, answer_new):
        return "PARAM_NEW"
    if check_match(a_param, answer_old):
        return "PARAM_OLD"
    return "PARAM_OTHER"


def prepare_f3_instances(
    model: HookedTransformer,
    b1_rows: list[dict],
    layer3_by_id: dict[str, dict],
    layer3_by_key: dict[tuple, dict],
) -> list[F3PreparedInstance]:
    """Join Layer-2 B1 rows with Layer-3 parametric answers."""
    prepared: list[F3PreparedInstance] = []
    skipped = {"no_layer3": 0, "bad_token": 0, "duplicate_token": 0}
    for row in b1_rows:
        iid = str(row.get("instance_id", ""))
        l3 = layer3_by_id.get(iid)
        if l3 is None:
            # Layer-3 rows are emitted for A1 instances; the B1 instance shares
            # (fact_id, t_old, t_new) but has a different instance_id.
            key = _layer3_key(row)
            l3 = layer3_by_key.get(key)
        if l3 is None:
            skipped["no_layer3"] += 1
            continue

        answer_new = str(row.get("answer_new", "")).strip()
        answer_old = str(row.get("answer_old", "")).strip()
        a_param, a_param_tid = _resolve_a_param(model, l3)
        new_tid = get_first_answer_token(model, answer_new)
        if a_param_tid < 0 or new_tid < 0:
            skipped["bad_token"] += 1
            continue
        # Multi-token first-token collision (methodology line 626) is reported
        # but does not drop the instance — the Step-4 multi-token handling
        # branch picks it up via the first-token-distinctness diagnostic.
        param_class = _classify_param(a_param, answer_new, answer_old)

        prepared.append(F3PreparedInstance(
            instance_id=iid,
            fact_id=str(row.get("fact_id", "")),
            row=row,
            answer_new=answer_new,
            answer_old=answer_old,
            a_param=a_param,
            answer_new_tid=new_tid,
            a_param_tid=a_param_tid,
            param_class=param_class,
        ))
    if any(skipped.values()):
        parts = [f"{k}={v}" for k, v in skipped.items() if v]
        print(f"  [F3] prepare skipped: {', '.join(parts)}")
    return prepared


# ── Prompt builders for paired A1 (closed book) / B1 (open book) ────────────


def build_f3_pair_prompts(
    *,
    context: str,
    question: str,
    template: str,
) -> tuple[str, str]:
    """Return ``(a1_prompt, b1_prompt)`` for F3-c Step 4 closed-book.

    The A1 (closed-book) prompt is the B1 prompt with the context stripped
    (methodology line 645 "Strip the context from each F3-trajectory
    instance").  Both prompts must share the same question and template
    so the residual-stream slot of the question / last token is comparable.
    """
    a1 = build_prompt("", question, template=template)
    b1 = build_prompt(context, question, template=template)
    return a1, b1


# ═════════════════════════════════════════════════════════════════════════════
# Methodology layer-window helpers
# ═════════════════════════════════════════════════════════════════════════════


def mid_window(n_layers: int) -> tuple[int, int]:
    """Return inclusive ``[L/4, 2L/3]`` mid-rise window (methodology line 371)."""
    lo = max(1, n_layers // F3_MID_WINDOW_LOW_DIV)
    hi = (n_layers * F3_MID_WINDOW_HIGH_FRAC[0]) // F3_MID_WINDOW_HIGH_FRAC[1]
    return lo, max(lo, hi - 1)  # half-open upper bound semantics → close at 2L/3 − 1


def late_window(n_layers: int) -> tuple[int, int]:
    """Return inclusive ``(2L/3, L]`` late window (methodology line 562)."""
    lo = (n_layers * F3_LATE_WINDOW_LOW_FRAC[0]) // F3_LATE_WINDOW_LOW_FRAC[1]
    return lo + 1, n_layers - 1


def alignment_k(n_layers: int) -> int:
    """Return ``k = ⌊L/8⌋`` (methodology line 379)."""
    return max(1, n_layers // F3A_ALIGNMENT_K_DIV)


# ═════════════════════════════════════════════════════════════════════════════
# Sublayer probability decomposition (F3-c Step 1)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class SublayerProbs:
    """Last-token P^l(t) at the three sub-positions ``pre / mid / post``.

    All arrays are ``[n_layers]``.  ``mid`` corresponds to
    ``hook_resid_mid`` (= ``hook_resid_pre`` + attn-out) and ``post`` to
    ``hook_resid_post`` (= ``hook_resid_mid`` + mlp-out).  The sublayer
    probability contribution to token ``t`` is
    ``Δ^l_attn(t) = P^l_mid(t) − P^l_pre(t)`` and
    ``Δ^l_mlp(t) = P^l_post(t) − P^l_mid(t)`` (methodology line 575).
    """
    p_pre:  np.ndarray
    p_mid:  np.ndarray
    p_post: np.ndarray


def compute_sublayer_probs(
    model: HookedTransformer,
    tokens: torch.Tensor,
    token_ids: Sequence[int],
    *,
    lens_kind: str = "raw",
) -> dict[int, SublayerProbs]:
    """Compute ``P^l_{pre/mid/post}(t)`` at the last token for each target ``t``.

    Returns ``{token_id: SublayerProbs}``.  Uses raw lens (RMSNorm + ``W_U``)
    by default; if ``lens_kind == "tuned"`` falls back to raw via
    :func:`tatm.logit_lens.run_logit_lens` semantics (warning emitted there).
    """
    if lens_kind not in ("raw", "tuned"):
        raise ValueError(f"lens_kind must be 'raw' or 'tuned', got {lens_kind!r}")
    if lens_kind == "tuned":
        # Single source of truth for the Tuned-Lens warning + fallback.
        run_logit_lens(model, tokens, token_ids[0], token_ids[0], lens_kind="tuned")
        lens_kind = "raw"

    n_layers = model.cfg.n_layers
    W_U = model.W_U
    has_b_U = model.b_U is not None and model.b_U.abs().sum() > 0
    target_tensor = torch.tensor(list(token_ids), device=W_U.device, dtype=torch.long)

    # Three [n_layers, n_targets] buffers.
    p_pre  = np.zeros((n_layers, len(token_ids)))
    p_mid  = np.zeros((n_layers, len(token_ids)))
    p_post = np.zeros((n_layers, len(token_ids)))

    def _project(resid: torch.Tensor) -> torch.Tensor:
        last = resid[0, -1, :]
        normed = model.ln_final(last.unsqueeze(0))[0]
        proj = normed @ W_U
        if has_b_U:
            proj = proj + model.b_U
        probs = torch.softmax(proj, dim=-1)
        return probs.index_select(0, target_tensor).float().cpu()

    def _hook_pre(layer):
        def _h(resid, hook):
            p_pre[layer] = _project(resid).numpy()
            return resid
        return _h

    def _hook_mid(layer):
        def _h(resid, hook):
            p_mid[layer] = _project(resid).numpy()
            return resid
        return _h

    def _hook_post(layer):
        def _h(resid, hook):
            p_post[layer] = _project(resid).numpy()
            return resid
        return _h

    hooks = []
    for layer in range(n_layers):
        hooks.append((f"blocks.{layer}.hook_resid_pre",  _hook_pre(layer)))
        hooks.append((f"blocks.{layer}.hook_resid_mid",  _hook_mid(layer)))
        hooks.append((f"blocks.{layer}.hook_resid_post", _hook_post(layer)))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks, prepend_bos=False)

    return {
        tid: SublayerProbs(
            p_pre =p_pre[:, k].copy(),
            p_mid =p_mid[:, k].copy(),
            p_post=p_post[:, k].copy(),
        )
        for k, tid in enumerate(token_ids)
    }


# ═════════════════════════════════════════════════════════════════════════════
# F3-a  Logit Lens Trajectory (Descriptive)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class F3aTrajectoryResult:
    """Per-instance F3-a outcome (methodology lines 367–414).

    ``trajectory_delta = max_{l ∈ [L/4, 2L/3]} P^l(answer_new) − P^L(answer_new)``
    is the layer-agnostic mid-rise score.  An instance is **F3-trajectory-bearing**
    iff ``trajectory_delta > τ`` (default τ=0.10, methodology line 374).
    """
    instance_id: str
    fact_id: str
    param_class: str
    b1_success: Optional[bool]
    n_layers: int

    p_new: np.ndarray = field(repr=False)
    p_param: np.ndarray = field(repr=False)

    mid_lo: int
    mid_hi: int
    mid_peak_layer_new: int
    p_new_mid_peak: float
    p_new_final: float
    trajectory_delta: float
    is_f3_trajectory: bool             # at default τ
    is_f3_trajectory_sweep: dict       # {τ: bool} across F3A_TAU_SWEEP

    late_drop: float                   # P^{l_peak}(new) − min_{l > l_peak} P^l(new)

    # Sublayer Δ stacks (last token) for ATTN + MLP, computed via raw lens.
    delta_attn_new: np.ndarray = field(repr=False)
    delta_mlp_new:  np.ndarray = field(repr=False)
    delta_attn_param: np.ndarray = field(repr=False)
    delta_mlp_param:  np.ndarray = field(repr=False)


def _trajectory_delta(
    probs: np.ndarray,
    n_layers: int,
) -> tuple[float, int]:
    """Return ``(max P^l − P^L, argmax_l)`` over the mid-window [L/4, 2L/3]."""
    lo, hi = mid_window(n_layers)
    window = probs[lo:hi + 1]
    if window.size == 0:
        return 0.0, lo
    arg = int(np.argmax(window))
    peak_layer = lo + arg
    return float(window[arg] - probs[-1]), peak_layer


def _late_drop(probs: np.ndarray, peak_layer: int) -> float:
    if peak_layer >= len(probs) - 1:
        return 0.0
    tail = probs[peak_layer + 1:]
    return float(probs[peak_layer] - tail.min()) if tail.size else 0.0


def run_f3a_trajectory(
    model: HookedTransformer,
    instances: Sequence[F3PreparedInstance],
    *,
    template: str = "phi3",
    lens_kind: str = "tuned",
    tau: float = F3A_DEFAULT_TAU,
    verbose: bool = True,
) -> list[F3aTrajectoryResult]:
    """F3-a: Logit Lens trajectory + sublayer Δ decomposition (methodology L367+).

    Per-instance: cache trajectory ``P^l(answer_new)``, ``P^l(a_param)`` at
    last token; cache sublayer ``P^l_{pre/mid/post}`` so ``Δ^l_attn``,
    ``Δ^l_mlp`` are computed once and reused for F3-c Step 1.

    Returns the per-instance list; the summary
    (:func:`summarize_f3a_population`) is computed separately so callers
    can re-summarise after the F3-0.5 panel-asymmetry pivot.
    """
    n_layers = model.cfg.n_layers
    results: list[F3aTrajectoryResult] = []
    bar = tqdm(instances, desc="F3-a  Logit Lens trajectory",
               disable=not verbose, unit="inst", dynamic_ncols=True)

    for inst in bar:
        row = inst.row
        prompt = build_prompt(
            str(row.get("context", "")),
            str(row.get("question", "")),
            template=template,
        )
        tokens = model.to_tokens(prompt, prepend_bos=False)

        traj = run_logit_lens(
            model,
            tokens,
            inst.answer_new_tid,
            inst.a_param_tid,
            lens_kind=lens_kind,
        )
        p_new   = np.asarray(traj.probs_new, dtype=float)
        p_param = np.asarray(traj.probs_old, dtype=float)  # second-target slot

        delta, peak_layer = _trajectory_delta(p_new, n_layers)
        is_f3 = delta > tau
        is_f3_sweep = {
            f"tau_{t:.2f}": bool(_trajectory_delta(p_new, n_layers)[0] > t)
            for t in F3A_TAU_SWEEP
        }
        ld = _late_drop(p_new, peak_layer)

        # Sublayer Δ — single extra forward pass (3-hook tap per layer).
        sub = compute_sublayer_probs(
            model, tokens, [inst.answer_new_tid, inst.a_param_tid], lens_kind=lens_kind,
        )
        sp_new   = sub[inst.answer_new_tid]
        sp_param = sub[inst.a_param_tid]
        d_attn_new   = sp_new.p_mid  - sp_new.p_pre
        d_mlp_new    = sp_new.p_post - sp_new.p_mid
        d_attn_param = sp_param.p_mid  - sp_param.p_pre
        d_mlp_param  = sp_param.p_post - sp_param.p_mid

        # Cache the trajectory-bearing flag on the prepared instance so
        # downstream filters (F3-0.5 Stage 2, F3-c Step 1 source pool) can
        # use `inst.is_f3_trajectory` without re-evaluating.
        inst.is_f3_trajectory = is_f3

        mid_lo, mid_hi = mid_window(n_layers)
        results.append(F3aTrajectoryResult(
            instance_id=inst.instance_id,
            fact_id=inst.fact_id,
            param_class=inst.param_class,
            b1_success=inst.b1_success,
            n_layers=n_layers,
            p_new=p_new,
            p_param=p_param,
            mid_lo=mid_lo,
            mid_hi=mid_hi,
            mid_peak_layer_new=peak_layer,
            p_new_mid_peak=float(p_new[peak_layer]),
            p_new_final=float(p_new[-1]),
            trajectory_delta=delta,
            is_f3_trajectory=is_f3,
            is_f3_trajectory_sweep=is_f3_sweep,
            late_drop=ld,
            delta_attn_new=d_attn_new,
            delta_mlp_new=d_mlp_new,
            delta_attn_param=d_attn_param,
            delta_mlp_param=d_mlp_param,
        ))

        bar.set_postfix(traj=is_f3, ok=len(results))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    bar.close()
    return results


# ── F3-a population-level summary + positive-control verdict ────────────────


def _alignment_rate(
    peak_layers: Sequence[int],
    ell_ref: int,
    k: int,
) -> float:
    if not peak_layers:
        return 0.0
    inside = [abs(int(p) - int(ell_ref)) <= k for p in peak_layers]
    return float(np.mean(inside))


def _cramer_v_2x2(matrix: np.ndarray) -> float:
    """Cramér's V for a 2×2 contingency.  Returns 0 on degenerate tables."""
    matrix = matrix.astype(float)
    n = matrix.sum()
    if n <= 0:
        return 0.0
    row = matrix.sum(axis=1, keepdims=True)
    col = matrix.sum(axis=0, keepdims=True)
    expected = (row @ col) / n
    if (expected <= 0).any():
        return 0.0
    chi2 = float(((matrix - expected) ** 2 / expected).sum())
    return float(math.sqrt(chi2 / n))  # min(k-1) = 1 for 2×2


@dataclass
class F3aSummary:
    """Population-level F3-a summary (methodology lines 405–414)."""
    tau: float
    n_layers: int
    mid_window: tuple[int, int]
    late_window: tuple[int, int]
    counts_by_class: dict[str, int]
    f3_traj_rate: dict[str, float]
    median_late_drop: dict[str, float]
    align_HT_rate: dict[str, Optional[float]]   # None if ell_HT not provided
    align_R_rate:  dict[str, Optional[float]]
    cramer_v_fail_vs_success: float
    param_old_over_param_other_pp: float
    positive_control: dict[str, bool]
    reading: str  # one of the asymmetric-reading verdict strings
    sweep_traj_rate: dict[str, dict[str, float]]  # {tau: {class: rate}}


def summarize_f3a_population(
    results: Sequence[F3aTrajectoryResult],
    *,
    ell_HT: Optional[int],
    ell_R: Optional[int],
    tau: float = F3A_DEFAULT_TAU,
) -> F3aSummary:
    """Aggregate F3-a results into the positive-control table (line 405)."""
    if not results:
        empty = {"PARAM_OLD": 0, "PARAM_OTHER": 0, "PARAM_NEW": 0}
        return F3aSummary(
            tau=tau, n_layers=0, mid_window=(0, 0), late_window=(0, 0),
            counts_by_class=empty,
            f3_traj_rate={k: 0.0 for k in empty},
            median_late_drop={k: 0.0 for k in empty},
            align_HT_rate={k: None for k in empty},
            align_R_rate ={k: None for k in empty},
            cramer_v_fail_vs_success=0.0,
            param_old_over_param_other_pp=0.0,
            positive_control={},
            reading="no_data",
            sweep_traj_rate={f"tau_{t:.2f}": {k: 0.0 for k in empty} for t in F3A_TAU_SWEEP},
        )

    n_layers = results[0].n_layers
    k_align = alignment_k(n_layers)
    classes = ["PARAM_OLD", "PARAM_OTHER", "PARAM_NEW"]

    by_class: dict[str, list[F3aTrajectoryResult]] = {c: [] for c in classes}
    for r in results:
        by_class.setdefault(r.param_class, []).append(r)

    counts = {c: len(by_class.get(c, [])) for c in classes}
    traj_rate = {
        c: float(np.mean([int(r.is_f3_trajectory) for r in by_class.get(c, [])]))
        if by_class.get(c) else 0.0
        for c in classes
    }
    median_drop = {
        c: float(np.median([r.late_drop for r in by_class.get(c, []) if r.is_f3_trajectory]))
        if any(r.is_f3_trajectory for r in by_class.get(c, [])) else 0.0
        for c in classes
    }
    # Alignment is reported among trajectory-bearing instances per class.
    align_HT = {
        c: (_alignment_rate(
            [r.mid_peak_layer_new for r in by_class.get(c, []) if r.is_f3_trajectory],
            ell_HT, k_align,
        ) if ell_HT is not None and any(r.is_f3_trajectory for r in by_class.get(c, []))
            else None)
        for c in classes
    }
    align_R = {
        c: (_alignment_rate(
            [r.mid_peak_layer_new for r in by_class.get(c, []) if r.is_f3_trajectory],
            ell_R, k_align,
        ) if ell_R is not None and any(r.is_f3_trajectory for r in by_class.get(c, []))
            else None)
        for c in classes
    }

    # Cramér's V on (B1-success vs B1-failure, trajectory yes/no).  Methodology
    # line 410: use the failure / success populations restricted to PARAM_OLD
    # ∪ PARAM_OTHER (the temporal-conflict populations).
    fail_yes = sum(
        1 for r in results
        if r.is_f3_trajectory and r.b1_success is False
        and r.param_class in ("PARAM_OLD", "PARAM_OTHER")
    )
    fail_no = sum(
        1 for r in results
        if not r.is_f3_trajectory and r.b1_success is False
        and r.param_class in ("PARAM_OLD", "PARAM_OTHER")
    )
    succ_yes = sum(
        1 for r in results
        if r.is_f3_trajectory and r.b1_success is True
        and r.param_class in ("PARAM_OLD", "PARAM_OTHER")
    )
    succ_no = sum(
        1 for r in results
        if not r.is_f3_trajectory and r.b1_success is True
        and r.param_class in ("PARAM_OLD", "PARAM_OTHER")
    )
    cramer_v = _cramer_v_2x2(np.array([[fail_yes, fail_no], [succ_yes, succ_no]]))

    gap_pp = traj_rate["PARAM_OLD"] - traj_rate["PARAM_OTHER"]

    # Positive-control gates (methodology lines 405–411 + asymmetric reading L412+).
    pc = {
        "PARAM_OLD_traj_rate_ge_0.40":
            traj_rate["PARAM_OLD"] >= F3A_PARAM_OLD_TRAJ_RATE,
        "PARAM_NEW_traj_rate_le_0.20":
            traj_rate["PARAM_NEW"] <= F3A_PARAM_NEW_TRAJ_RATE_MAX,
        "PARAM_OLD_align_ge_0.60_any":
            ((align_HT["PARAM_OLD"] or 0) >= F3A_ALIGN_RATE_THR
             or (align_R["PARAM_OLD"] or 0) >= F3A_ALIGN_RATE_THR),
        "median_late_drop_PARAM_OLD_ge_0.15":
            median_drop["PARAM_OLD"] >= F3A_MEDIAN_LATE_DROP_PARAM_OLD,
        "median_late_drop_PARAM_NEW_le_0.05":
            median_drop["PARAM_NEW"] <= F3A_MEDIAN_LATE_DROP_CONTROL_MAX,
        "cramer_v_ge_0.30": cramer_v >= F3A_CRAMER_V_THR,
        "PARAM_OLD_over_OTHER_pp_ge_0.10":
            gap_pp >= F3A_PARAM_OLD_OVER_OTHER_PP,
    }

    # Asymmetric reading (methodology lines 412–416).
    if (not pc["PARAM_OLD_traj_rate_ge_0.40"]
            and traj_rate["PARAM_OTHER"] >= F3A_PARAM_OLD_TRAJ_RATE):
        reading = ("PARAM_OLD_premise_undermined_PARAM_OTHER_present — "
                   "continue only as broader A1-parametric override analysis")
    elif (traj_rate["PARAM_OTHER"] >= traj_rate["PARAM_OLD"]
          and traj_rate["PARAM_OTHER"] >= F3A_PARAM_OLD_TRAJ_RATE):
        reading = ("temporal_specificity_falsified — PARAM_OTHER matches "
                   "PARAM_OLD thresholds; F3 reframed as general A1-parametric override")
    elif pc["PARAM_OLD_over_OTHER_pp_ge_0.10"] and pc["PARAM_OLD_traj_rate_ge_0.40"]:
        reading = "F3_stale_temporal_value_supported"
    else:
        reading = "F3_partial — see positive_control gates"

    sweep = {}
    for t in F3A_TAU_SWEEP:
        key = f"tau_{t:.2f}"
        sweep[key] = {
            c: (float(np.mean([int(r.is_f3_trajectory_sweep[key]) for r in by_class.get(c, [])]))
                if by_class.get(c) else 0.0)
            for c in classes
        }

    return F3aSummary(
        tau=tau,
        n_layers=n_layers,
        mid_window=mid_window(n_layers),
        late_window=late_window(n_layers),
        counts_by_class=counts,
        f3_traj_rate=traj_rate,
        median_late_drop=median_drop,
        align_HT_rate=align_HT,
        align_R_rate=align_R,
        cramer_v_fail_vs_success=cramer_v,
        param_old_over_param_other_pp=gap_pp,
        positive_control=pc,
        reading=reading,
        sweep_traj_rate=sweep,
    )


# ═════════════════════════════════════════════════════════════════════════════
# F3-0.5  Bridge Experiment — Attribution Patching
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class HeadAttribution:
    """One ``(layer, head)`` attribution score from F3-0.5."""
    layer: int
    head: int
    score: float


@dataclass
class F3HalfStage:
    """Per-stage attribution outcome (Partition A, fail-OLD, fail-OTHER, fail-pooled)."""
    label: str
    n_instances: int
    per_head_score: np.ndarray  # [n_layers, n_heads]
    top_decile: list[tuple[int, int]]


@dataclass
class F3HalfResult:
    """F3-0.5 routing-set discovery (methodology lines 422–478)."""
    stage1: F3HalfStage              # Partition A B1-success
    stage2_pooled: F3HalfStage       # B1-failure F3-traj pooled
    stage2_param_old: F3HalfStage
    stage2_param_other: F3HalfStage
    omega_pooled: float
    omega_param_old: float
    omega_param_other: float
    panel_asymmetric: bool
    selection_rule: str              # "succ" / "intersection" / "fail"
    R_pooled: list[tuple[int, int]]
    R_param_old: list[tuple[int, int]]
    R_param_other: list[tuple[int, int]]
    H_T_in_R_top5: bool
    H_T_in_R_top10: bool
    H_T_membership_decision: str     # see methodology lines 471–474
    note: str


def _resolve_l_star(probs: np.ndarray) -> int:
    """Return ``l* = argmax_l P^l(answer_new)`` (methodology line 433)."""
    return int(np.argmax(probs))


def _find_answer_positions(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answer_new: str,
) -> list[int]:
    """Find token positions where the first BPE of *answer_new* appears.

    Used as a coarse counterfactual locus: the methodology counterfactual
    (line 435) replaces ``answer_new`` *in context* by a distractor.  We
    take all occurrences of its first BPE inside the prompt and treat
    those positions as the attribution focal points.  If no occurrence
    is found we fall back to the last token, which makes the attribution
    score the last-token logit gradient (still informative as a routing
    ranking, even though it loses the in-context anchor).
    """
    tid = get_first_answer_token(model, answer_new)
    if tid < 0:
        return [int(tokens.shape[-1]) - 1]
    ids = tokens[0].tolist()
    positions = [i for i, t in enumerate(ids) if t == tid]
    if not positions:
        positions = [int(tokens.shape[-1]) - 1]
    return positions


def attribution_patch_at_l_star(
    model: HookedTransformer,
    instance: F3PreparedInstance,
    *,
    template: str = "phi3",
    lens_kind: str = "tuned",
) -> np.ndarray:
    """Attribution-patching score per ``(layer, head)`` at ``l*``.

    Implements Syed et al. (2023) EAP at the head granularity, restricted
    to the layer ``l*`` that maximises ``P^l(answer_new)``.  The score is
    a first-order Taylor expansion of the activation-patching effect:

    .. math::
        s_{l^*, h} \\approx -(z_{\\text{clean}, h} - z_{\\text{cf}, h})
                          \\cdot \\nabla_{z_h} \\mathcal{L}|_{\\text{cf}}

    where the metric ``L`` is the last-token logit of ``answer_new`` and
    the counterfactual run replaces the first BPE of ``answer_new`` in
    context by a distractor.  This is the methodology-prescribed 1–2 OOM
    speed-up over full activation patching (line 478).

    Returns ``[n_layers, n_heads]``; only the ``l*`` row is populated,
    other layers stay at 0 (the routing set is selected from the ``l*``
    row only).  Pooling across instances averages these matrices and
    therefore correctly aggregates "top-decile attribution averaged
    across instances" (line 449).
    """
    row = instance.row
    context = str(row.get("context", ""))
    question = str(row.get("question", ""))
    answer_new = instance.answer_new

    prompt = build_prompt(context, question, template=template)
    tokens = model.to_tokens(prompt, prepend_bos=False)

    # Counterfactual: replace first BPE of answer_new in context by a
    # distractor first-BPE.  If the prepared instance has no distractor
    # set, fall back to a generic token id (the unk / first-vocab id) —
    # this still produces a meaningful attribution ranking because the
    # method is a *gradient* through the routing path, not a marginal
    # likelihood over the answer string.
    distractors = row.get("distractors") or []
    distr_first = ""
    for d in distractors:
        d = str(d).strip()
        if d and not check_match(d, answer_new) and not check_match(d, instance.answer_old):
            distr_first = d
            break
    if not distr_first:
        distr_first = "thing"  # generic neutral counterfactual

    ans_first_tid = instance.answer_new_tid
    distr_first_tid = get_first_answer_token(model, distr_first)
    if distr_first_tid < 0:
        distr_first_tid = ans_first_tid  # degenerate, skips below

    # Build counterfactual token tensor: replace every occurrence of the
    # first-BPE of answer_new (in context, not in the question) by the
    # distractor's first-BPE.  Question text begins after the system /
    # user header — we approximate by skipping the last token (always
    # the assistant generation slot) and the question-region heuristic
    # used in find_answer_positions.
    ids = tokens[0].tolist()
    counterfactual_ids = list(ids)
    n_replaced = 0
    for i, t in enumerate(ids):
        if t == ans_first_tid:
            counterfactual_ids[i] = distr_first_tid
            n_replaced += 1
    if n_replaced == 0 or distr_first_tid == ans_first_tid:
        # Counterfactual is degenerate (no in-context occurrence or same token);
        # use last-token-only zero-baseline as a fallback.  Attribution still
        # yields a head ranking via the answer_new logit gradient.
        cf_tokens = tokens.clone()
        cf_tokens[0, -1] = distr_first_tid
    else:
        cf_tokens = torch.tensor([counterfactual_ids], device=tokens.device)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Cache clean z at all attention layers.
    cache_names = {f"blocks.{l}.attn.hook_z" for l in range(n_layers)}
    with torch.no_grad():
        traj = run_logit_lens(
            model, tokens, ans_first_tid, instance.a_param_tid, lens_kind=lens_kind,
        )
    l_star = _resolve_l_star(np.asarray(traj.probs_new, dtype=float))

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(
            tokens, names_filter=lambda n: n in cache_names, prepend_bos=False,
        )
    z_clean = clean_cache[f"blocks.{l_star}.attn.hook_z"].detach()
    del clean_cache

    # Forward + backward on the counterfactual run with gradients on z_cf.
    # We hook hook_z at l_star to keep the activation tensor in the autograd
    # graph (TransformerLens by default detaches caches).
    grad_buffer = {}

    def _grad_hook(z, hook):
        z_var = z.detach().clone().requires_grad_(True)
        grad_buffer["z_var"] = z_var
        return z_var

    fwd_hooks = [(f"blocks.{l_star}.attn.hook_z", _grad_hook)]
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cf_logits = model.run_with_hooks(cf_tokens, fwd_hooks=fwd_hooks, prepend_bos=False)
    metric = cf_logits[0, -1, ans_first_tid]
    metric.backward()
    z_var = grad_buffer.get("z_var")
    if z_var is None or z_var.grad is None:
        # Backward failed (model in inference-only mode or hook bypassed grad);
        # return zeros so the instance is effectively ignored in the pool mean.
        return np.zeros((n_layers, n_heads))
    grad = z_var.grad.detach()
    z_cf  = z_var.detach()

    score = -((z_clean - z_cf) * grad)        # [batch, seq, head, d_head]
    s_per_head = score[0].sum(dim=(0, -1))    # [head]  — sum over seq + d_head

    out = np.zeros((n_layers, n_heads))
    out[l_star] = s_per_head.float().cpu().numpy()

    # Clean up.
    model.zero_grad(set_to_none=True)
    del grad, z_clean, z_cf, cf_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out


def _top_decile_heads(
    score_matrix: np.ndarray,
    fraction: float = F3HALF_TOP_DECILE_FRAC,
) -> list[tuple[int, int]]:
    flat = []
    for l in range(score_matrix.shape[0]):
        for h in range(score_matrix.shape[1]):
            s = float(score_matrix[l, h])
            if s != 0.0:
                flat.append(((l, h), s))
    if not flat:
        return []
    flat.sort(key=lambda x: x[1], reverse=True)
    k = max(1, int(math.ceil(len(flat) * fraction)))
    return [head for head, _ in flat[:k]]


def _overlap_fraction(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    if not a:
        return 0.0
    sa = set(a)
    sb = set(b)
    return float(len(sa & sb) / len(sa))


def run_f3_half_bridge(
    model: HookedTransformer,
    partition_A: Sequence[F3PreparedInstance],
    failure_traj_param_old: Sequence[F3PreparedInstance],
    failure_traj_param_other: Sequence[F3PreparedInstance],
    *,
    template: str = "phi3",
    lens_kind: str = "tuned",
    H_T: Optional[Sequence[tuple[int, int]]] = None,
    verbose: bool = True,
) -> F3HalfResult:
    """F3-0.5 — Stage 1 (B1-success) + Stage 2 (B1-failure, PARAM-split).

    Returns the routing-set ``R`` for downstream F3-b/c, the panel-asymmetry
    decision, and the ``H_T``-membership reading (lines 470–474).
    """
    def _aggregate(label: str, pool: Sequence[F3PreparedInstance]) -> F3HalfStage:
        if not pool:
            return F3HalfStage(label=label, n_instances=0,
                               per_head_score=np.zeros(
                                   (model.cfg.n_layers, model.cfg.n_heads)),
                               top_decile=[])
        acc = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
        for inst in tqdm(pool, desc=f"F3-0.5  {label}",
                         disable=not verbose, unit="inst"):
            try:
                acc += attribution_patch_at_l_star(
                    model, inst, template=template, lens_kind=lens_kind,
                )
            except Exception as exc:  # pragma: no cover — best-effort logging
                print(f"  [F3-0.5] attribution failed for {inst.instance_id}: {exc}")
        acc /= max(1, len(pool))
        return F3HalfStage(
            label=label, n_instances=len(pool),
            per_head_score=acc, top_decile=_top_decile_heads(acc),
        )

    stage1 = _aggregate("Stage1 PartA B1-success", partition_A)
    s2_old   = _aggregate("Stage2 PARAM_OLD",   failure_traj_param_old)
    s2_other = _aggregate("Stage2 PARAM_OTHER", failure_traj_param_other)
    pooled_pool = list(failure_traj_param_old) + list(failure_traj_param_other)
    s2_pooled = _aggregate("Stage2 pooled", pooled_pool)

    ω        = _overlap_fraction(stage1.top_decile, s2_pooled.top_decile)
    ω_old    = _overlap_fraction(stage1.top_decile, s2_old.top_decile)
    ω_other  = _overlap_fraction(stage1.top_decile, s2_other.top_decile)
    panel_asym = abs(ω_old - ω_other) > F3HALF_PANEL_ASYMMETRY_THR

    # Routing-set selection (table at lines 463–467).
    succ = set(stage1.top_decile)
    fail = set(s2_pooled.top_decile)
    inter = succ & fail
    if ω >= F3HALF_OMEGA_HIGH:
        R_pooled = list(succ)
        rule = "succ"
    elif ω >= F3HALF_OMEGA_LOW and len(inter) >= F3HALF_MIN_INTERSECT_SIZE:
        R_pooled = list(inter)
        rule = "intersection"
    else:
        R_pooled = list(fail)
        rule = "fail"

    # Per-panel R when panel-asymmetry triggers (methodology line 458).
    if panel_asym:
        R_old = s2_old.top_decile
        R_other = s2_other.top_decile
    else:
        R_old = R_pooled
        R_other = R_pooled

    # H_T-membership reading.
    H_T_set = set(H_T or [])
    R_top5  = set(R_pooled[:5])
    R_top10 = set(R_pooled[:10])
    in_top5  = bool(H_T_set and H_T_set.issubset(R_top5))
    in_top10 = bool(H_T_set and H_T_set.issubset(R_top10))
    if not H_T_set:
        decision = "H_T_fallback"
    elif in_top5:
        decision = "H_T_subset_top5"
    elif in_top10:
        decision = "H_T_in_R_not_top5 — specificity probe retained"
    else:
        decision = "H_T_not_in_R_top10 — H_T-only ablation reported as null"

    note = (
        f"ω(pooled)={ω:.2f}, ω(PARAM_OLD)={ω_old:.2f}, ω(PARAM_OTHER)={ω_other:.2f}; "
        f"panel_asymmetric={panel_asym}; rule={rule}; H_T={decision}"
    )

    return F3HalfResult(
        stage1=stage1, stage2_pooled=s2_pooled,
        stage2_param_old=s2_old, stage2_param_other=s2_other,
        omega_pooled=ω, omega_param_old=ω_old, omega_param_other=ω_other,
        panel_asymmetric=panel_asym,
        selection_rule=rule,
        R_pooled=R_pooled, R_param_old=R_old, R_param_other=R_other,
        H_T_in_R_top5=in_top5,
        H_T_in_R_top10=in_top10,
        H_T_membership_decision=decision,
        note=note,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Shared head / sublayer ablation primitives (used by F3-b and F3-c)
# ═════════════════════════════════════════════════════════════════════════════


def _head_z_donor_mean(
    model: HookedTransformer,
    donors: Sequence[F3PreparedInstance],
    head_layers: Sequence[int],
    *,
    template: str = "phi3",
) -> dict[int, torch.Tensor]:
    """Compute the donor-mean of ``hook_z[:, -1, :, :]`` per layer.

    Used for (M) mean-ablation (methodology line 487).  We aggregate per
    instance at the last (prediction) token only, which yields a fixed
    ``[head, d_head]`` constant per layer that can be substituted at any
    sequence position during the ablated run.
    """
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    sums = {l: torch.zeros(n_heads, d_head, device=model.cfg.device, dtype=torch.float32)
            for l in head_layers}
    counts = {l: 0 for l in head_layers}
    cache_names = {f"blocks.{l}.attn.hook_z" for l in head_layers}

    for inst in tqdm(donors, desc="Donor-mean (M)", unit="donor", dynamic_ncols=True):
        prompt = build_prompt(
            str(inst.row.get("context", "")),
            str(inst.row.get("question", "")),
            template=template,
        )
        tokens = model.to_tokens(prompt, prepend_bos=False)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in cache_names, prepend_bos=False,
            )
        for l in head_layers:
            sums[l] += cache[f"blocks.{l}.attn.hook_z"][0, -1].float()
            counts[l] += 1
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {l: (sums[l] / max(1, counts[l])).to(model.cfg.dtype) for l in head_layers}


def _make_head_ablation_hooks(
    heads_by_layer: dict[int, list[int]],
    *,
    protocol: str,
    donor_mean: Optional[dict[int, torch.Tensor]] = None,
) -> list[tuple[str, Callable]]:
    """Return TransformerLens hooks for (M)/(Z) head ablation at last position."""
    def _hook_z(z: torch.Tensor, hook, *, layer: int, heads: list[int]) -> torch.Tensor:
        # z: [batch, seq, head, d_head]
        ablated = z.clone()
        for h in heads:
            if protocol == "Z":
                ablated[:, -1, h, :] = 0.0
            elif protocol == "M":
                if donor_mean is None:
                    raise ValueError("(M) ablation requires donor_mean")
                ablated[:, -1, h, :] = donor_mean[layer][h].to(ablated.dtype)
            else:
                raise ValueError(f"protocol must be 'M' or 'Z', got {protocol!r}")
        return ablated

    return [
        (f"blocks.{l}.attn.hook_z", partial(_hook_z, layer=l, heads=hs))
        for l, hs in heads_by_layer.items()
    ]


def _sublayer_donor_mean(
    model: HookedTransformer,
    donors: Sequence[F3PreparedInstance],
    layers: Sequence[int],
    sigma: str,
    *,
    template: str = "phi3",
) -> dict[int, torch.Tensor]:
    """Donor-mean of ``hook_attn_out`` or ``hook_mlp_out`` at last token."""
    if sigma not in ("attn", "mlp"):
        raise ValueError(f"sigma must be 'attn' or 'mlp', got {sigma!r}")
    hook_name = "hook_attn_out" if sigma == "attn" else "hook_mlp_out"
    d_model = model.cfg.d_model
    sums = {l: torch.zeros(d_model, device=model.cfg.device, dtype=torch.float32)
            for l in layers}
    counts = {l: 0 for l in layers}
    cache_names = {f"blocks.{l}.{hook_name}" for l in layers}

    for inst in tqdm(donors, desc=f"Donor-mean {sigma}", unit="donor",
                     dynamic_ncols=True):
        prompt = build_prompt(
            str(inst.row.get("context", "")),
            str(inst.row.get("question", "")),
            template=template,
        )
        tokens = model.to_tokens(prompt, prepend_bos=False)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in cache_names, prepend_bos=False,
            )
        for l in layers:
            sums[l] += cache[f"blocks.{l}.{hook_name}"][0, -1].float()
            counts[l] += 1
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {l: (sums[l] / max(1, counts[l])).to(model.cfg.dtype) for l in layers}


def _make_sublayer_ablation_hooks(
    layers: Sequence[int],
    sigma: str,
    *,
    protocol: str,
    donor_mean: Optional[dict[int, torch.Tensor]] = None,
) -> list[tuple[str, Callable]]:
    hook_name = "hook_attn_out" if sigma == "attn" else "hook_mlp_out"

    def _hook(act: torch.Tensor, hook, *, layer: int) -> torch.Tensor:
        ablated = act.clone()
        if protocol == "Z":
            ablated[:, -1, :] = 0.0
        elif protocol == "M":
            if donor_mean is None:
                raise ValueError("(M) sublayer KO requires donor_mean")
            ablated[:, -1, :] = donor_mean[layer].to(ablated.dtype)
        else:
            raise ValueError(f"protocol must be 'M' or 'Z', got {protocol!r}")
        return ablated

    return [
        (f"blocks.{l}.{hook_name}", partial(_hook, layer=l))
        for l in layers
    ]


# ═════════════════════════════════════════════════════════════════════════════
# F3-b  Routing Causality
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class F3bConditionStats:
    """Per-condition population stats (methodology line 504)."""
    name: str
    n: int
    r_mid_mean: float
    r_mid_std: float
    delta_mean: float
    delta_std: float
    r_mid_per_instance: list[float]
    delta_per_instance: list[float]


@dataclass
class F3bResult:
    """F3-b ablation outcome (methodology lines 489–536)."""
    R: list[tuple[int, int]]
    H_T: list[tuple[int, int]]
    population_label: str         # e.g. "B1-failure F3-trajectory pooled"
    n_instances: int

    per_condition: dict[str, F3bConditionStats]
    ratio_R_over_rand_same: float
    ratio_R_over_rand_same_CI: tuple[float, float]
    ratio_R_over_rand_depth: float
    ratio_R_over_rand_depth_CI: tuple[float, float]
    rho_HT: Optional[float]
    rho_HT_CI: Optional[tuple[float, float]]
    raw_HT_over_R: Optional[float]
    primary_protocol: str         # "M" | "Z" | "disagree"
    routed_verdict: str
    note: str


def _r_mid(probs: np.ndarray, n_layers: int) -> float:
    lo, hi = mid_window(n_layers)
    window = probs[lo:hi + 1]
    if window.size == 0:
        return 0.0
    return float(window.max() - probs[0])  # baseline-layer = 0 (line 503)


def _bootstrap_ratio_CI(
    num: np.ndarray,
    den: np.ndarray,
    n_resamples: int = F3B_BOOTSTRAP_N,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap mean(num)/mean(den) with 95% CI on the paired sample."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = min(len(num), len(den))
    num = np.asarray(num[:n], dtype=float)
    den = np.asarray(den[:n], dtype=float)
    if n == 0 or float(den.sum()) == 0:
        return 0.0, (0.0, 0.0)
    point = float(num.mean() / den.mean()) if den.mean() != 0 else 0.0
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        d = den[idx].mean()
        samples[i] = (num[idx].mean() / d) if d != 0 else 0.0
    lo, hi = float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
    return point, (lo, hi)


def _per_instance_probs_under_ablation(
    model: HookedTransformer,
    instances: Sequence[F3PreparedInstance],
    *,
    template: str,
    lens_kind: str,
    hooks_fn: Optional[Callable[[F3PreparedInstance], list[tuple[str, Callable]]]],
) -> tuple[list[np.ndarray], list[F3PreparedInstance]]:
    """Run an ablation hook over instances and collect ``P^l(answer_new)``."""
    out_probs = []
    kept = []
    for inst in instances:
        prompt = build_prompt(
            str(inst.row.get("context", "")),
            str(inst.row.get("question", "")),
            template=template,
        )
        tokens = model.to_tokens(prompt, prepend_bos=False)
        try:
            n_layers = model.cfg.n_layers
            W_U = model.W_U
            has_b_U = model.b_U is not None and model.b_U.abs().sum() > 0
            target_tid = torch.tensor([inst.answer_new_tid], device=W_U.device)
            buf = np.zeros(n_layers)

            def _make_lens(layer: int):
                def _h(resid, hook):
                    last = resid[0, -1, :]
                    normed = model.ln_final(last.unsqueeze(0))[0]
                    proj = normed @ W_U
                    if has_b_U:
                        proj = proj + model.b_U
                    probs = torch.softmax(proj, dim=-1)
                    buf[layer] = float(probs.index_select(0, target_tid)[0])
                    return resid
                return _h

            lens_hooks = [
                (f"blocks.{l}.hook_resid_post", _make_lens(l))
                for l in range(n_layers)
            ]
            fwd_hooks = list(lens_hooks)
            if hooks_fn is not None:
                fwd_hooks.extend(hooks_fn(inst))
            with torch.no_grad():
                model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, prepend_bos=False)
            out_probs.append(buf)
            kept.append(inst)
        except Exception as exc:  # pragma: no cover
            print(f"  [F3-b] ablation failed on {inst.instance_id}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out_probs, kept


def _sample_random_heads(
    R: Sequence[tuple[int, int]],
    n_layers: int,
    n_heads: int,
    *,
    mode: str,
    rng: np.random.Generator,
    n_samples: int = 20,
) -> list[list[tuple[int, int]]]:
    """Draw ``n_samples`` random head sets of size ``|R|``.

    * ``mode == "same-layer"``: heads from the same layer as the mode of R.
    * ``mode == "depth-matched"``: heads with the same per-layer
      distribution as R.
    """
    R_set = set(R)
    size = len(R)
    samples: list[list[tuple[int, int]]] = []
    if size == 0:
        return [[]] * n_samples
    if mode == "same-layer":
        # Most common layer in R; pick |R| heads (excluding R) from that layer.
        layers = [l for l, _ in R]
        target_layer = max(set(layers), key=layers.count)
        avail = [h for h in range(n_heads) if (target_layer, h) not in R_set]
        for _ in range(n_samples):
            if len(avail) < size:
                samples.append([(target_layer, h) for h in avail])
            else:
                samples.append([(target_layer, int(h))
                                for h in rng.choice(avail, size=size, replace=False)])
    elif mode == "depth-matched":
        # Match per-layer count distribution of R.
        per_layer_counts: dict[int, int] = {}
        for l, _ in R:
            per_layer_counts[l] = per_layer_counts.get(l, 0) + 1
        for _ in range(n_samples):
            picks: list[tuple[int, int]] = []
            for l, count in per_layer_counts.items():
                avail = [h for h in range(n_heads) if (l, h) not in R_set]
                if len(avail) < count:
                    picks.extend([(l, h) for h in avail])
                else:
                    chosen = rng.choice(avail, size=count, replace=False)
                    picks.extend([(l, int(h)) for h in chosen])
            samples.append(picks)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return samples


def run_f3b_ablation(
    model: HookedTransformer,
    instances: Sequence[F3PreparedInstance],
    R: Sequence[tuple[int, int]],
    *,
    H_T: Optional[Sequence[tuple[int, int]]] = None,
    partition_B_donors: Sequence[F3PreparedInstance] = (),
    template: str = "phi3",
    lens_kind: str = "tuned",
    n_random_samples: int = 20,
    population_label: str = "B1-failure F3-trajectory pooled",
    rng_seed: int = 42,
    verbose: bool = True,
) -> F3bResult:
    """F3-b — ablation of ``R`` / ``H_T`` / random baselines (methodology L489+).

    Implements the dual (M)/(Z) protocol, computes specificity ratios with
    bootstrap CIs, and resolves the primary protocol for downstream F3-c.
    """
    if not instances:
        raise ValueError("F3-b requires a non-empty instance pool")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    rng = np.random.default_rng(rng_seed)

    heads_R = {l: [h for ll, h in R if ll == l] for l in {l for l, _ in R}}
    heads_HT = {l: [h for ll, h in (H_T or []) if ll == l] for l in {l for l, _ in (H_T or [])}}

    # Donor-mean for (M) — computed once and reused for R, H_T, and all
    # random samples (since same layers + same protocol).
    layers_for_M = sorted(set(heads_R.keys()) | set(heads_HT.keys()))
    donor_mean = (_head_z_donor_mean(model, partition_B_donors, layers_for_M,
                                     template=template)
                  if partition_B_donors else {})

    # ── Clean baseline ────────────────────────────────────────────────
    clean_probs, kept = _per_instance_probs_under_ablation(
        model, instances, template=template, lens_kind=lens_kind, hooks_fn=None,
    )
    clean_r = np.array([_r_mid(p, n_layers) for p in clean_probs])

    def _run(label: str, heads: dict[int, list[int]], protocol: str) -> F3bConditionStats:
        if not heads:
            return F3bConditionStats(
                name=label, n=0,
                r_mid_mean=0.0, r_mid_std=0.0, delta_mean=0.0, delta_std=0.0,
                r_mid_per_instance=[], delta_per_instance=[],
            )
        hooks_fn = lambda inst: _make_head_ablation_hooks(
            heads, protocol=protocol, donor_mean=donor_mean if protocol == "M" else None,
        )
        probs, _ = _per_instance_probs_under_ablation(
            model, kept, template=template, lens_kind=lens_kind, hooks_fn=hooks_fn,
        )
        r = np.array([_r_mid(p, n_layers) for p in probs])
        n = min(len(r), len(clean_r))
        delta = clean_r[:n] - r[:n]
        return F3bConditionStats(
            name=label, n=n,
            r_mid_mean=float(r[:n].mean()) if n else 0.0,
            r_mid_std =float(r[:n].std()) if n else 0.0,
            delta_mean=float(delta.mean()) if n else 0.0,
            delta_std =float(delta.std()) if n else 0.0,
            r_mid_per_instance=r[:n].tolist(),
            delta_per_instance=delta.tolist(),
        )

    per_condition: dict[str, F3bConditionStats] = {
        "clean": F3bConditionStats(
            name="clean", n=len(clean_r),
            r_mid_mean=float(clean_r.mean()) if clean_r.size else 0.0,
            r_mid_std =float(clean_r.std())  if clean_r.size else 0.0,
            delta_mean=0.0, delta_std=0.0,
            r_mid_per_instance=clean_r.tolist(),
            delta_per_instance=[0.0] * len(clean_r),
        )
    }

    for proto in ("M", "Z"):
        per_condition[f"R_{proto}"]  = _run(f"R ({proto})",  heads_R,  proto)
        if H_T:
            per_condition[f"HT_{proto}"] = _run(f"H_T ({proto})", heads_HT, proto)

    # Random baselines — averaged across n_random_samples draws.
    def _run_random(mode: str) -> F3bConditionStats:
        deltas_avg: list[float] = []
        r_mid_avg: list[float] = []
        # Use the primary protocol (M by default; falls back to Z if no donors).
        protocol = "M" if donor_mean else "Z"
        for sample in _sample_random_heads(R, n_layers, n_heads,
                                            mode=mode, rng=rng,
                                            n_samples=n_random_samples):
            heads_rand = {l: [h for ll, h in sample if ll == l]
                          for l in {l for l, _ in sample}}
            if protocol == "M" and not all(l in donor_mean for l in heads_rand):
                # Need to extend donor_mean to the random sample's layers.
                missing = [l for l in heads_rand if l not in donor_mean]
                if missing:
                    extra = (_head_z_donor_mean(model, partition_B_donors, missing,
                                                template=template)
                             if partition_B_donors else {})
                    donor_mean.update(extra)
            hooks_fn = lambda inst: _make_head_ablation_hooks(
                heads_rand, protocol=protocol,
                donor_mean=donor_mean if protocol == "M" else None,
            )
            probs, _ = _per_instance_probs_under_ablation(
                model, kept, template=template, lens_kind=lens_kind, hooks_fn=hooks_fn,
            )
            r = np.array([_r_mid(p, n_layers) for p in probs])
            r_mid_avg.append(float(r.mean()))
            deltas_avg.append(float((clean_r[:len(r)] - r).mean()))
        return F3bConditionStats(
            name=f"random_{mode}", n=len(kept),
            r_mid_mean=float(np.mean(r_mid_avg)) if r_mid_avg else 0.0,
            r_mid_std =float(np.std(r_mid_avg))  if r_mid_avg else 0.0,
            delta_mean=float(np.mean(deltas_avg)) if deltas_avg else 0.0,
            delta_std =float(np.std(deltas_avg))  if deltas_avg else 0.0,
            r_mid_per_instance=r_mid_avg,
            delta_per_instance=deltas_avg,
        )

    per_condition["rand_same_layer"]    = _run_random("same-layer")
    per_condition["rand_depth_matched"] = _run_random("depth-matched")

    # ── Specificity ratios (methodology lines 506–518) ───────────────
    # Primary tie-break: choose protocol with larger |δ_R|.
    delta_R_M = np.asarray(per_condition.get("R_M", per_condition["R_Z"]).delta_per_instance)
    delta_R_Z = np.asarray(per_condition["R_Z"].delta_per_instance)
    agree = (np.sign(delta_R_M.mean()) == np.sign(delta_R_Z.mean())) if delta_R_M.size and delta_R_Z.size else False
    primary = "M" if abs(delta_R_M.mean() if delta_R_M.size else 0) \
                   >= abs(delta_R_Z.mean() if delta_R_Z.size else 0) else "Z"
    if not agree:
        primary = "disagree"

    delta_R = np.asarray(per_condition[f"R_{primary}"].delta_per_instance
                          if primary != "disagree" else delta_R_M)
    delta_rs = np.asarray(per_condition["rand_same_layer"].delta_per_instance)
    delta_rd = np.asarray(per_condition["rand_depth_matched"].delta_per_instance)
    # delta_rs is per-sample mean (length n_random_samples), but for the ratio
    # we want a paired bootstrap with the same length as delta_R; broadcast
    # to instance-level paired pseudo-samples by tiling sample means to N.
    if delta_rs.size and delta_R.size:
        rs_tiled = np.tile(delta_rs, int(np.ceil(len(delta_R) / len(delta_rs))))[: len(delta_R)]
    else:
        rs_tiled = np.zeros_like(delta_R)
    if delta_rd.size and delta_R.size:
        rd_tiled = np.tile(delta_rd, int(np.ceil(len(delta_R) / len(delta_rd))))[: len(delta_R)]
    else:
        rd_tiled = np.zeros_like(delta_R)

    ratio_rs, ratio_rs_CI = _bootstrap_ratio_CI(delta_R, rs_tiled, rng=rng)
    ratio_rd, ratio_rd_CI = _bootstrap_ratio_CI(delta_R, rd_tiled, rng=rng)

    rho_HT = None
    rho_HT_CI = None
    raw_HT_over_R = None
    if H_T:
        delta_HT = np.asarray(per_condition[f"HT_{primary if primary != 'disagree' else 'M'}"].delta_per_instance)
        if delta_HT.size and delta_R.size and len(R) > 0 and len(H_T) > 0:
            per_head_HT = delta_HT / max(1, len(H_T))
            per_head_R  = delta_R  / max(1, len(R))
            rho_HT, rho_HT_CI = _bootstrap_ratio_CI(per_head_HT, per_head_R, rng=rng)
            raw_HT_over_R, _ = _bootstrap_ratio_CI(delta_HT, delta_R, rng=rng)

    # ── Decision matrix verdict (methodology table lines 521–530) ────
    routed = "unresolved"
    if primary == "disagree":
        routed = "OOD-confound — method-dependent"
    elif ratio_rs_CI[0] > 1.0 and ratio_rd_CI[0] > 1.0:
        if rho_HT is None:
            routed = "Routed_confirmed"
        elif rho_HT >= F3B_RHO_HIGH:
            routed = "Routed_confirmed_temporal_specific"
        elif rho_HT >= F3B_RHO_LOW:
            routed = "Routed_confirmed_temporal_partial"
        else:
            routed = "Routed_confirmed_not_temporal_specific"
    elif ratio_rs_CI[0] > 1.0 and ratio_rd[0] <= 1.0 if isinstance(ratio_rd_CI, tuple) and ratio_rd_CI[0] <= 1.0 else False:
        routed = "Routed_layer-depth-explained"
    elif ratio_rs_CI[1] <= 1.0 and ratio_rd_CI[1] <= 1.0:
        routed = "Routed_falsified"
    elif delta_R.size and delta_R.mean() < 0:
        routed = "Anti-routing"
    else:
        routed = f"Routed_method_dependent (rs_CI={ratio_rs_CI}, rd_CI={ratio_rd_CI})"

    note = (
        f"primary={primary}; ratio_rs={ratio_rs:.2f} CI={ratio_rs_CI}; "
        f"ratio_rd={ratio_rd:.2f} CI={ratio_rd_CI}; "
        f"rho_HT={rho_HT}; verdict={routed}"
    )

    return F3bResult(
        R=list(R), H_T=list(H_T or []),
        population_label=population_label, n_instances=len(kept),
        per_condition=per_condition,
        ratio_R_over_rand_same=ratio_rs, ratio_R_over_rand_same_CI=ratio_rs_CI,
        ratio_R_over_rand_depth=ratio_rd, ratio_R_over_rand_depth_CI=ratio_rd_CI,
        rho_HT=rho_HT, rho_HT_CI=rho_HT_CI, raw_HT_over_R=raw_HT_over_R,
        primary_protocol=primary, routed_verdict=routed, note=note,
    )


# ═════════════════════════════════════════════════════════════════════════════
# F3-c Step 1  L^*_σ identification (split-sample on S)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class LStarResult:
    """L^*_σ identification result (methodology lines 564–571)."""
    sigma: str
    layers: list[int]                 # top-3 late layers
    s_scores: np.ndarray = field(repr=False)
    selector_used: str                # "S_lσ" / "S_lσ_contrast"
    committal_overlap_n: int
    robustness_topk: dict[str, list[int]]   # alternative-selector top-3s
    n_S: int
    n_success_C: int


def _contribution_to_target(
    sublayer_probs_list: list[SublayerProbs],
    sigma: str,
) -> np.ndarray:
    """Mean ``Δ^l_σ(t)`` across instances (methodology line 575)."""
    stack = np.array([
        (s.p_mid - s.p_pre) if sigma == "attn" else (s.p_post - s.p_mid)
        for s in sublayer_probs_list
    ])
    return stack.mean(axis=0) if stack.size else np.zeros(0)


def run_f3c_step1_l_star(
    model: HookedTransformer,
    selection_S: Sequence[F3PreparedInstance],
    success_C: Sequence[F3PreparedInstance],
    *,
    template: str = "phi3",
    lens_kind: str = "tuned",
    top_k: int = F3C_TOP_K_LATE,
    verbose: bool = True,
) -> dict[str, LStarResult]:
    """F3-c Step 1 — identify L^*_σ on ``S`` with committal positive control."""
    n_layers = model.cfg.n_layers
    lo, hi = late_window(n_layers)

    def _sublayers_for(pool: Sequence[F3PreparedInstance]) -> dict[str, list[SublayerProbs]]:
        per_inst = {"new": [], "param": []}
        for inst in tqdm(pool, desc="L*  sublayer cache", unit="inst",
                         disable=not verbose, dynamic_ncols=True):
            prompt = build_prompt(
                str(inst.row.get("context", "")),
                str(inst.row.get("question", "")),
                template=template,
            )
            tokens = model.to_tokens(prompt, prepend_bos=False)
            sub = compute_sublayer_probs(
                model, tokens, [inst.answer_new_tid, inst.a_param_tid],
                lens_kind=lens_kind,
            )
            per_inst["new"].append(sub[inst.answer_new_tid])
            per_inst["param"].append(sub[inst.a_param_tid])
        return per_inst

    S_subs = _sublayers_for(selection_S)
    C_subs = _sublayers_for(success_C) if success_C else {"new": [], "param": []}

    results: dict[str, LStarResult] = {}

    for sigma in ("attn", "mlp"):
        d_param = _contribution_to_target(S_subs["param"], sigma)
        d_new   = _contribution_to_target(S_subs["new"],   sigma)
        s_vec = d_param - d_new   # S^l_σ (methodology line 577)
        # Restrict to late window.
        late_idx = np.arange(lo, hi + 1)
        late_s = s_vec[late_idx] if late_idx.size else np.array([])

        # Committal positive control on Partition C (B1-success).
        if C_subs["new"]:
            d_param_c = _contribution_to_target(C_subs["param"], sigma)
            d_new_c   = _contribution_to_target(C_subs["new"],   sigma)
            s_c = d_param_c - d_new_c
            top_S = set(np.argsort(-late_s)[:top_k] + lo) if late_s.size else set()
            late_s_c = s_c[late_idx] if late_idx.size else np.array([])
            top_C = set(np.argsort(-late_s_c)[:top_k] + lo) if late_s_c.size else set()
            overlap = len(top_S & top_C)
            if overlap > 1:
                # Fall back to contrast selector.
                contrast = s_vec - s_c
                late_s = contrast[late_idx] if late_idx.size else np.array([])
                selector = "S_lσ_contrast"
            else:
                selector = "S_lσ"
        else:
            overlap = 0
            selector = "S_lσ"

        top_layers = (np.argsort(-late_s)[:top_k] + lo).tolist() if late_s.size else []

        # Robustness alternative selectors (methodology line 570).
        d_only_param = d_param[late_idx] if late_idx.size else np.array([])
        d_only_new = d_new[late_idx] if late_idx.size else np.array([])
        robustness = {
            "top3_by_dParam":
                (np.argsort(-d_only_param)[:top_k] + lo).tolist() if d_only_param.size else [],
            "top3_by_neg_dNew":
                (np.argsort(d_only_new)[:top_k] + lo).tolist() if d_only_new.size else [],
        }

        results[sigma] = LStarResult(
            sigma=sigma,
            layers=top_layers,
            s_scores=s_vec,
            selector_used=selector,
            committal_overlap_n=overlap,
            robustness_topk=robustness,
            n_S=len(selection_S),
            n_success_C=len(success_C),
        )
    return results


# ═════════════════════════════════════════════════════════════════════════════
# F3-c Steps 2–3  Override + Chain (2×2 conditions)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class F3cInstance:
    """Per-instance 2×2 outcome for one σ (methodology lines 590–620)."""
    instance_id: str
    fact_id: str
    param_class: str
    sigma: str
    protocol: str   # "M" or "Z"

    # Final-layer logits (last token).
    logit_new_1: float
    logit_new_2: float
    logit_new_3: float
    logit_new_4: float
    logit_param_1: float
    logit_param_2: float
    logit_param_3: float
    logit_param_4: float

    # Override
    D_1: float
    D_3: float
    delta_D: float
    delta_P_new: float
    delta_P_param: float
    top1_id_1: int
    top1_id_3: int
    flip_to_new: bool
    flip_to_third: bool

    # Chain (logit space) — None if R-KO disabled
    I_sigma: Optional[float]
    I_rand_mean: Optional[float]
    floor_flagged: bool             # P_(2)(answer_new) < 1e-3

    # Vanished-signal diagnostic
    R_pre_L1: float
    R_pre_L2: Optional[float]


@dataclass
class F3cResult:
    """Population-level F3-c Step 2–3 outcome (methodology lines 590–620)."""
    sigma: str
    protocol: str
    L_star: list[int]
    n_instances: int
    population_label: str

    per_instance: list[F3cInstance]

    override_dD_mean: float
    override_dD_CI: tuple[float, float]
    override_dP_new_mean: float
    override_dP_new_CI: tuple[float, float]
    override_dP_param_mean: float
    override_dP_param_CI: tuple[float, float]
    flip_to_new_rate: float
    flip_to_third_rate: float
    success_ceiling_flip_rate: Optional[float]

    chain_I_sigma_mean: Optional[float]
    chain_I_sigma_CI: Optional[tuple[float, float]]
    chain_I_rand_mean: Optional[float]
    chain_I_minus_rand_CI: Optional[tuple[float, float]]
    floor_flagged_fraction: float

    vanished_signal_share: dict[str, float]
    verdict_override: str
    verdict_chain: str


def _logit_safe(p: float) -> float:
    """log(p/(1-p)) clamped against overflow."""
    p = float(p)
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def _bootstrap_paired_mean_CI(
    data: np.ndarray,
    n_resamples: int = F3B_BOOTSTRAP_N,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, tuple[float, float]]:
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return 0.0, (0.0, 0.0)
    point = float(arr.mean())
    samples = np.array([arr[rng.integers(0, arr.size, arr.size)].mean()
                        for _ in range(n_resamples)])
    return point, (float(np.percentile(samples, 2.5)),
                   float(np.percentile(samples, 97.5)))


def _final_logits_with_hooks(
    model: HookedTransformer,
    tokens: torch.Tensor,
    targets: list[int],
    *,
    hooks: list[tuple[str, Callable]],
) -> tuple[np.ndarray, int]:
    """Run with hooks, return (logits[targets], argmax_top1_id)."""
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks, prepend_bos=False)
    last = logits[0, -1].float().cpu()
    top1 = int(last.argmax())
    return last.index_select(0, torch.tensor(targets)).numpy(), top1


def _final_probs_with_hooks(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target: int,
    *,
    hooks: list[tuple[str, Callable]],
) -> float:
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks, prepend_bos=False)
    return float(torch.softmax(logits[0, -1].float(), dim=-1)[target])


def _trajectory_with_hooks(
    model: HookedTransformer,
    tokens: torch.Tensor,
    target: int,
    *,
    extra_hooks: list[tuple[str, Callable]],
) -> np.ndarray:
    """Full ``P^l(target)`` trajectory at last token under arbitrary hooks."""
    n_layers = model.cfg.n_layers
    W_U = model.W_U
    has_b_U = model.b_U is not None and model.b_U.abs().sum() > 0
    buf = np.zeros(n_layers)
    target_tid = torch.tensor([target], device=W_U.device)

    def _make(layer):
        def _h(resid, hook):
            last = resid[0, -1, :]
            normed = model.ln_final(last.unsqueeze(0))[0]
            proj = normed @ W_U
            if has_b_U:
                proj = proj + model.b_U
            probs = torch.softmax(proj, dim=-1)
            buf[layer] = float(probs.index_select(0, target_tid)[0])
            return resid
        return _h

    lens_hooks = [
        (f"blocks.{l}.hook_resid_post", _make(l)) for l in range(n_layers)
    ]
    with torch.no_grad():
        model.run_with_hooks(
            tokens, fwd_hooks=list(extra_hooks) + lens_hooks, prepend_bos=False,
        )
    return buf


def run_f3c_step2_3(
    model: HookedTransformer,
    test_T: Sequence[F3PreparedInstance],
    *,
    sigma: str,
    L_star: Sequence[int],
    R: Sequence[tuple[int, int]],
    R_protocol: str,
    late_ko_protocol: str,
    partition_B_donors: Sequence[F3PreparedInstance] = (),
    partition_C_success: Sequence[F3PreparedInstance] = (),
    template: str = "phi3",
    lens_kind: str = "tuned",
    n_random_late: int = F3C_N_RANDOM_LATE,
    rng_seed: int = 42,
    population_label: str = "B1-failure F3-trajectory ∩ T",
    verbose: bool = True,
) -> F3cResult:
    """F3-c Steps 2–3 — 2×2 Override + Chain (methodology lines 590–620)."""
    if sigma not in ("attn", "mlp"):
        raise ValueError(f"sigma must be 'attn' or 'mlp', got {sigma!r}")
    n_layers = model.cfg.n_layers
    rng = np.random.default_rng(rng_seed)

    heads_R = {l: [h for ll, h in R if ll == l] for l in {l for l, _ in R}}
    R_layers_for_donor = sorted(heads_R.keys())

    # Pre-compute donor means for the (M) protocol on R-KO and Late-KO.
    donor_R_z = (_head_z_donor_mean(model, partition_B_donors, R_layers_for_donor,
                                    template=template)
                 if (R and partition_B_donors) else {})
    donor_sublayer = (_sublayer_donor_mean(model, partition_B_donors, L_star, sigma,
                                            template=template)
                      if (L_star and partition_B_donors) else {})

    # Random-layer baseline samples (paired across (3) and (4)).
    rand_layer_sets: list[list[int]] = []
    lo, hi = late_window(n_layers)
    pool = [l for l in range(lo, hi + 1) if l not in set(L_star)]
    for _ in range(n_random_late):
        if len(pool) < len(L_star):
            rand_layer_sets.append(list(pool))
        else:
            rand_layer_sets.append(sorted(rng.choice(pool, size=len(L_star),
                                                     replace=False).tolist()))
    rand_donor_means: dict[int, torch.Tensor] = dict(donor_sublayer)
    if late_ko_protocol == "M" and partition_B_donors:
        # Ensure all random layers have a donor-mean entry.
        missing = sorted({l for s in rand_layer_sets for l in s} - rand_donor_means.keys())
        if missing:
            rand_donor_means.update(
                _sublayer_donor_mean(model, partition_B_donors, missing, sigma,
                                     template=template))

    per_inst: list[F3cInstance] = []
    for inst in tqdm(test_T, desc=f"F3-c 2x2 σ={sigma}/{late_ko_protocol}",
                     disable=not verbose, unit="inst", dynamic_ncols=True):
        prompt = build_prompt(
            str(inst.row.get("context", "")),
            str(inst.row.get("question", "")),
            template=template,
        )
        tokens = model.to_tokens(prompt, prepend_bos=False)
        targets = [inst.answer_new_tid, inst.a_param_tid]

        # Hooks for each of the four conditions.
        def _ko_R():
            if not heads_R:
                return []
            return _make_head_ablation_hooks(
                heads_R, protocol=R_protocol,
                donor_mean=donor_R_z if R_protocol == "M" else None,
            )

        def _ko_late(layers: Sequence[int], protocol: str,
                     donor: dict[int, torch.Tensor]):
            return _make_sublayer_ablation_hooks(
                layers, sigma, protocol=protocol,
                donor_mean=donor if protocol == "M" else None,
            )

        # Condition (1) clean
        l1, top1_1 = _final_logits_with_hooks(model, tokens, targets, hooks=[])
        # Condition (2) R-KO only
        l2, top1_2 = _final_logits_with_hooks(model, tokens, targets, hooks=_ko_R())
        # Condition (3) Late-KO only
        late_hooks = _ko_late(L_star, late_ko_protocol, donor_sublayer)
        l3, top1_3 = _final_logits_with_hooks(model, tokens, targets, hooks=late_hooks)
        # Condition (4) Both
        l4, top1_4 = _final_logits_with_hooks(
            model, tokens, targets, hooks=list(_ko_R()) + list(late_hooks),
        )

        # P^L for chain + floor flag.
        p_L_2 = float(np.exp(l2[0]) / (np.exp(l2[0]) + 1e-12))   # not exact but cheap
        # Better: re-extract P^L(answer_new) precisely.
        p_L_2 = _final_probs_with_hooks(model, tokens, inst.answer_new_tid,
                                         hooks=_ko_R())
        p_L_1 = _final_probs_with_hooks(model, tokens, inst.answer_new_tid, hooks=[])
        p_L_3 = _final_probs_with_hooks(model, tokens, inst.answer_new_tid,
                                         hooks=late_hooks)
        p_L_4 = _final_probs_with_hooks(model, tokens, inst.answer_new_tid,
                                         hooks=list(_ko_R()) + list(late_hooks))
        p_param_3 = _final_probs_with_hooks(model, tokens, inst.a_param_tid,
                                             hooks=late_hooks)
        p_param_1 = _final_probs_with_hooks(model, tokens, inst.a_param_tid, hooks=[])

        # Pre-L* diagnostic (probability at last layer just *before* the lowest
        # L*_σ layer — methodology line 597).
        pre_l_idx = max(0, min(L_star) - 1) if L_star else 0
        traj_1 = _trajectory_with_hooks(model, tokens, inst.answer_new_tid,
                                         extra_hooks=[])
        R_pre_1 = float(traj_1[pre_l_idx])
        traj_2 = _trajectory_with_hooks(model, tokens, inst.answer_new_tid,
                                         extra_hooks=_ko_R()) if R else None
        R_pre_2 = float(traj_2[pre_l_idx]) if traj_2 is not None else None

        # Chain: logit-space I_σ
        ell_1 = _logit_safe(p_L_1)
        ell_2 = _logit_safe(p_L_2) if R else None
        ell_3 = _logit_safe(p_L_3)
        ell_4 = _logit_safe(p_L_4) if R else None
        floor_flagged = (R is not None) and (p_L_2 < F3C_FLOOR_PROB)
        I_sigma = ((ell_3 - ell_1) - (ell_4 - ell_2)) if R else None

        # Random-layer baseline I_rand averaged across draws.
        I_rand_vals: list[float] = []
        if R:
            for rand_layers in rand_layer_sets:
                r3_hooks = _ko_late(rand_layers, late_ko_protocol,
                                    rand_donor_means)
                r4_hooks = list(_ko_R()) + list(r3_hooks)
                p_r3 = _final_probs_with_hooks(model, tokens, inst.answer_new_tid,
                                               hooks=r3_hooks)
                p_r4 = _final_probs_with_hooks(model, tokens, inst.answer_new_tid,
                                               hooks=r4_hooks)
                I_rand_vals.append(
                    (_logit_safe(p_r3) - ell_1) - (_logit_safe(p_r4) - (ell_2 or 0))
                )
        I_rand_mean = float(np.mean(I_rand_vals)) if I_rand_vals else None

        D_1 = float(l1[0] - l1[1])
        D_3 = float(l3[0] - l3[1])

        per_inst.append(F3cInstance(
            instance_id=inst.instance_id, fact_id=inst.fact_id,
            param_class=inst.param_class, sigma=sigma, protocol=late_ko_protocol,
            logit_new_1=float(l1[0]), logit_new_2=float(l2[0]),
            logit_new_3=float(l3[0]), logit_new_4=float(l4[0]),
            logit_param_1=float(l1[1]), logit_param_2=float(l2[1]),
            logit_param_3=float(l3[1]), logit_param_4=float(l4[1]),
            D_1=D_1, D_3=D_3, delta_D=D_3 - D_1,
            delta_P_new=p_L_3 - p_L_1, delta_P_param=p_param_3 - p_param_1,
            top1_id_1=top1_1, top1_id_3=top1_3,
            flip_to_new=(top1_1 != inst.answer_new_tid and top1_3 == inst.answer_new_tid),
            flip_to_third=(top1_1 != top1_3
                           and top1_3 != inst.answer_new_tid
                           and top1_3 != inst.a_param_tid),
            I_sigma=I_sigma, I_rand_mean=I_rand_mean,
            floor_flagged=floor_flagged,
            R_pre_L1=R_pre_1, R_pre_L2=R_pre_2,
        ))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Aggregate ───────────────────────────────────────────────────
    dD = np.array([p.delta_D for p in per_inst])
    dP_new = np.array([p.delta_P_new for p in per_inst])
    dP_param = np.array([p.delta_P_param for p in per_inst])

    flip_new_rate = float(np.mean([int(p.flip_to_new) for p in per_inst])) if per_inst else 0.0
    flip_third_rate = float(np.mean([int(p.flip_to_third) for p in per_inst])) if per_inst else 0.0

    success_ceiling = None
    if partition_C_success:
        # Match-stratum B1-success flip rate as ceiling reference (methodology
        # line 605).  Re-run the same protocol on the success pool; we only
        # need argmax at the final layer, so this is cheap.
        flips = []
        for inst in partition_C_success:
            prompt = build_prompt(
                str(inst.row.get("context", "")),
                str(inst.row.get("question", "")),
                template=template,
            )
            tokens = model.to_tokens(prompt, prepend_bos=False)
            late_hooks = _make_sublayer_ablation_hooks(
                L_star, sigma, protocol=late_ko_protocol,
                donor_mean=donor_sublayer if late_ko_protocol == "M" else None,
            )
            l3, top1_3 = _final_logits_with_hooks(model, tokens,
                                                   [inst.answer_new_tid,
                                                    inst.a_param_tid],
                                                   hooks=late_hooks)
            l1, top1_1 = _final_logits_with_hooks(model, tokens,
                                                   [inst.answer_new_tid,
                                                    inst.a_param_tid],
                                                   hooks=[])
            flips.append(int(top1_1 != inst.answer_new_tid
                             and top1_3 == inst.answer_new_tid))
        success_ceiling = float(np.mean(flips)) if flips else None

    _, dD_CI = _bootstrap_paired_mean_CI(dD, rng=rng)
    _, dPnew_CI = _bootstrap_paired_mean_CI(dP_new, rng=rng)
    _, dPp_CI = _bootstrap_paired_mean_CI(dP_param, rng=rng)

    # Chain stats (logit space).
    chain_vals = np.array([p.I_sigma for p in per_inst
                           if p.I_sigma is not None and not p.floor_flagged])
    chain_rand_vals = np.array([p.I_rand_mean for p in per_inst
                                if p.I_rand_mean is not None and not p.floor_flagged])
    chain_mean = None
    chain_CI = None
    chain_minus_rand_CI = None
    chain_rand_mean = None
    if chain_vals.size:
        chain_mean, chain_CI = _bootstrap_paired_mean_CI(chain_vals, rng=rng)
    if chain_rand_vals.size:
        chain_rand_mean = float(chain_rand_vals.mean())
    if chain_vals.size and chain_rand_vals.size:
        n = min(len(chain_vals), len(chain_rand_vals))
        diff = chain_vals[:n] - chain_rand_vals[:n]
        _, chain_minus_rand_CI = _bootstrap_paired_mean_CI(diff, rng=rng)

    floor_share = (float(np.mean([int(p.floor_flagged) for p in per_inst]))
                   if per_inst else 0.0)

    # Vanished-signal classification (methodology lines 619–620).
    vanished = {"chain_broken_upstream": 0.0,
                "well_posed": 0.0,
                "indeterminate": 0.0}
    for p in per_inst:
        if p.R_pre_L1 <= 0.05:  # near-zero floor (proxy for layer-0 prob)
            vanished["chain_broken_upstream"] += 1
        elif p.R_pre_L2 is not None and (p.R_pre_L1 - p.R_pre_L2) >= 0.10:
            vanished["well_posed"] += 1
        else:
            vanished["indeterminate"] += 1
    if per_inst:
        for k in vanished:
            vanished[k] /= len(per_inst)

    # ── Verdicts (mapped via methodology combined matrix lines 678–688) ─
    # Override: ΔD > 0 with CI excluding 0 + (for "full" verdict) flip rate above 0.
    if dD_CI[0] > 0:
        if flip_new_rate > 0:
            verdict_override = "Overridden_confirmed"
        else:
            verdict_override = "Overridden_partial — flip rate ≤ random"
    elif dD_CI[1] < 0:
        verdict_override = "Anti-override"
    else:
        verdict_override = "Overridden_falsified_at_top3"

    if R and chain_CI is not None:
        if chain_CI[0] > 0 and (chain_minus_rand_CI is not None
                                and chain_minus_rand_CI[0] > 0):
            verdict_chain = "Chain_confirmed"
        elif chain_CI[1] < 0:
            verdict_chain = "Anti-chain — compensatory pathway"
        elif floor_share > F3C_FLOOR_FLAG_LIMIT:
            verdict_chain = "Chain_underdetermined — floor flag share > 25%"
        else:
            verdict_chain = "Chain_independent_or_null"
    else:
        verdict_chain = "Chain_not_run (Routed precondition not met or R empty)"

    return F3cResult(
        sigma=sigma, protocol=late_ko_protocol, L_star=list(L_star),
        n_instances=len(per_inst), population_label=population_label,
        per_instance=per_inst,
        override_dD_mean=float(dD.mean()) if dD.size else 0.0,
        override_dD_CI=dD_CI,
        override_dP_new_mean=float(dP_new.mean()) if dP_new.size else 0.0,
        override_dP_new_CI=dPnew_CI,
        override_dP_param_mean=float(dP_param.mean()) if dP_param.size else 0.0,
        override_dP_param_CI=dPp_CI,
        flip_to_new_rate=flip_new_rate,
        flip_to_third_rate=flip_third_rate,
        success_ceiling_flip_rate=success_ceiling,
        chain_I_sigma_mean=chain_mean,
        chain_I_sigma_CI=chain_CI,
        chain_I_rand_mean=chain_rand_mean,
        chain_I_minus_rand_CI=chain_minus_rand_CI,
        floor_flagged_fraction=floor_share,
        vanished_signal_share=vanished,
        verdict_override=verdict_override,
        verdict_chain=verdict_chain,
    )


# ═════════════════════════════════════════════════════════════════════════════
# F3-c Step 4  Encoded Content (raw-W_U projection of sublayer updates)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class F3cContentInstance:
    """Per-instance Step-4 outcome."""
    instance_id: str
    fact_id: str
    sigma: str
    in_top_k_conflict: bool
    in_top_k_closed_book: bool
    in_top_k_random_late: float       # fraction of 20 samples
    in_top_k_random_mid: float
    a_param_argmax_closed_book: bool
    p_a_param_closed_book: float
    dropped_reason: Optional[str]
    dla_used: bool


@dataclass
class F3cContentResult:
    sigma: str
    L_star: list[int]
    n_instances: int
    used_dla: bool

    r_conflict: float
    r_closed_book: float
    r_random_late: float
    r_random_mid: float

    dropped_fraction: float
    dropped_breakdown: dict[str, int]
    closed_book_anchor: float
    sweep_decision: dict[str, str]   # {anchor: verdict}

    verdict: str
    per_instance: list[F3cContentInstance]


def _project_update_raw_wu(
    model: HookedTransformer,
    tokens: torch.Tensor,
    layer: int,
    sigma: str,
    *,
    target_tid: int,
    top_k: int = F3C_TOP_K_VOCAB,
) -> tuple[bool, float]:
    """Project sublayer update ``u^l_σ = x_post − x_mid`` (mlp) /
    ``x_mid − x_pre`` (attn) through raw ``W_U`` and check top-k containment.

    Returns ``(in_top_k, logit_for_target)``.  No layer-norm is applied —
    methodology line 624 mandates "raw ``W_U``, not Tuned Lens".
    """
    pre_buf: list[torch.Tensor] = []
    mid_buf: list[torch.Tensor] = []
    post_buf: list[torch.Tensor] = []

    def _hp(resid, hook):
        pre_buf.append(resid[0, -1].detach().clone())
        return resid

    def _hm(resid, hook):
        mid_buf.append(resid[0, -1].detach().clone())
        return resid

    def _hpost(resid, hook):
        post_buf.append(resid[0, -1].detach().clone())
        return resid

    hooks = [
        (f"blocks.{layer}.hook_resid_pre", _hp),
        (f"blocks.{layer}.hook_resid_mid", _hm),
        (f"blocks.{layer}.hook_resid_post", _hpost),
    ]
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks, prepend_bos=False)

    if not pre_buf:
        return False, 0.0
    if sigma == "attn":
        u = mid_buf[0] - pre_buf[0]
    else:
        u = post_buf[0] - mid_buf[0]

    logits = u @ model.W_U
    if model.b_U is not None and model.b_U.abs().sum() > 0:
        logits = logits + model.b_U
    top_ids = torch.topk(logits, top_k).indices.cpu().tolist()
    in_topk = target_tid in top_ids
    return in_topk, float(logits[target_tid].item())


def _stable_param_check(
    model: HookedTransformer,
    inst: F3PreparedInstance,
    *,
    template: str,
) -> tuple[bool, bool, float]:
    """Check methodology line 651 stability gates on the closed-book pass.

    Returns ``(argmax_match, p_above_floor, p_a_param)``.
    """
    a1_prompt, _ = build_f3_pair_prompts(
        context=str(inst.row.get("context", "")),
        question=str(inst.row.get("question", "")),
        template=template,
    )
    tokens = model.to_tokens(a1_prompt, prepend_bos=False)
    with torch.no_grad():
        logits = model(tokens, prepend_bos=False)[0, -1].float().cpu()
    probs = torch.softmax(logits, dim=-1)
    argmax_match = int(probs.argmax()) == inst.a_param_tid
    p_param = float(probs[inst.a_param_tid])
    p_above = p_param >= F3C_A_PARAM_PROB_FLOOR
    return argmax_match, p_above, p_param


def run_f3c_step4_content(
    model: HookedTransformer,
    test_T: Sequence[F3PreparedInstance],
    *,
    sigma: str,
    L_star: Sequence[int],
    template: str = "phi3",
    n_random_late: int = F3C_N_RANDOM_CONTENT,
    n_random_mid: int = F3C_N_RANDOM_CONTENT,
    rng_seed: int = 42,
    closed_book_min: float = F3C_CLOSED_BOOK_DETECT_MIN,
    verbose: bool = True,
) -> F3cContentResult:
    """F3-c Step 4 — encoded-content verification (methodology L621–688)."""
    if sigma not in ("attn", "mlp"):
        raise ValueError(f"sigma must be 'attn' or 'mlp', got {sigma!r}")
    n_layers = model.cfg.n_layers
    rng = np.random.default_rng(rng_seed)
    lo_late, hi_late = late_window(n_layers)
    lo_mid, hi_mid = mid_window(n_layers)

    rand_late_pool = [l for l in range(lo_late, hi_late + 1) if l not in set(L_star)]
    rand_mid_pool  = list(range(lo_mid, hi_mid + 1))

    # 10-instance pilot on the closed-book detector (line 660).
    pilot = list(test_T[: min(10, len(test_T))])
    pilot_r_closed = []
    for inst in pilot:
        a1_prompt, _ = build_f3_pair_prompts(
            context=str(inst.row.get("context", "")),
            question=str(inst.row.get("question", "")),
            template=template,
        )
        tokens = model.to_tokens(a1_prompt, prepend_bos=False)
        in_any = False
        for l in L_star:
            in_topk, _ = _project_update_raw_wu(
                model, tokens, l, sigma, target_tid=inst.a_param_tid,
            )
            if in_topk:
                in_any = True
                break
        pilot_r_closed.append(int(in_any))
    used_dla = bool(pilot_r_closed) and (np.mean(pilot_r_closed) < closed_book_min)

    per_inst: list[F3cContentInstance] = []
    dropped = {"a_param_not_argmax": 0, "a_param_below_floor": 0}

    for inst in tqdm(test_T, desc=f"F3-c Step4 σ={sigma}",
                     disable=not verbose, unit="inst", dynamic_ncols=True):
        argmax_match, p_above, p_a_param_cb = _stable_param_check(
            model, inst, template=template,
        )
        if not argmax_match:
            dropped["a_param_not_argmax"] += 1
            per_inst.append(F3cContentInstance(
                instance_id=inst.instance_id, fact_id=inst.fact_id, sigma=sigma,
                in_top_k_conflict=False, in_top_k_closed_book=False,
                in_top_k_random_late=0.0, in_top_k_random_mid=0.0,
                a_param_argmax_closed_book=False,
                p_a_param_closed_book=p_a_param_cb,
                dropped_reason="a_param_not_argmax", dla_used=used_dla,
            ))
            continue
        if not p_above:
            dropped["a_param_below_floor"] += 1
            per_inst.append(F3cContentInstance(
                instance_id=inst.instance_id, fact_id=inst.fact_id, sigma=sigma,
                in_top_k_conflict=False, in_top_k_closed_book=False,
                in_top_k_random_late=0.0, in_top_k_random_mid=0.0,
                a_param_argmax_closed_book=True,
                p_a_param_closed_book=p_a_param_cb,
                dropped_reason="a_param_below_floor", dla_used=used_dla,
            ))
            continue

        # Conflict pass (Clean B1 conflict context).
        b1_prompt = build_prompt(
            str(inst.row.get("context", "")),
            str(inst.row.get("question", "")),
            template=template,
        )
        tokens_b1 = model.to_tokens(b1_prompt, prepend_bos=False)
        in_top_b1 = False
        for l in L_star:
            ok, dla = _project_update_raw_wu(
                model, tokens_b1, l, sigma,
                target_tid=inst.a_param_tid,
            )
            if used_dla:
                if dla > 0:
                    in_top_b1 = True
                    break
            elif ok:
                in_top_b1 = True
                break

        # Closed-book pass.
        a1_prompt, _ = build_f3_pair_prompts(
            context=str(inst.row.get("context", "")),
            question=str(inst.row.get("question", "")),
            template=template,
        )
        tokens_a1 = model.to_tokens(a1_prompt, prepend_bos=False)
        in_top_a1 = False
        for l in L_star:
            ok, dla = _project_update_raw_wu(
                model, tokens_a1, l, sigma,
                target_tid=inst.a_param_tid,
            )
            if used_dla:
                if dla > 0:
                    in_top_a1 = True
                    break
            elif ok:
                in_top_a1 = True
                break

        # Random-late / random-mid baselines on the conflict pass.
        def _random_contain(pool: list[int], n_samples: int) -> float:
            if not pool:
                return 0.0
            counts = 0
            for _ in range(n_samples):
                if len(pool) < len(L_star):
                    sample = pool
                else:
                    sample = rng.choice(pool, size=len(L_star), replace=False).tolist()
                in_any = False
                for l in sample:
                    ok, dla = _project_update_raw_wu(
                        model, tokens_b1, int(l), sigma,
                        target_tid=inst.a_param_tid,
                    )
                    if used_dla:
                        if dla > 0:
                            in_any = True
                            break
                    elif ok:
                        in_any = True
                        break
                counts += int(in_any)
            return counts / n_samples

        r_late = _random_contain(rand_late_pool, n_random_late)
        r_mid  = _random_contain(rand_mid_pool,  n_random_mid)

        per_inst.append(F3cContentInstance(
            instance_id=inst.instance_id, fact_id=inst.fact_id, sigma=sigma,
            in_top_k_conflict=in_top_b1, in_top_k_closed_book=in_top_a1,
            in_top_k_random_late=r_late, in_top_k_random_mid=r_mid,
            a_param_argmax_closed_book=True, p_a_param_closed_book=p_a_param_cb,
            dropped_reason=None, dla_used=used_dla,
        ))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    kept = [p for p in per_inst if p.dropped_reason is None]
    r_conf = float(np.mean([int(p.in_top_k_conflict) for p in kept])) if kept else 0.0
    r_cb   = float(np.mean([int(p.in_top_k_closed_book) for p in kept])) if kept else 0.0
    r_late = float(np.mean([p.in_top_k_random_late for p in kept])) if kept else 0.0
    r_mid  = float(np.mean([p.in_top_k_random_mid  for p in kept])) if kept else 0.0
    n_total = max(1, len(test_T))
    dropped_share = (len(test_T) - len(kept)) / n_total

    # Decision rule (methodology table lines 681–688) with anchor sweep.
    def _decide(anchor: float) -> str:
        if r_conf < anchor * r_cb:
            return "Content_not_supported"
        if r_conf > r_late and r_conf > r_mid:
            return "Content_confirmed"
        if r_conf <= r_late and r_conf > r_mid:
            return "Content_weakly_supported_distributed_late"
        if r_conf <= r_mid:
            return "Content_not_localised_to_late"
        return "Content_indeterminate"

    sweep = {f"anchor_{a:.2f}": _decide(a) for a in F3C_CLOSED_BOOK_ANCHOR_SWEEP}
    verdict = _decide(F3C_CLOSED_BOOK_ANCHOR)
    if dropped_share > F3C_A_PARAM_DROP_LIMIT:
        verdict = f"{verdict} — restricted A1-parametric subset (dropped>{F3C_A_PARAM_DROP_LIMIT:.0%})"

    return F3cContentResult(
        sigma=sigma, L_star=list(L_star), n_instances=len(per_inst),
        used_dla=used_dla,
        r_conflict=r_conf, r_closed_book=r_cb,
        r_random_late=r_late, r_random_mid=r_mid,
        dropped_fraction=dropped_share, dropped_breakdown=dropped,
        closed_book_anchor=F3C_CLOSED_BOOK_ANCHOR, sweep_decision=sweep,
        verdict=verdict, per_instance=per_inst,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Partition assignment (methodology B1-success Pool Partition section)
# ═════════════════════════════════════════════════════════════════════════════


def partition_b1_success_pool(
    instances: Sequence[F3PreparedInstance],
    *,
    n_partition_A: int = 100,
    seed: int = 42,
) -> dict[str, list[F3PreparedInstance]]:
    """Stratified random split of B1-success temporal instances.

    Partitions A/B/C are *frozen* after F3-a labels trajectory-bearing
    instances (methodology lines 421–423).  Stratification keys:
    ``param_class`` and ``is_f3_trajectory``.  Returns
    ``{"A": [...], "B": [...], "C": [...]}``; the ``A`` budget is up to
    ``n_partition_A`` (fixed by methodology line 419), the remainder is
    split 50/50 into B and C.
    """
    rng = _random.Random(seed)
    pool = [
        i for i in instances
        if i.param_class in ("PARAM_OLD", "PARAM_OTHER", "PARAM_NEW")
        and i.b1_success is True
    ]
    if not pool:
        return {"A": [], "B": [], "C": []}

    strata: dict[tuple[str, Optional[bool]], list[F3PreparedInstance]] = {}
    for inst in pool:
        key = (inst.param_class, inst.is_f3_trajectory)
        strata.setdefault(key, []).append(inst)

    A: list[F3PreparedInstance] = []
    rest: list[F3PreparedInstance] = []
    # Proportionally fill Partition A.
    a_budget_remaining = min(n_partition_A, len(pool))
    for key, members in strata.items():
        share = int(round(len(members) / len(pool) * a_budget_remaining))
        share = min(share, len(members), a_budget_remaining)
        rng.shuffle(members)
        A.extend(members[:share])
        rest.extend(members[share:])
        a_budget_remaining -= share
    # Fill any rounding leftover from `rest`.
    while a_budget_remaining > 0 and rest:
        A.append(rest.pop(0))
        a_budget_remaining -= 1

    rng.shuffle(rest)
    mid = len(rest) // 2
    B, C = rest[:mid], rest[mid:]
    for inst in A:
        inst.partition = "A"
    for inst in B:
        inst.partition = "B"
    for inst in C:
        inst.partition = "C"
    return {"A": A, "B": B, "C": C}


# ═════════════════════════════════════════════════════════════════════════════
# Combined F3 verdict + title-resolution policy (methodology lines 367–376)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class F3Verdict:
    routed: str
    overridden: str
    chain: str
    content: str
    panel_asymmetry: dict[str, Any]
    rho_HT: Optional[float]
    title: str            # one of the 4 resolutions
    notes: list[str]


def _panel_asymmetry_test(
    f3c_by_panel: dict[str, F3cResult],
    f3c_content_by_panel: dict[str, F3cContentResult],
    rng_seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap 95% CI for PARAM_OLD − PARAM_OTHER on Override / Chain / Content."""
    rng = np.random.default_rng(rng_seed)
    out: dict[str, Any] = {}

    def _diff_CI(a: np.ndarray, b: np.ndarray) -> tuple[float, tuple[float, float]]:
        if a.size == 0 or b.size == 0:
            return 0.0, (0.0, 0.0)
        point = float(a.mean() - b.mean())
        samples = np.array([
            a[rng.integers(0, a.size, a.size)].mean()
            - b[rng.integers(0, b.size, b.size)].mean()
            for _ in range(F3B_BOOTSTRAP_N)
        ])
        return point, (float(np.percentile(samples, 2.5)),
                       float(np.percentile(samples, 97.5)))

    old = f3c_by_panel.get("PARAM_OLD")
    other = f3c_by_panel.get("PARAM_OTHER")
    if old and other:
        a = np.array([p.delta_D for p in old.per_instance])
        b = np.array([p.delta_D for p in other.per_instance])
        out["override_dD"], out["override_dD_CI"] = _diff_CI(a, b)
        if old.chain_I_sigma_mean is not None and other.chain_I_sigma_mean is not None:
            a = np.array([p.I_sigma for p in old.per_instance
                          if p.I_sigma is not None and not p.floor_flagged])
            b = np.array([p.I_sigma for p in other.per_instance
                          if p.I_sigma is not None and not p.floor_flagged])
            out["chain_I"], out["chain_I_CI"] = _diff_CI(a, b)
    cold = f3c_content_by_panel.get("PARAM_OLD")
    cother = f3c_content_by_panel.get("PARAM_OTHER")
    if cold and cother:
        a = np.array([int(p.in_top_k_conflict) for p in cold.per_instance
                      if p.dropped_reason is None])
        b = np.array([int(p.in_top_k_conflict) for p in cother.per_instance
                      if p.dropped_reason is None])
        out["content_r_conflict"], out["content_r_conflict_CI"] = _diff_CI(a, b)

    # Verdict mapping (methodology lines 716–721).
    excludes_zero_pos = []
    for k, ck in (("override_dD_CI", "override_dD"),
                  ("chain_I_CI", "chain_I"),
                  ("content_r_conflict_CI", "content_r_conflict")):
        ci = out.get(k)
        v = out.get(ck)
        if ci is not None and v is not None and ci[0] > 0:
            excludes_zero_pos.append(k)
    if excludes_zero_pos:
        out["verdict"] = ("PARAM_OLD_strictly_greater (CI excludes 0 on: "
                          + ", ".join(excludes_zero_pos) + ")")
    elif all(ci is not None and ci[0] <= 0 <= ci[1]
             for ci in (out.get("override_dD_CI"), out.get("chain_I_CI"),
                        out.get("content_r_conflict_CI"))
             if ci is not None):
        out["verdict"] = "PARAM_OLD_approx_PARAM_OTHER"
    else:
        out["verdict"] = "indeterminate"
    return out


def assign_f3_verdict(
    f3a_summary: F3aSummary,
    f3_half: Optional[F3HalfResult],
    f3b: Optional[F3bResult],
    f3c_by_arm_panel: dict[tuple[str, str], F3cResult],          # (sigma, panel)
    f3c_content_by_arm_panel: dict[tuple[str, str], F3cContentResult],
) -> F3Verdict:
    """Combine F3-a/b/c outcomes into the title-resolved F3 verdict.

    Implements the policy at methodology lines 367–376.
    """
    notes: list[str] = []
    rho_HT = f3b.rho_HT if f3b is not None else None

    routed = f3b.routed_verdict if f3b is not None else "not_run"

    # Override / Chain / Content read from the primary substrate arm (ATTN
    # under the line-583 prior; MLP if (Content) on ATTN is null and MLP is
    # confirmed).  For simplicity we pool panels: a verdict-mapped string
    # per arm.
    def _arm_verdicts(arm: str) -> tuple[str, str, str]:
        old = f3c_by_arm_panel.get((arm, "PARAM_OLD"))
        other = f3c_by_arm_panel.get((arm, "PARAM_OTHER"))
        cold = f3c_content_by_arm_panel.get((arm, "PARAM_OLD"))
        cother = f3c_content_by_arm_panel.get((arm, "PARAM_OTHER"))
        if not (old or other):
            return "not_run", "not_run", "not_run"
        verdict_o = (old.verdict_override if old else None) or (
            other.verdict_override if other else "not_run")
        verdict_c = (old.verdict_chain if old else None) or (
            other.verdict_chain if other else "not_run")
        verdict_content = (cold.verdict if cold else None) or (
            cother.verdict if cother else "not_run")
        return verdict_o, verdict_c, verdict_content

    o_a, c_a, content_a = _arm_verdicts("attn")
    o_m, c_m, content_m = _arm_verdicts("mlp")

    # Primary substrate = ATTN (methodology line 583).  Fall back to MLP only
    # if ATTN is null and MLP is positive.
    if "Overridden_confirmed" in o_a:
        overridden, chain, content = o_a, c_a, content_a
        substrate_used = "attn"
    elif "Overridden_confirmed" in o_m:
        overridden, chain, content = o_m, c_m, content_m
        substrate_used = "mlp"
    else:
        overridden, chain, content = o_a, c_a, content_a
        substrate_used = "attn"
    notes.append(f"substrate_used={substrate_used}")

    # Panel asymmetry (methodology lines 716–721).
    pa = {}
    for arm in ("attn", "mlp"):
        by_panel = {p: r for (a, p), r in f3c_by_arm_panel.items() if a == arm}
        by_panel_content = {p: r for (a, p), r in f3c_content_by_arm_panel.items() if a == arm}
        pa[arm] = _panel_asymmetry_test(by_panel, by_panel_content)

    panel_old_gg_other = (
        "PARAM_OLD_strictly_greater" in pa.get(substrate_used, {}).get("verdict", "")
    )
    rho_HT_high = (rho_HT is not None and rho_HT >= F3B_RHO_HIGH)

    # Title resolution (methodology lines 367–376).
    routed_confirmed = "Routed_confirmed" in routed
    overridden_confirmed = "Overridden_confirmed" in overridden
    chain_confirmed = "Chain_confirmed" in chain
    content_confirmed = "Content_confirmed" in content

    if (routed_confirmed and overridden_confirmed
            and chain_confirmed and content_confirmed
            and (panel_old_gg_other or rho_HT_high)):
        title = "Time Routed but Overridden"
    elif routed_confirmed and overridden_confirmed:
        if (not panel_old_gg_other
                and (rho_HT is not None and rho_HT < F3B_RHO_LOW)):
            title = "Parametric Default Overrides Routed Context"
        else:
            title = "Time Routed but Overridden (partial)"
    elif overridden_confirmed and not routed_confirmed:
        title = "Late Parametric Suppression"
    elif not overridden_confirmed:
        title = "F3_not_supported"
    else:
        title = "F3_indeterminate"

    return F3Verdict(
        routed=routed, overridden=overridden, chain=chain, content=content,
        panel_asymmetry=pa, rho_HT=rho_HT, title=title, notes=notes,
    )


# ═════════════════════════════════════════════════════════════════════════════
# JSON serialisation helpers (numpy → list, dataclass → dict)
# ═════════════════════════════════════════════════════════════════════════════


def to_jsonable(obj: Any) -> Any:
    """Recursively coerce numpy / torch / dataclass / set objects to JSON."""
    from dataclasses import is_dataclass, asdict

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, torch.Tensor):
        return obj.detach().float().cpu().tolist()
    return obj


__all__ = [
    # Dataclasses
    "F3PreparedInstance", "F3aTrajectoryResult", "F3aSummary",
    "F3HalfStage", "F3HalfResult", "F3bConditionStats", "F3bResult",
    "LStarResult", "F3cInstance", "F3cResult",
    "F3cContentInstance", "F3cContentResult", "F3Verdict",
    "SublayerProbs",
    # Helpers
    "mid_window", "late_window", "alignment_k",
    "compute_sublayer_probs",
    # IO
    "load_layer3_by_key", "prepare_f3_instances", "build_f3_pair_prompts",
    # Phase entries
    "run_f3a_trajectory", "summarize_f3a_population",
    "run_f3_half_bridge",
    "run_f3b_ablation",
    "run_f3c_step1_l_star",
    "run_f3c_step2_3",
    "run_f3c_step4_content",
    "partition_b1_success_pool",
    "assign_f3_verdict",
    "to_jsonable",
    # Constants
    "F3A_DEFAULT_TAU", "F3A_TAU_SWEEP",
    "F3HALF_TOP_DECILE_FRAC", "F3HALF_PANEL_ASYMMETRY_THR",
    "F3B_RHO_HIGH", "F3B_RHO_LOW", "F3B_BOOTSTRAP_N",
    "F3C_TOP_K_LATE", "F3C_FLOOR_PROB", "F3C_FLOOR_FLAG_LIMIT",
    "F3C_CLOSED_BOOK_ANCHOR", "F3C_TOP_K_VOCAB",
]
