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

from tatm.f2_diagnosis import compute_head_dla
from tatm.logit_lens import LogitTrajectory, run_logit_lens
from tatm.model import (
    _clean_generated,
    _collect_eos_ids,
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
#: F3-a data-driven window (measurable redesign).  On late-crystallization
#: models (Phi-3-mini, MHA) the answer_new rise-then-drop occurs in the late
#: layers (~24-31), NOT the fixed mid-window [L/4, 2L/3]; the fixed window
#: produced the all-zero F3-a result.  We therefore take the peak over ALL
#: layers and define suppression as (peak − final).  ``F3A_DROP_TAU`` is the
#: peak-to-final drop above which answer_new counts as "routed then suppressed".
F3A_DROP_TAU              = 0.10
#: Rank below which answer_new counts as "rank-competitive" (routed) at a layer
#: — calibration-free readout, robust to crystallization (0 = top-1).
F3A_RANK_COMPETITIVE     = 10
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

# ── F3 spine: crystallization-robust head-ablation verdict (Changes 3/5/6) ───
#: Candidate override-head pool = layers ``l > 2L/3`` (late attention pool).
F3_HEAD_POOL_FRAC          = (2, 3)
#: Default number of cohort-top-k DLA heads ablated on the spine.
F3_HEAD_TOPK_DEFAULT       = 4
#: Power floor on the OVERRIDE claim (clean F3 cohort) — Finding 9.3 / Change 5.
F3_CLEAN_F3_MIN            = 15
#: ``lens_decodable_fraction`` below this ⇒ ``lens_na`` (trajectory metrics +
#: appendix span-knockout suppressed; the causal head-ablation spine still runs).
F3_LENS_NA_FRACTION        = 0.10
#: Per-instance lens-decodability: answer_new becomes rank-competitive at a
#: layer strictly before the final layer (calibration-free, crystallization-aware).
#: New one-directional, timeline-confirmed param taxonomy (Finding 0 / Change 0).
PARAM_CLASSES              = ("TEMPORAL_STALE_CONFIRMED", "PARAM_AMBIGUOUS", "PARAM_NEW")

# ═════════════════════════════════════════════════════════════════════════════
# Prepared instance + Layer-3 / Layer-4 IO
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class F3PreparedInstance:
    """B1 instance enriched with the F3 fields required by methodology L364+.

    ``a_param`` is **the A1(t_new) parametric answer** (methodology line 365).
    ``param_class`` is one of (Finding 0 / Change 0 — one-directional taxonomy):

    * ``"PARAM_NEW"`` — ``a_param == answer_new`` (first-token negative control).
    * ``"TEMPORAL_STALE_CONFIRMED"`` — ``a_param`` matches the recorded
      ``answer_old`` OR any pre-``t_new`` timeline primary object; provably
      once-true.  Reported as a one-directional LOWER BOUND only.
    * ``"PARAM_AMBIGUOUS"`` — neither new nor confirmed-stale (unrecorded-stale
      OR hallucination).  NEVER used as a not-stale control.

    ``stale_exact`` is True only when ``a_param`` matched the exact recorded
    ``answer_old`` (kept for transparency, not for any contrast).
    """

    instance_id: str
    fact_id: str
    row: dict                       # full Layer-2 record
    answer_new: str
    answer_old: str
    a_param: str
    answer_new_tid: int
    a_param_tid: int
    param_class: str                # TEMPORAL_STALE_CONFIRMED / PARAM_AMBIGUOUS / PARAM_NEW
    stale_exact: bool = False       # matched the exact recorded answer_old

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
    *,
    stale_objects: Optional[Sequence[str]] = None,
) -> tuple[str, bool]:
    """One-directional, timeline-confirmed temporal-stale classification.

    Returns ``(param_class, stale_exact)``.  The contaminated
    PARAM_OLD-vs-PARAM_OTHER contrast is retired (Finding 0): only the
    positive, provably-once-true class is asserted, as a LOWER BOUND.
    Incompleteness of the DB / timeline can only cause misses (low recall),
    never false confirmations (precision is safe by construction).
    """
    if check_match(a_param, answer_new):
        return "PARAM_NEW", False
    if answer_old and check_match(a_param, answer_old):
        return "TEMPORAL_STALE_CONFIRMED", True
    if stale_objects:
        for obj in stale_objects:
            if obj and check_match(a_param, obj):
                return "TEMPORAL_STALE_CONFIRMED", False
    return "PARAM_AMBIGUOUS", False


def load_timelines(path: str) -> dict[str, Any]:
    """Load ``{fact_id: FactTimeline}`` from a Layer-1 timeline JSONL.

    Used by :func:`prepare_f3_instances` to build the confirmed-stale set
    (Change 0).  Returns an empty dict on any load failure so the caller
    falls back to single-``answer_old`` confirmation.
    """
    import json

    try:
        from fact_timeline.models import FactTimeline
    except Exception as exc:  # noqa: BLE001
        print(f"  [F3] could not import FactTimeline ({exc}); "
              "confirmed-stale falls back to single answer_old.")
        return {}
    out: dict[str, Any] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                tl = FactTimeline.from_dict(json.loads(line))
            except Exception:  # noqa: BLE001 — skip malformed rows
                continue
            if tl.fact_id:
                out[tl.fact_id] = tl
    return out


def _confirmed_stale_objects(timeline: Any, t_new: Any, answer_new: str) -> list[str]:
    """Primary object per year strictly before ``t_new``, minus answer_new.

    Uses ``primary_object_for_year`` (not all ``objects``) to keep precision
    high (Change 0): a match proves the value was once true; spurious
    alias / co-listed matches are avoided.
    """
    if timeline is None or t_new is None:
        return []
    try:
        tnew_int = int(t_new)
    except (TypeError, ValueError):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for s in getattr(timeline, "states", []):
        try:
            year = int(s.year)
        except (TypeError, ValueError):
            continue
        if year >= tnew_int:
            continue
        obj = timeline.primary_object_for_year(year)
        if obj and obj not in seen:
            seen.add(obj)
            out.append(obj)
    return [o for o in out if not check_match(o, answer_new)]


def prepare_f3_instances(
    model: HookedTransformer,
    b1_rows: list[dict],
    layer3_by_id: dict[str, dict],
    layer3_by_key: dict[tuple, dict],
    *,
    timelines: Optional[dict[str, Any]] = None,
) -> list[F3PreparedInstance]:
    """Join Layer-2 B1 rows with Layer-3 parametric answers.

    When ``timelines`` is provided (Change 0) the parametric class uses the
    one-directional, timeline-confirmed temporal-stale lower bound; otherwise
    confirmation falls back to the single recorded ``answer_old`` (a valid,
    smaller lower bound; backwards-compatible).
    """
    prepared: list[F3PreparedInstance] = []
    skipped = {"no_layer3": 0, "bad_token": 0, "duplicate_token": 0}
    n_missing_timeline = 0
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
        fid = str(row.get("fact_id", ""))
        timeline = timelines.get(fid) if timelines else None
        if timelines is not None and timeline is None:
            n_missing_timeline += 1
        stale_objects = _confirmed_stale_objects(
            timeline, row.get("t_new"), answer_new,
        )
        param_class, stale_exact = _classify_param(
            a_param, answer_new, answer_old, stale_objects=stale_objects,
        )

        prepared.append(F3PreparedInstance(
            instance_id=iid,
            fact_id=fid,
            row=row,
            answer_new=answer_new,
            answer_old=answer_old,
            a_param=a_param,
            answer_new_tid=new_tid,
            a_param_tid=a_param_tid,
            param_class=param_class,
            stale_exact=stale_exact,
        ))
    if any(skipped.values()):
        parts = [f"{k}={v}" for k, v in skipped.items() if v]
        print(f"  [F3] prepare skipped: {', '.join(parts)}")
    if timelines is not None and n_missing_timeline:
        print(f"  [F3] timeline missing for {n_missing_timeline} fact(s); "
              "confirmed-stale fell back to single answer_old for those rows.")
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
    is_f3_trajectory: bool             # at default τ (legacy mid-window)
    is_f3_trajectory_sweep: dict       # {τ: bool} across F3A_TAU_SWEEP

    late_drop: float                   # P^{l_peak}(new) − min_{l > l_peak} P^l(new)

    # Sublayer Δ stacks (last token) for ATTN + MLP, computed via raw lens.
    delta_attn_new: np.ndarray = field(repr=False)
    delta_mlp_new:  np.ndarray = field(repr=False)
    delta_attn_param: np.ndarray = field(repr=False)
    delta_mlp_param:  np.ndarray = field(repr=False)

    # ── Data-driven window (measurable redesign) ──────────────────────────────
    # Peak over ALL layers + peak-to-final drop.  This captures the late
    # rise-then-drop the fixed mid-window misses on crystallization models.
    peak_layer_all: int = -1           # data-driven L* (argmax over all layers)
    p_new_peak_all: float = 0.0
    suppression_drop: float = 0.0      # p_new_peak_all − p_new_final
    is_suppression: bool = False       # suppression_drop > F3A_DROP_TAU
    # Rank-based readout (calibration-free).  -1 = answer_new never competitive.
    first_rank_competitive_layer: int = -1
    rank_new_final: int = -1
    # Behavioral anchor (filled from b1_outputs_param; the F3 classifier).
    b1_outputs_param: Optional[bool] = None
    # Timeline-confirmed exact-stale sub-flag (Change 0; transparency only).
    stale_exact: bool = False


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


def peak_over_all_layers(probs: np.ndarray) -> tuple[float, int]:
    """Data-driven peak: ``(max_l P^l, argmax_l)`` over **all** layers.

    Replaces the fixed mid-window peak (methodology F3-a measurable redesign).
    On late-crystallization models the rise occurs in late layers, so anchoring
    the peak to a fixed mid-window misses it entirely.
    """
    if probs.size == 0:
        return 0.0, 0
    arg = int(np.argmax(probs))
    return float(probs[arg]), arg


def suppression_drop(probs: np.ndarray) -> tuple[float, int]:
    """Return ``(peak − final, peak_layer)`` over all layers.

    This is the F3 "routed then overridden" signal: answer_new reaches a peak
    at some (data-driven) layer L*, then is suppressed by the final layer.
    """
    peak_val, peak_layer = peak_over_all_layers(probs)
    drop = float(peak_val - probs[-1]) if probs.size else 0.0
    return drop, peak_layer


def first_rank_competitive_layer(
    ranks: np.ndarray | None,
    *,
    rank_thr: int = F3A_RANK_COMPETITIVE,
) -> int:
    """First layer where ``answer_new`` enters the top-``rank_thr`` (-1 if never).

    Calibration-free "routed" detector: if answer_new never becomes
    rank-competitive at any layer it was not routed into the readout at all.
    """
    if ranks is None or len(ranks) == 0:
        return -1
    hits = np.nonzero(np.asarray(ranks) < rank_thr)[0]
    return int(hits[0]) if hits.size else -1


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

        # Data-driven window: peak over ALL layers + peak-to-final drop.
        drop_all, peak_layer_all = suppression_drop(p_new)
        ranks_new = getattr(traj, "ranks_new", None)
        first_rc = first_rank_competitive_layer(ranks_new)
        rank_new_final = int(ranks_new[-1]) if ranks_new is not None else -1

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
            peak_layer_all=peak_layer_all,
            p_new_peak_all=float(p_new[peak_layer_all]),
            suppression_drop=drop_all,
            is_suppression=bool(drop_all > F3A_DROP_TAU),
            first_rank_competitive_layer=first_rc,
            rank_new_final=rank_new_final,
            b1_outputs_param=inst.b1_outputs_param,
            stale_exact=inst.stale_exact,
        ))

        bar.set_postfix(traj=is_f3, ok=len(results))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    bar.close()
    return results


# ── F3 causal-test cohort gate ──────────────────────────────────────────────
# The population-level F1/F2/F3/MIXED *hard taxonomy* (the old
# ``classify_failure_mode`` 4-way labeller + ``summarize_failure_modes``
# distribution / ``f3a_failure_modes.json``) has been RETIRED. Single-label
# classification forces mutually-exclusive buckets that the mechanisms do not
# obey (competition of mechanisms / superposition); the graded
# mechanism-propensity decomposition (methodology Part III +
# ``scripts/compute_mechanism_propensity.py``) replaces it. The only thing the
# pipeline still needs is the binary F3-cohort gate below, used to select the
# instances the head-ablation *causal* test runs on.


def is_f3_cohort(r: "F3aTrajectoryResult") -> bool:
    """F3 causal-test cohort membership (narrowed from the retired classifier).

    True iff the instance is a genuine "Routed but Overridden" candidate: a
    B1-failure where the model emitted ``a_param`` AND ``answer_new`` was
    genuinely suppressed (``rank_new_final != 0``). The rank guard excludes
    PARAM_NEW first-token collisions (``answer_new`` actually won, rank 0) that
    would otherwise inflate the cohort (Finding 1 / Change 1). The trajectory
    (``is_suppression`` / ``suppression_drop``) is the mechanistic *explanation*
    of F3, never the gate. Used only to select the head-ablation cohort and the
    confirmed-stale lower bound.
    """
    if r.b1_success:
        return False
    return bool(r.b1_outputs_param) and r.rank_new_final != 0


# ── F3 causal main line: late-window SPAN intervention ──────────────────────


@dataclass
class F3InterventionResult:
    """One late-window span-intervention trial (methodology F3 causal leg)."""
    instance_id: str
    fact_id: str
    param_class: str
    variant: str
    l_star: int
    span_lo: int
    span_hi: int
    baseline_gen: str
    intervened_gen: str
    random_gen: str
    recovered: bool          # intervened matches answer_new AND baseline did not
    random_recovered: Optional[bool]  # control; None when no disjoint span exists


@dataclass
class F3InterventionSummary:
    variant: str
    n: int
    n_control: int           # instances with a valid disjoint random control
    recovery_rate: float
    random_recovery_rate: float  # NaN when no instance had a disjoint control
    delta: float             # recovery_rate − random_recovery_rate (NaN if no control)


# Span-intervention variants (methodology F3 causal main line).  All operate on
# the data-driven span (L*, final]: the layers between the answer_new peak and
# the output, where the suppression occurs.
#   peak_freeze    : zero attn_out AND mlp_out in the span -> residual frozen at
#                    its L* state (clean reading of "freeze L* forward to final").
#   mlp_disinhibit : zero mlp_out only (CoRect-style late-FFN disinhibition).
#   attn_ko        : zero attn_out only (late-attention knockout).
F3_INTERVENTION_VARIANTS = ("peak_freeze", "mlp_disinhibit", "attn_ko")


def _zero_hook(act: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
    return torch.zeros_like(act)


def _span_hooks(variant: str, span: Sequence[int]) -> list[tuple[str, Callable]]:
    hooks: list[tuple[str, Callable]] = []
    for l in span:
        if variant in ("peak_freeze", "attn_ko"):
            hooks.append((f"blocks.{l}.hook_attn_out", _zero_hook))
        if variant in ("peak_freeze", "mlp_disinhibit"):
            hooks.append((f"blocks.{l}.hook_mlp_out", _zero_hook))
    return hooks


def _generate_with_hooks(
    model: HookedTransformer,
    prompt: str,
    fwd_hooks: list[tuple[str, Callable]],
    *,
    max_new_tokens: int = 24,
) -> str:
    """Greedy decode under persistent forward hooks (mirrors model.generate_answer)."""
    tok = model.tokenizer
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    current_ids = enc["input_ids"].to(model.cfg.device)
    eos_ids = set(_collect_eos_ids(tok))
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if fwd_hooks:
                logits = model.run_with_hooks(
                    current_ids, fwd_hooks=fwd_hooks, prepend_bos=False,
                )
            else:
                logits = model(current_ids, prepend_bos=False)
            next_id = int(logits[0, -1, :].argmax())
            del logits
            if next_id in eos_ids:
                break
            generated.append(next_id)
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_id]], device=current_ids.device)],
                dim=1,
            )
    raw = tok.decode(generated, skip_special_tokens=True)
    return _clean_generated(raw)


def run_f3_late_window_intervention(
    model: HookedTransformer,
    instances: Sequence[F3PreparedInstance],
    *,
    variants: Sequence[str] = F3_INTERVENTION_VARIANTS,
    template: str = "phi3",
    max_new_tokens: int = 24,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[list[F3InterventionResult], list[F3InterventionSummary]]:
    """F3 causal main line: late-window SPAN intervention on the F3 cohort.

    Cohort = behavioral F3 (``b1_outputs_param == True``).  For each instance:

    1. Find the data-driven peak layer ``L*`` of ``P^l(answer_new)`` (the layer
       where the routed answer is strongest before being suppressed).
    2. Apply the span intervention over ``(L*, final]`` and greedily regenerate;
       record whether the generation now matches ``answer_new`` (recovery).
    3. Apply the **same-size span** at a random early/mid location as the
       baseline control, regenerate, record recovery.

    A causal F3 confirmation = recovery_rate >> random_recovery_rate.  SPAN
    (not single-layer) intervention is used because suppression may be
    distributed across late layers (MechLens, Gini < 0.015).
    """
    rng = _random.Random(seed)
    cohort = [i for i in instances if i.b1_outputs_param]
    n_layers = model.cfg.n_layers
    results: list[F3InterventionResult] = []

    bar = tqdm(cohort, desc="F3 late-window intervention",
               disable=not verbose, unit="inst", dynamic_ncols=True)
    for inst in bar:
        row = inst.row
        prompt = build_prompt(
            str(row.get("context", "")), str(row.get("question", "")),
            template=template,
        )
        tokens = model.to_tokens(prompt, prepend_bos=False)
        traj = run_logit_lens(
            model, tokens, inst.answer_new_tid, inst.a_param_tid, lens_kind="raw",
        )
        p_new = np.asarray(traj.probs_new, dtype=float)
        _, l_star = peak_over_all_layers(p_new)
        if l_star >= n_layers - 1:
            continue  # no span to intervene on
        span = list(range(l_star + 1, n_layers))
        span_len = len(span)
        # Disjoint random control of equal length in the early/mid region
        # [0, l_star].  If there is no room for a span that does NOT overlap the
        # treatment span, skip the control entirely (random_recovered=None)
        # rather than reusing the treatment span — equating them would force
        # delta=0 and contaminate the random baseline.
        if l_star >= span_len:
            rand_start = rng.randint(0, l_star - span_len + 1)
            rand_span: list[int] | None = list(range(rand_start, rand_start + span_len))
        else:
            rand_span = None

        baseline_gen = _generate_with_hooks(
            model, prompt, [], max_new_tokens=max_new_tokens,
        )
        # A baseline that already emits answer_new is not a genuine F3 override
        # and must not be counted as a "recovery".
        baseline_is_new = check_match(baseline_gen, inst.answer_new)
        for variant in variants:
            interv_gen = _generate_with_hooks(
                model, prompt, _span_hooks(variant, span),
                max_new_tokens=max_new_tokens,
            )
            if rand_span is not None:
                rand_gen = _generate_with_hooks(
                    model, prompt, _span_hooks(variant, rand_span),
                    max_new_tokens=max_new_tokens,
                )
                random_recovered: Optional[bool] = check_match(rand_gen, inst.answer_new)
            else:
                rand_gen = ""
                random_recovered = None
            results.append(F3InterventionResult(
                instance_id=inst.instance_id,
                fact_id=inst.fact_id,
                param_class=inst.param_class,
                variant=variant,
                l_star=l_star,
                span_lo=span[0],
                span_hi=span[-1],
                baseline_gen=baseline_gen,
                intervened_gen=interv_gen,
                random_gen=rand_gen,
                recovered=(check_match(interv_gen, inst.answer_new) and not baseline_is_new),
                random_recovered=random_recovered,
            ))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    bar.close()

    summaries: list[F3InterventionSummary] = []
    for variant in variants:
        rows = [r for r in results if r.variant == variant]
        if not rows:
            continue
        rec = float(np.mean([r.recovered for r in rows]))
        ctrl = [r.random_recovered for r in rows if r.random_recovered is not None]
        rnd = float(np.mean(ctrl)) if ctrl else float("nan")
        delta = (rec - rnd) if ctrl else float("nan")
        summaries.append(F3InterventionSummary(
            variant=variant, n=len(rows), n_control=len(ctrl),
            recovery_rate=rec, random_recovery_rate=rnd, delta=delta,
        ))
    return results, summaries


# ── F3 causal SPINE: DLA-localized late-head ablation (Changes 2, 3, 10) ────


@dataclass
class F3HeadAblationResult:
    """Crystallization-robust F3 verdict carrier (Findings 9.1, 10).

    Ablate the cohort-top-k DLA-localized late override heads and read the
    GRADED change in the final logit diff ``logit(a_param) − logit(answer_new)``
    on the clean F3 cohort vs a B1-success population null.  A positive
    per-instance ``effect`` = ablation reduced the parametric advantage;
    ``delta = mean(effect_failure) − mean(effect_success)`` with a CI excluding
    0 ⇒ ``F3_supported`` (Finding 12.2).
    """
    n_clean_f3: int
    n_success_null: int
    head_pool_size: int
    top_k: int
    top_k_heads: list[tuple[int, int]]
    per_head_dla: dict[str, float]          # cohort-mean DLA per "l.h" (top-k)
    effect_failure_mean: float
    effect_success_mean: float
    delta: float
    delta_CI: tuple[float, float]
    failure_effects: list[float] = field(repr=False, default_factory=list)
    success_effects: list[float] = field(repr=False, default_factory=list)
    notes: list[str] = field(default_factory=list)


def _late_head_pool(model: HookedTransformer) -> list[tuple[int, int]]:
    """Candidate override-head pool: all heads in layers ``l > 2L/3``."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    lo = (n_layers * F3_HEAD_POOL_FRAC[0]) // F3_HEAD_POOL_FRAC[1]
    return [(l, h) for l in range(lo + 1, n_layers) for h in range(n_heads)]


def _ablation_effect(
    model: HookedTransformer,
    inst: F3PreparedInstance,
    heads_by_layer: dict[int, list[int]],
    *,
    template: str,
) -> Optional[float]:
    """Per-instance graded effect = clean_logit_diff − ablated_logit_diff.

    ``logit_diff = logit(a_param) − logit(answer_new)``.  Positive effect ⇒
    ablating the heads reduced the parametric advantage (override evidence).
    """
    prompt = build_prompt(
        str(inst.row.get("context", "")), str(inst.row.get("question", "")),
        template=template,
    )
    tokens = model.to_tokens(prompt, prepend_bos=False)
    targets = [inst.a_param_tid, inst.answer_new_tid]
    clean_logits, _ = _final_logits_with_hooks(model, tokens, targets, hooks=[])
    abl_hooks = _make_head_ablation_hooks(heads_by_layer, protocol="Z")
    abl_logits, _ = _final_logits_with_hooks(model, tokens, targets, hooks=abl_hooks)
    clean_diff = float(clean_logits[0] - clean_logits[1])
    abl_diff = float(abl_logits[0] - abl_logits[1])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return clean_diff - abl_diff


def _diff_means_CI(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = F3B_BOOTSTRAP_N,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, tuple[float, float]]:
    """Bootstrap CI for ``mean(a) − mean(b)`` (two independent samples)."""
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return 0.0, (0.0, 0.0)
    point = float(a.mean() - (b.mean() if b.size else 0.0))
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        am = a[rng.integers(0, a.size, a.size)].mean()
        bm = b[rng.integers(0, b.size, b.size)].mean() if b.size else 0.0
        samples[i] = am - bm
    return point, (float(np.percentile(samples, 2.5)),
                   float(np.percentile(samples, 97.5)))


def run_f3_head_ablation(
    model: HookedTransformer,
    clean_f3: Sequence[F3PreparedInstance],
    success_null: Sequence[F3PreparedInstance],
    *,
    template: str = "phi3",
    top_k: int = F3_HEAD_TOPK_DEFAULT,
    seed: int = 0,
    verbose: bool = True,
) -> F3HeadAblationResult:
    """F3 spine: localize late override heads via DLA, then ablate + measure Δ.

    1. Compute per-head DLA onto ``(a_param − answer_new)`` over the late pool
       (``l > 2L/3``) for every clean-F3 instance; take the cohort-level top-k
       ``(l,h)`` by mean DLA (localization only — NOT the verdict, Finding 9.1).
    2. Ablate exactly those heads (zero ``hook_z`` at the last position) on every
       clean-F3 AND B1-success-null instance; read the graded logit-diff effect.
    3. Verdict effect ``delta = mean(effect_failure) − mean(effect_success)``
       with a bootstrap CI (between-population, same perturbation — subsumes the
       Finding 2 early/mid-null fix).
    """
    notes: list[str] = []
    pool = _late_head_pool(model)
    if not clean_f3:
        notes.append("no_clean_f3")
        return F3HeadAblationResult(
            n_clean_f3=0, n_success_null=len(success_null),
            head_pool_size=len(pool), top_k=top_k, top_k_heads=[],
            per_head_dla={}, effect_failure_mean=0.0, effect_success_mean=0.0,
            delta=0.0, delta_CI=(0.0, 0.0), notes=notes,
        )

    # ── 1. DLA localization (cohort-mean over the clean F3 cohort) ──────────
    dla_sums: dict[str, float] = {}
    dla_counts: dict[str, int] = {}
    bar = tqdm(clean_f3, desc="F3 spine DLA localize", disable=not verbose,
               unit="inst", dynamic_ncols=True)
    for inst in bar:
        prompt = build_prompt(
            str(inst.row.get("context", "")), str(inst.row.get("question", "")),
            template=template,
        )
        tokens = model.to_tokens(prompt, prepend_bos=False)
        per_head, _ = compute_head_dla(
            model, tokens, pool, inst.a_param_tid, inst.answer_new_tid,
        )
        for key, val in per_head.items():
            dla_sums[key] = dla_sums.get(key, 0.0) + val
            dla_counts[key] = dla_counts.get(key, 0) + 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    bar.close()

    if not dla_sums:
        notes.append("dla_failed_on_all_instances")
        return F3HeadAblationResult(
            n_clean_f3=len(clean_f3), n_success_null=len(success_null),
            head_pool_size=len(pool), top_k=top_k, top_k_heads=[],
            per_head_dla={}, effect_failure_mean=0.0, effect_success_mean=0.0,
            delta=0.0, delta_CI=(0.0, 0.0), notes=notes,
        )

    mean_dla = {k: dla_sums[k] / max(1, dla_counts[k]) for k in dla_sums}
    ranked = sorted(mean_dla.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:top_k]
    top_k_heads = [(int(k.split(".")[0]), int(k.split(".")[1])) for k, _ in top]
    per_head_dla = {k: v for k, v in top}
    heads_by_layer: dict[int, list[int]] = {}
    for (l, h) in top_k_heads:
        heads_by_layer.setdefault(l, []).append(h)

    # ── 2. Ablate the SAME heads on both arms; read graded logit-diff ──────
    failure_effects: list[float] = []
    for inst in tqdm(clean_f3, desc="F3 spine ablate (failure)",
                     disable=not verbose, unit="inst", dynamic_ncols=True):
        eff = _ablation_effect(model, inst, heads_by_layer, template=template)
        if eff is not None:
            failure_effects.append(eff)
    success_effects: list[float] = []
    for inst in tqdm(success_null, desc="F3 spine ablate (B1-success null)",
                     disable=not verbose, unit="inst", dynamic_ncols=True):
        eff = _ablation_effect(model, inst, heads_by_layer, template=template)
        if eff is not None:
            success_effects.append(eff)

    if not success_effects:
        notes.append("empty_success_null — Δ falls back to failure-only mean (no between-pop CI)")
    fa = np.asarray(failure_effects, dtype=float)
    sa = np.asarray(success_effects, dtype=float)
    delta, ci = _diff_means_CI(fa, sa, rng=np.random.default_rng(seed))

    return F3HeadAblationResult(
        n_clean_f3=len(clean_f3),
        n_success_null=len(success_null),
        head_pool_size=len(pool),
        top_k=top_k,
        top_k_heads=top_k_heads,
        per_head_dla=per_head_dla,
        effect_failure_mean=float(fa.mean()) if fa.size else 0.0,
        effect_success_mean=float(sa.mean()) if sa.size else 0.0,
        delta=delta,
        delta_CI=ci,
        failure_effects=failure_effects,
        success_effects=success_effects,
        notes=notes,
    )


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


def _is_lens_decodable(r: "F3aTrajectoryResult") -> bool:
    """Per-instance lens decodability (Change 6 / Finding 3).

    The raw logit lens can read answer_new's trajectory iff it becomes
    rank-competitive at a layer strictly before the final layer.  On
    high-crystallization models the answer only resolves at the very last
    layer, so this fraction collapses (Phi-3 ~6%) and ``lens_na`` trips.
    """
    frc = r.first_rank_competitive_layer
    return frc >= 0 and frc < (r.n_layers - 1)


@dataclass
class F3aSummary:
    """Population-level F3-a summary (lean, descriptive — Change 6 / Finding 0-4).

    The contaminated PARAM_OLD-vs-PARAM_OTHER positive-control lattice is
    retired (Finding 0/4).  What remains is descriptive (per-class trajectory /
    suppression) plus two spine-relevant scalars: ``lens_decodable_fraction``
    (gates the lens trajectory readout) and ``confirmed_stale_fraction`` (a
    one-directional LOWER BOUND over the clean F3 cohort).
    """
    tau: float
    n_layers: int
    mid_window: tuple[int, int]
    late_window: tuple[int, int]
    counts_by_class: dict[str, int]
    f3_traj_rate: dict[str, float]                # legacy mid-window (≈0 on late-crystallization)
    median_late_drop: dict[str, float]
    suppression_mean_by_class: dict[str, float]   # descriptive (Finding 4)
    lens_decodable_fraction: float                # Change 6 / Finding 3
    confirmed_stale_fraction: float               # LOWER BOUND over clean F3
    n_clean_f3: int                               # rank-guarded override cohort
    sweep_traj_rate: dict[str, dict[str, float]]  # {tau: {class: rate}}
    # Data-driven readouts (late-crystallization-robust; what the figure should
    # surface instead of the fixed mid-window f3_traj_rate):
    #   suppression_rate     — fraction with is_suppression (peak-over-all → final drop)
    #   rank_competitive_rate — fraction where answer_new ever became rank-competitive
    suppression_rate: dict[str, float] = field(default_factory=dict)
    rank_competitive_rate: dict[str, float] = field(default_factory=dict)


def summarize_f3a_population(
    results: Sequence[F3aTrajectoryResult],
    *,
    tau: float = F3A_DEFAULT_TAU,
    rank_thr: int = F3A_RANK_COMPETITIVE,
) -> F3aSummary:
    """Aggregate F3-a results into the lean descriptive summary (Change 6).

    No positive-control lattice, no OLD-vs-OTHER asymmetry, no Cramér's V —
    those relied on the contaminated PARAM_OTHER control (Finding 0/4).
    Trajectory / suppression are reported per class descriptively; the two
    spine scalars are ``lens_decodable_fraction`` and the one-directional
    ``confirmed_stale_fraction`` over the clean F3 cohort.
    """
    classes = list(PARAM_CLASSES)
    if not results:
        empty = {c: 0 for c in classes}
        return F3aSummary(
            tau=tau, n_layers=0, mid_window=(0, 0), late_window=(0, 0),
            counts_by_class=empty,
            f3_traj_rate={c: 0.0 for c in classes},
            median_late_drop={c: 0.0 for c in classes},
            suppression_mean_by_class={c: 0.0 for c in classes},
            lens_decodable_fraction=0.0,
            confirmed_stale_fraction=0.0,
            n_clean_f3=0,
            sweep_traj_rate={f"tau_{t:.2f}": {c: 0.0 for c in classes}
                             for t in F3A_TAU_SWEEP},
            suppression_rate={c: 0.0 for c in classes},
            rank_competitive_rate={c: 0.0 for c in classes},
        )

    n_layers = results[0].n_layers
    by_class: dict[str, list[F3aTrajectoryResult]] = {c: [] for c in classes}
    for r in results:
        by_class.setdefault(r.param_class, []).append(r)

    counts = {c: len(by_class.get(c, [])) for c in classes}
    traj_rate = {
        c: (float(np.mean([int(r.is_f3_trajectory) for r in by_class.get(c, [])]))
            if by_class.get(c) else 0.0)
        for c in classes
    }
    median_drop = {
        c: (float(np.median([r.late_drop for r in by_class.get(c, [])
                             if r.is_f3_trajectory]))
            if any(r.is_f3_trajectory for r in by_class.get(c, [])) else 0.0)
        for c in classes
    }
    suppression_mean = {
        c: (float(np.mean([r.suppression_drop for r in by_class.get(c, [])]))
            if by_class.get(c) else 0.0)
        for c in classes
    }
    suppression_rate = {
        c: (float(np.mean([int(r.is_suppression) for r in by_class.get(c, [])]))
            if by_class.get(c) else 0.0)
        for c in classes
    }
    rank_competitive_rate = {
        c: (float(np.mean([int(r.first_rank_competitive_layer >= 0)
                           for r in by_class.get(c, [])]))
            if by_class.get(c) else 0.0)
        for c in classes
    }

    lens_decodable_fraction = float(np.mean([int(_is_lens_decodable(r)) for r in results]))

    # Confirmed-stale LOWER BOUND over the clean (rank-guarded) F3 cohort.
    clean_f3 = [r for r in results if is_f3_cohort(r)]
    confirmed_stale_fraction = (
        float(np.mean([int(r.param_class == "TEMPORAL_STALE_CONFIRMED")
                       for r in clean_f3]))
        if clean_f3 else 0.0
    )

    sweep = {}
    for t in F3A_TAU_SWEEP:
        key = f"tau_{t:.2f}"
        sweep[key] = {
            c: (float(np.mean([int(r.is_f3_trajectory_sweep[key])
                               for r in by_class.get(c, [])]))
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
        suppression_mean_by_class=suppression_mean,
        lens_decodable_fraction=lens_decodable_fraction,
        confirmed_stale_fraction=confirmed_stale_fraction,
        n_clean_f3=len(clean_f3),
        sweep_traj_rate=sweep,
        suppression_rate=suppression_rate,
        rank_competitive_rate=rank_competitive_rate,
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
        if i.param_class in PARAM_CLASSES
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
    """Spine F3 verdict (Change 5 / Finding 12.2).

    Three states only — ``F3_supported`` / ``F3_not_supported`` /
    ``F3_underpowered`` — carried by the head-ablation Δ vs the B1-success
    population null.  The conjunctive routed/overridden/chain/content lattice
    is retired from the spine (``--hardening`` only).
    """
    verdict: str                       # F3_supported / F3_not_supported / F3_underpowered
    title: str                         # human-facing display title
    delta: Optional[float]
    delta_CI: Optional[tuple[float, float]]
    n_clean_f3: int
    lens_na: bool
    lens_decodable_fraction: float
    confirmed_stale_lower_bound: float
    top_k_heads: list[tuple[int, int]]
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
    head_ablation: Optional[F3HeadAblationResult],
    *,
    n_clean_f3: Optional[int] = None,
) -> F3Verdict:
    """Spine F3 verdict: head-ablation Δ vs the B1-success population null.

    Three states (Finding 12.2):

    * ``F3_underpowered`` — clean F3 cohort ``< F3_CLEAN_F3_MIN`` (power floor on
      the OVERRIDE claim, Finding 9.3) or no head-ablation result.
    * ``F3_supported`` — Δ CI excludes 0 on the positive side ⇒ ablating the
      override heads reduced the parametric advantage.  Display title
      "Parametric Override Confirmed".
    * ``F3_not_supported`` — a true falsification (Δ CI includes 0 or is negative).

    ``lens_na`` (``lens_decodable_fraction < F3_LENS_NA_FRACTION``) only
    suppresses the lens-trajectory metrics + appendix span-knockout; it never
    decides the causal verdict.  ``confirmed_stale_fraction`` rides along as a
    one-directional LOWER-BOUND annotation, never a gate.
    """
    notes: list[str] = []
    lens_dec = f3a_summary.lens_decodable_fraction
    lens_na = lens_dec < F3_LENS_NA_FRACTION
    if lens_na:
        notes.append(
            f"lens_na: lens_decodable_fraction={lens_dec:.3f} < {F3_LENS_NA_FRACTION} "
            "— trajectory metrics + appendix span-knockout suppressed; causal spine still runs"
        )
    n = n_clean_f3 if n_clean_f3 is not None else f3a_summary.n_clean_f3
    confirmed_stale = f3a_summary.confirmed_stale_fraction
    notes.append(
        f"confirmed_stale_lower_bound={confirmed_stale:.3f} "
        "(one-directional; not a verdict gate)"
    )

    delta = head_ablation.delta if head_ablation else None
    delta_CI = head_ablation.delta_CI if head_ablation else None
    top_k_heads = head_ablation.top_k_heads if head_ablation else []
    if head_ablation is not None:
        notes.extend(head_ablation.notes)

    if n < F3_CLEAN_F3_MIN or head_ablation is None:
        verdict = "F3_underpowered"
        title = "F3 Underpowered"
        notes.append(
            f"underpowered: n_clean_f3={n} < {F3_CLEAN_F3_MIN} (power floor on override cohort)"
            if head_ablation is not None else "underpowered: no head-ablation result"
        )
    elif delta_CI is not None and delta_CI[0] > 0:
        verdict = "F3_supported"
        title = "Parametric Override Confirmed"
    else:
        verdict = "F3_not_supported"
        title = "F3 Not Supported (falsified)"

    return F3Verdict(
        verdict=verdict,
        title=title,
        delta=delta,
        delta_CI=delta_CI,
        n_clean_f3=n,
        lens_na=lens_na,
        lens_decodable_fraction=lens_dec,
        confirmed_stale_lower_bound=confirmed_stale,
        top_k_heads=top_k_heads,
        notes=notes,
    )


# ═════════════════════════════════════════════════════════════════════════════
# JSON serialisation helpers (numpy → list, dataclass → dict)
# ═════════════════════════════════════════════════════════════════════════════


def to_jsonable(obj: Any) -> Any:
    """Recursively coerce numpy / torch / dataclass / set objects to JSON."""
    from dataclasses import is_dataclass, asdict

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
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
    "F3InterventionResult", "F3InterventionSummary",
    "F3HeadAblationResult",
    # Helpers
    "mid_window", "late_window", "alignment_k",
    "peak_over_all_layers", "suppression_drop", "first_rank_competitive_layer",
    "is_f3_cohort",
    "compute_sublayer_probs",
    # IO
    "load_layer3_by_key", "load_timelines", "prepare_f3_instances",
    "build_f3_pair_prompts",
    # Phase entries
    "run_f3a_trajectory", "summarize_f3a_population",
    "run_f3_head_ablation",
    "run_f3_late_window_intervention",
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
