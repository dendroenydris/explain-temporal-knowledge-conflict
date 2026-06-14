"""F2 Diagnosis — "Time Set but Not Routed".

Three complementary experiments:

F2-a  STR Activation Patching — causal confirmation that temporal heads
      mediate the year routing signal.  Runs on instances where the clean
      run (question year = t_new) produces answer_new and the corrupted
      run (question year → t_old) produces answer_old (pure behavioral
      filter, no parametric-memory restriction).

F2-b  RouteScore — continuous metric for temporal signal attenuation
      between temporal-head layers and the final output, derived from
      Logit Lens trajectories cached by ``logit_lens.run_logit_lens``.

F2-c  B5 vs B6 Behavioral Cross-Analysis — cheapest F2 detector.
      Compares model accuracy when both evidence passages are present
      with years intact (B5) vs stripped (B6), cross-referenced with
      single-evidence B1 results.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from tatm.logit_lens import LogitTrajectory, get_first_answer_token, run_logit_lens
from tatm.model import (
    build_prompt,
    check_match,
    find_year_positions,
    generate_answer,
)

# ═════════════════════════════════════════════════════════════════════════════
# F2-a  STR Activation Patching
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class PatchingResult:
    """Outcome of one STR patching trial.

    Methodology F2-a line 307: "Report logit difference $\\Delta_{\\text{logit}}$
    and **top-1 shift**" — and the headline question line 311 is "Does
    $p(\\text{answer\\_new})$ recover?".  We therefore store the answer
    probabilities alongside the logit diffs **and** pre-compute the
    top-1 shift flag so downstream readers don't have to recompute it.
    """
    instance_id: str
    logit_diff_clean: float       # logit(new) − logit(old), clean run
    logit_diff_corrupted: float   # logit(new) − logit(old), corrupted run
    logit_diff_patched: float     # logit(new) − logit(old), patched run
    recovery_fraction: float      # (patched − corrupted) / (clean − corrupted)

    # Probability-based recovery (methodology line 311: "Does p(answer_new) recover?")
    p_new_clean:     float
    p_new_corrupted: float
    p_new_patched:   float
    p_new_recovery_fraction: float  # (p_patched − p_corrupted)/(p_clean − p_corrupted)

    # Top-1 decoded strings
    top1_clean: str
    top1_corrupted: str
    top1_patched: str
    # Methodology line 307 "top-1 shift": True iff patching flipped the top-1
    # token away from the corrupted-run answer (typically answer_old) — i.e.
    # patch causally moved the model's top-1 toward answer_new.
    top1_shifted_to_new: bool
    top1_flipped_from_corrupted: bool


def _patch_z_hook(
    z: torch.Tensor,
    hook,
    *,
    clean_z: torch.Tensor,
    head_indices: list[int],
    positions: list[int],
) -> torch.Tensor:
    """Replace z[batch, positions, heads, :] with values from *clean_z*."""
    patched = z.clone()
    for pos in positions:
        for h in head_indices:
            patched[:, pos, h, :] = clean_z[:, pos, h, :]
    return patched


def run_str_patching(
    model: HookedTransformer,
    instances: list[dict],
    temporal_heads: list[tuple[int, int]],
    *,
    template: str = "plain",
    verbose: bool = True,
) -> list[PatchingResult]:
    """F2-a: STR Activation Patching on B5 instances.

    For each instance that passes the behavioral filter (clean → answer_new
    AND corrupted → answer_old), patches the ``hook_z`` activations at the
    year-token positions of every temporal head from the clean run into the
    corrupted run and measures logit-difference recovery.

    Position alignment strategy: rather than heuristically locating the
    question region, we diff the two token sequences and patch exactly the
    positions that changed (= the question-year tokens).  If the two
    sequences differ in length we skip the instance.

    Parameters
    ----------
    model : HookedTransformer
    instances : B5-type EvalInstance dicts (must have ``context``,
        ``question``, ``answer_new``, ``answer_old``, ``t_new``, ``t_old``)
    temporal_heads : list of ``(layer, head)`` tuples from EAP-IG / circuit
        discovery
    template : prompt template name
    verbose : show progress bar
    """
    heads_by_layer: dict[int, list[int]] = {}
    for layer, head in temporal_heads:
        heads_by_layer.setdefault(layer, []).append(head)
    patch_layers = sorted(heads_by_layer.keys())
    cache_names = {f"blocks.{l}.attn.hook_z" for l in patch_layers}

    results: list[PatchingResult] = []
    skipped = {"gen_clean": 0, "gen_corr": 0, "len_mismatch": 0,
               "no_diff": 0, "token_id": 0}

    bar = tqdm(instances, desc="F2-a  STR patching", disable=not verbose)

    for inst in bar:
        iid = inst.get("instance_id", "")
        question = inst.get("question", "")
        context = inst.get("context", "")
        answer_new = inst.get("answer_new", "")
        answer_old = inst.get("answer_old", "")
        t_new = inst.get("t_new")
        t_old = inst.get("t_old")

        if t_new is None or t_old is None:
            continue

        # ── Build clean & corrupted prompts ───────────────────────────
        clean_prompt = build_prompt(context, question, template=template)
        corrupted_q = question.replace(str(t_new), str(t_old))
        corrupted_prompt = build_prompt(context, corrupted_q, template=template)

        clean_tokens = model.to_tokens(clean_prompt, prepend_bos=False)
        corrupted_tokens = model.to_tokens(corrupted_prompt, prepend_bos=False)

        # Sequences must be equal length for position-aligned patching
        if clean_tokens.shape != corrupted_tokens.shape:
            skipped["len_mismatch"] += 1
            continue

        # Diff positions = question-year tokens that changed
        diff_pos = (clean_tokens[0] != corrupted_tokens[0]).nonzero(
            as_tuple=True
        )[0].tolist()
        if not diff_pos:
            skipped["no_diff"] += 1
            continue

        # ── Behavioral filter: generate & verify ──────────────────────
        clean_gen = generate_answer(model, clean_prompt)
        if not check_match(clean_gen, answer_new):
            skipped["gen_clean"] += 1
            continue

        corrupted_gen = generate_answer(model, corrupted_prompt)
        if not check_match(corrupted_gen, answer_old):
            skipped["gen_corr"] += 1
            continue

        # Answer token IDs — must be valid and distinct
        new_tid = get_first_answer_token(model, answer_new)
        old_tid = get_first_answer_token(model, answer_old)
        if new_tid < 0 or old_tid < 0 or new_tid == old_tid:
            skipped["token_id"] += 1
            continue

        # ── Clean run: forward + cache z at temporal-head layers ──────
        with torch.no_grad():
            clean_logits, cache = model.run_with_cache(
                clean_tokens,
                names_filter=lambda n: n in cache_names,
                prepend_bos=False,
            )
        clean_last = clean_logits[0, -1].float().cpu()

        # ── Corrupted run (no patching) ───────────────────────────────
        with torch.no_grad():
            corr_last = model(
                corrupted_tokens, prepend_bos=False
            )[0, -1].float().cpu()

        # ── Patched run: replace z at diff positions from clean ───────
        patch_hooks = []
        for layer in patch_layers:
            cached_z = cache[f"blocks.{layer}.attn.hook_z"]
            patch_hooks.append((
                f"blocks.{layer}.attn.hook_z",
                partial(
                    _patch_z_hook,
                    clean_z=cached_z,
                    head_indices=heads_by_layer[layer],
                    positions=diff_pos,
                ),
            ))

        with torch.no_grad():
            patched_last = model.run_with_hooks(
                corrupted_tokens, fwd_hooks=patch_hooks, prepend_bos=False,
            )[0, -1].float().cpu()

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ── Metrics ───────────────────────────────────────────────────
        # Logit-diff recovery (Zhang & Nanda 2024 standard)
        ld_clean = (clean_last[new_tid] - clean_last[old_tid]).item()
        ld_corr  = (corr_last[new_tid]  - corr_last[old_tid]).item()
        ld_patch = (patched_last[new_tid] - patched_last[old_tid]).item()
        denom = ld_clean - ld_corr
        recovery = (ld_patch - ld_corr) / denom if abs(denom) > 1e-6 else 0.0

        # Probability-based recovery (methodology line 311 "Does p(new) recover?")
        p_clean    = float(torch.softmax(clean_last,    dim=-1)[new_tid].item())
        p_corr     = float(torch.softmax(corr_last,     dim=-1)[new_tid].item())
        p_patch    = float(torch.softmax(patched_last,  dim=-1)[new_tid].item())
        p_denom    = p_clean - p_corr
        p_recovery = (p_patch - p_corr) / p_denom if abs(p_denom) > 1e-6 else 0.0

        # Top-1 shift (methodology line 307)
        clean_top1_id   = int(clean_last.argmax())
        corr_top1_id    = int(corr_last.argmax())
        patched_top1_id = int(patched_last.argmax())
        top1_clean_s    = model.tokenizer.decode([clean_top1_id])
        top1_corr_s     = model.tokenizer.decode([corr_top1_id])
        top1_patched_s  = model.tokenizer.decode([patched_top1_id])

        shifted_to_new      = (patched_top1_id == new_tid) and (corr_top1_id != new_tid)
        flipped_from_corr   = patched_top1_id != corr_top1_id

        results.append(PatchingResult(
            instance_id=iid,
            logit_diff_clean=ld_clean,
            logit_diff_corrupted=ld_corr,
            logit_diff_patched=ld_patch,
            recovery_fraction=recovery,
            p_new_clean=p_clean,
            p_new_corrupted=p_corr,
            p_new_patched=p_patch,
            p_new_recovery_fraction=p_recovery,
            top1_clean=top1_clean_s,
            top1_corrupted=top1_corr_s,
            top1_patched=top1_patched_s,
            top1_shifted_to_new=shifted_to_new,
            top1_flipped_from_corrupted=flipped_from_corr,
        ))
        bar.set_postfix(rec=f"{recovery:.2f}",
                        p_rec=f"{p_recovery:.2f}",
                        ok=len(results))

    bar.close()
    if verbose and any(skipped.values()):
        parts = [f"{k}={v}" for k, v in skipped.items() if v]
        print(f"  [F2-a] Skipped: {', '.join(parts)}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# F2-b  RouteScore
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class RouteScoreResult:
    """RouteScore and full Logit Lens trajectory for one instance.

    Two complementary scores:

    route_score      = P^{L_T}(answer_new) − P^{L_final}(answer_new)
                       Methodology F2-b primary definition (line 323):
                       L_T is the temporal-head layer (median of H_T).

    route_score_peak = max_{l ≥ L_T} P^l(answer_new) − P^{L_final}(answer_new)
                       Robust variant: captures mid-rises that occur a few
                       layers after L_T as well, which is necessary because
                       open-book DYNAMICQA mid-rises often happen at
                       mover-head layers rather than at $\\ell_{H_T}$
                       itself (Park §4 / Wang IOI mover heads).

    f2_regime classification follows methodology table (lines 288–292):

    ===========  =======================================================
    Regime       Pattern
    ===========  =======================================================
    not_f2       Final P^L(answer_new) >= ``p_final_high`` — signal
                 survived; not a routing failure.
    f2_strong    Low final + route_score_peak below ``rs_strong_thr``
                 (P^l never rises).
    f2_weak      Low final + route_score_peak in
                 [``rs_strong_thr``, ``rs_f3_thr``)
                 (moderate mid-rise, passive decay).
    f3           Low final + route_score_peak >= ``rs_f3_thr``
                 (strong mid-rise, active reversal — pooled here for
                 detection, but the *causal* F3 distinction is settled
                 in F3-b/c, not F2-b).
    ===========  =======================================================
    """
    instance_id: str
    p_new_at_temporal: float   # P^{L_T}(answer_new)
    p_new_at_final: float      # P^{L_final}(answer_new)
    p_new_peak: float          # max P^l(answer_new) for l ≥ L_T
    peak_layer: int            # layer where max is achieved
    route_score: float         # P^{L_T}(answer_new) − P^L(answer_new)  (methodology primary)
    route_score_peak: float    # p_new_peak − p_new_at_final            (robust variant)
    f2_regime: str             # one of {not_f2, f2_strong, f2_weak, f3}
    trajectory: LogitTrajectory
    # ── Rank-based readout (measurable redesign, PRIMARY) ─────────────────────
    # Calibration-free "routed" detector: answer_new becomes rank-competitive
    # (top-F2_RANK_COMPETITIVE) at some layer.  ``routed`` + low final rank with
    # no a_param emission ⇒ F2 ("Set but Not Routed").  -1 = never competitive.
    first_rank_competitive_layer: int = -1
    rank_new_final: int = -1
    routed: bool = False


# Default regime thresholds — match methodology F3-a's τ = 0.10 default and
# its {0.05, 0.10, 0.15} sweep.  Exposed so callers / sensitivity sweeps can
# override at the call site without re-classifying.
F2_DEFAULT_P_FINAL_HIGH    = 0.30   # final P^L above this → signal survived → not_f2
F2_DEFAULT_RS_STRONG_THR   = 0.02   # peak-final delta ≥ this → mid-rise drop occurred
F2_DEFAULT_RS_F3_THR       = 0.10   # peak ≥ this above final → strong mid-rise → f3
F2_DEFAULT_PEAK_LOW_THR    = 0.05   # absolute peak P^l below this ⇒ "never rises"
#: Rank below which answer_new counts as "routed" into the readout (0 = top-1).
#: Measurable-redesign PRIMARY readout; the absolute thresholds above are the
#: legacy/Optional-hardening path (methodology F2-b).
F2_RANK_COMPETITIVE        = 10


def classify_f2_regime(
    route_score_peak: float,
    p_new_at_final: float,
    *,
    p_new_peak: float | None = None,
    p_final_high: float = F2_DEFAULT_P_FINAL_HIGH,
    rs_strong_thr: float = F2_DEFAULT_RS_STRONG_THR,
    rs_f3_thr: float = F2_DEFAULT_RS_F3_THR,
    peak_low_thr: float = F2_DEFAULT_PEAK_LOW_THR,
) -> str:
    """Per-instance F2 regime classifier (methodology table lines 288–292).

    Returns one of: ``"not_f2"`` / ``"f2_strong"`` / ``"f2_weak"`` / ``"f3"``.
    F2-strong vs F2-weak are pooled as ``"F2"`` for Part III's distribution
    histogram; reported as separate cells here per methodology line 294.

    Methodology's three trajectory patterns (lines 290–292):

    *F2-strong*  — "$P^l(\\text{answer\\_new})$ **never rises**" → the
    *absolute* peak value is low.  Detected by ``p_new_peak < peak_low_thr``
    (when ``p_new_peak`` is provided).  The peak-minus-final delta alone
    is insufficient: a trajectory like (peak=0.25, final=0.24) would have
    a tiny delta but clearly rose, so it should be F2-weak.

    *F2-weak* — moderate rise then passive attenuation: peak rose
    appreciably (≥ ``peak_low_thr``) **and** moderate drop
    (``rs_strong_thr`` ≤ peak−final < ``rs_f3_thr``).

    *F3* — strong rise then sharp drop: peak−final ≥ ``rs_f3_thr``,
    final low.

    Parameters
    ----------
    p_new_peak : if ``None`` (legacy callers), the absolute-peak gate is
        skipped and F2-strong is decided from the delta alone.  Pass the
        absolute peak whenever possible.
    """
    if p_new_at_final >= p_final_high:
        return "not_f2"
    if route_score_peak >= rs_f3_thr:
        return "f3"

    # F2-strong (methodology line 279/290): "P^l never rises".
    # Authoritative check is on the *absolute peak value* — gate on
    # ``peak_low_thr`` whenever ``p_new_peak`` is supplied.  When the
    # caller doesn't supply ``p_new_peak`` (legacy), fall back to the
    # delta-only criterion so existing call sites still classify.
    if p_new_peak is not None:
        if p_new_peak < peak_low_thr:
            return "f2_strong"
        # peak rose meaningfully → never F2-strong; classify by drop size
        return "f2_weak"

    # Legacy / no-peak path: delta-only.
    if route_score_peak >= rs_strong_thr:
        return "f2_weak"
    return "f2_strong"


def compute_route_scores(
    model: HookedTransformer,
    instances: list[dict],
    temporal_head_layers: list[int],
    *,
    template: str = "plain",
    l_t_mode: str = "median",
    lens_kind: str = "raw",
    p_final_high: float = F2_DEFAULT_P_FINAL_HIGH,
    rs_strong_thr: float = F2_DEFAULT_RS_STRONG_THR,
    rs_f3_thr: float = F2_DEFAULT_RS_F3_THR,
    peak_low_thr: float = F2_DEFAULT_PEAK_LOW_THR,
    cached_trajectories: dict[str, "LogitTrajectory"] | None = None,
    verbose: bool = True,
) -> tuple[list[RouteScoreResult], int]:
    """F2-b: Compute RouteScore for each instance.

    ``RouteScore = P^{L_T}(answer_new) − P^{L_final}(answer_new)``

    where ``L_T`` is **the temporal-head layer** (methodology line 325).
    For multi-head $\\mathcal{H}_T$, $L_T = \\ell_{\\mathcal{H}_T}$ =
    *median* of temporal-head layers (consistent with F3-a's pre-flight
    diagnostic line 206).  ``l_t_mode`` is exposed for sensitivity
    analysis only — the methodology primary is ``"median"``.

    Each result additionally carries an F2 regime label via
    :func:`classify_f2_regime` so per-instance F2-strong / F2-weak / F3
    cells can be tabulated and pooled as ``"F2"`` for Part III.

    Parameters
    ----------
    lens_kind : ``"raw"`` (default) or ``"tuned"``.  Methodology F3-a
        Step 3 specifies **Tuned Lens** (Belrose 2023) as the primary
        lens, with raw RMSNorm-scaled lens reported as appendix
        robustness.  ``"tuned"`` is not yet implemented and currently
        falls back to raw with a warning so the field is plumbed through
        for callers; results stay appendix-grade until the per-layer
        Tuned-Lens probes are trained.
    cached_trajectories : optional ``{instance_id: LogitTrajectory}``
        dict.  When supplied, the function skips ``run_logit_lens`` for
        any instance whose ID is in the cache and reuses the cached
        trajectory — implements the methodology line 325 "computed via
        the Logit Lens intermediate projections already cached in F3-a"
        directive when F2-b and F3-a share a process / disk cache.

    Returns
    -------
    (results, l_t)  — list of :class:`RouteScoreResult` and the
    resolved ``L_T`` value.
    """
    # ``run_logit_lens`` is authoritative for the raw↔tuned warning + fallback
    # logic; we just propagate the kwarg so the warning fires exactly once
    # and the field plumbing is intact.
    if not temporal_head_layers:
        return [], -1

    if l_t_mode == "median":
        l_t = int(np.median(sorted(set(temporal_head_layers))))
    elif l_t_mode == "max":
        l_t = max(temporal_head_layers)
    elif l_t_mode == "min":
        l_t = min(temporal_head_layers)
    else:
        raise ValueError(f"l_t_mode must be 'median' / 'max' / 'min', got {l_t_mode!r}")

    results: list[RouteScoreResult] = []
    cache = cached_trajectories or {}
    n_cache_hits = 0
    bar = tqdm(instances, desc=f"F2-b  RouteScore (L_T={l_t}, {l_t_mode})",
               disable=not verbose)

    for inst in bar:
        iid = inst.get("instance_id", "")
        question = inst.get("question", "")
        context = inst.get("context", "")
        answer_new = inst.get("answer_new", "")
        answer_old = inst.get("answer_old", "")

        new_tid = get_first_answer_token(model, answer_new)
        old_tid = get_first_answer_token(model, answer_old)
        if new_tid < 0 or old_tid < 0 or new_tid == old_tid:
            continue

        if iid and iid in cache:
            traj = cache[iid]
            n_cache_hits += 1
        else:
            prompt = build_prompt(context, question, template=template)
            tokens = model.to_tokens(prompt, prepend_bos=False)
            traj = run_logit_lens(
                model, tokens, new_tid, old_tid, lens_kind=lens_kind,
            )

        p_temporal = float(traj.probs_new[l_t])
        p_final = float(traj.probs_new[-1])
        rs = p_temporal - p_final

        # Robust RouteScore: peak probability at or after L_T, then drop.
        probs_from_lt = traj.probs_new[l_t:]
        peak_idx = int(max(range(len(probs_from_lt)), key=lambda i: probs_from_lt[i]))
        p_peak = float(probs_from_lt[peak_idx])
        peak_layer = l_t + peak_idx
        rs_peak = p_peak - p_final

        regime = classify_f2_regime(
            rs_peak, p_final,
            p_new_peak=p_peak,
            p_final_high=p_final_high,
            rs_strong_thr=rs_strong_thr,
            rs_f3_thr=rs_f3_thr,
            peak_low_thr=peak_low_thr,
        )

        # Rank-based readout (measurable-redesign PRIMARY).
        ranks_new = getattr(traj, "ranks_new", None)
        if ranks_new is not None and len(ranks_new) > 0:
            ranks_arr = np.asarray(ranks_new)
            hits = np.nonzero(ranks_arr < F2_RANK_COMPETITIVE)[0]
            first_rc = int(hits[0]) if hits.size else -1
            rank_final = int(ranks_arr[-1])
        else:
            first_rc, rank_final = -1, -1

        results.append(RouteScoreResult(
            instance_id=iid,
            p_new_at_temporal=p_temporal,
            p_new_at_final=p_final,
            p_new_peak=p_peak,
            peak_layer=peak_layer,
            route_score=rs,
            route_score_peak=rs_peak,
            f2_regime=regime,
            trajectory=traj,
            first_rank_competitive_layer=first_rc,
            rank_new_final=rank_final,
            routed=(first_rc >= 0),
        ))
        bar.set_postfix(rs=f"{rs:+.3f}", regime=regime, n=len(results))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bar.close()
    if verbose and cache:
        print(f"  [F2-b cache] reused {n_cache_hits}/{len(results)} trajectories "
              "from the cross-script cache.")
    return results, l_t


# ═════════════════════════════════════════════════════════════════════════════
# F2-c  B5 vs B6 Behavioral Cross-Analysis
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class BehavioralCrossResult:
    """Aggregate and per-instance B1 / B5 / B6 comparison.

    Methodology F2-c (lines 334–349) makes **B5 × B6** the primary F2
    behavioral detector and **B1 × B5** a supplementary cross-reference.
    Both 2×2 contingencies are populated here.
    """
    b1_accuracy: float
    b5_accuracy: float
    b6_accuracy: float

    # ── Primary F2 detector: B5 × B6 (methodology lines 340–344) ───────────
    n_b5_success_b6_fail: int    # year-dependent routing works (anti-F2)
    n_b5_fail_b6_fail:    int    # candidate F2 (year-based routing broken)
    n_b5_fail_b6_success: int    # paradoxical: year HURTS performance
    n_b5_success_b6_success: int # year is not necessary for this instance

    # ── Supplementary: B1 × B5 (methodology lines 345–349) ─────────────────
    n_b1_fail_b5_success: int    # excludes F2 (B1 was single-passage weak)
    n_b1_fail_b5_fail:    int    # ambiguous — B5 harder than B1 in general
    n_b1_fail_b5_fail_rate: float

    # ── 3-way disambiguation (methodology line 347) ────────────────────────
    # "Disambiguate by checking B6 for these [B1-fail ∩ B5-fail] instances:
    #  if also B6-fail, likely parametric-memory dominance rather than
    #  routing failure."  These are the cells that distinguish F2 (routing
    #  failure) from parametric-memory dominance among ambiguous instances.
    n_b1f_b5f_b6f: int   # B1-fail ∩ B5-fail ∩ B6-fail → parametric dominance
    n_b1f_b5f_b6s: int   # B1-fail ∩ B5-fail ∩ B6-success → context-routing,
                         #   year *not* required for routing (anti-F2 at the
                         #   3-way level; same year-stripping increases acc).
    n_b1f_b5s_b6f: int   # B1-fail ∩ B5-success ∩ B6-fail → year-driven rescue
                         #   (routing works; rules out F2 for this instance).
    n_b1f_b5s_b6s: int   # B1-fail ∩ B5-success ∩ B6-success → dual-evidence
                         #   rescues regardless of year — robust routing.

    n_total: int
    details: list[dict] = field(default_factory=list)


def behavioral_cross_analysis(
    model: HookedTransformer,
    b1_instances: list[dict],
    b5_instances: list[dict],
    b6_instances: list[dict],
    *,
    b1_success_map: dict[str, bool] | None = None,
    template: str = "plain",
    verbose: bool = True,
) -> BehavioralCrossResult:
    """F2-c: B5 vs B6 behavioral cross-analysis.

    Instances must share the same ordering so that ``b1[i]``, ``b5[i]``,
    ``b6[i]`` correspond to the same ``(subject, property, t_old, t_new)``
    tuple.  The easiest way is to sort each list by ``fact_id`` before
    calling this function.

    Parameters
    ----------
    b1_success_map : optional {instance_id → bool} from a prior B1 generation
        pass (e.g. REVERTS_OLD filter).  When provided, B1 generation is
        skipped and the cached result is used instead — avoids duplicate
        forward passes.

    Methodology cells (lines 340–349):

    Primary (B5 × B6):
      * ``b5_success ∩ b6_fail`` — year-dependent routing **works** for these
        instances (year is the decisive factor); anti-F2 evidence.
      * ``b5_fail   ∩ b6_fail`` — candidate F2 (year-based routing broken),
        modulo the parametric-memory confound.
      * ``b5_fail   ∩ b6_success`` — paradoxical: year tokens *hurt* — check
        whether years reinforce parametric priors.

    Supplementary (B1 × B5):
      * ``b1_fail   ∩ b5_success`` — single-passage persuasion was the
        bottleneck; rules out F2.
      * ``b1_fail   ∩ b5_fail``    — ambiguous; B5 is strictly harder.
    """
    assert len(b1_instances) == len(b5_instances) == len(b6_instances), (
        f"Instance lists must have equal length: "
        f"B1={len(b1_instances)}, B5={len(b5_instances)}, B6={len(b6_instances)}"
    )

    b1_ok = b5_ok = b6_ok = 0
    # Primary B5 × B6 contingencies
    b5s_b6f = 0   # anti-F2
    b5f_b6f = 0   # candidate F2
    b5f_b6s = 0   # paradoxical
    b5s_b6s = 0   # year not needed
    # Supplementary B1 × B5
    b1f_b5s = 0
    b1f_b5f = 0
    # 3-way disambiguation cells (methodology line 347)
    b1f_b5f_b6f = 0  # parametric dominance
    b1f_b5f_b6s = 0  # year not required (routing works without year)
    b1f_b5s_b6f = 0  # year-driven rescue
    b1f_b5s_b6s = 0  # dual-evidence rescue regardless of year
    details: list[dict] = []

    desc = "F2-c  B5 vs B6"
    if b1_success_map is not None:
        desc += " (B1 cached)"

    bar = tqdm(
        zip(b1_instances, b5_instances, b6_instances),
        total=len(b1_instances),
        desc=desc,
        disable=not verbose,
    )

    for b1, b5, b6 in bar:
        answer_new = b1.get("answer_new", "")

        def _gen_eval(inst: dict) -> tuple[bool, str]:
            ctx = inst.get("context", "")
            q = inst.get("question", "")
            prompt = build_prompt(ctx, q, template=template)
            gen = generate_answer(model, prompt)
            return check_match(gen, answer_new), gen

        # B1: reuse cached result when available
        iid = b1.get("instance_id", "")
        if b1_success_map is not None and iid in b1_success_map:
            r1 = b1_success_map[iid]
            gen1 = "(cached)"
        else:
            r1, gen1 = _gen_eval(b1)

        r5, gen5 = _gen_eval(b5)
        r6, gen6 = _gen_eval(b6)

        if r1:
            b1_ok += 1
        if r5:
            b5_ok += 1
        if r6:
            b6_ok += 1

        # Primary B5 × B6 cells
        if r5 and not r6:
            b5s_b6f += 1
            f2c_cell = "b5s_b6f"
        elif not r5 and not r6:
            b5f_b6f += 1
            f2c_cell = "b5f_b6f"
        elif not r5 and r6:
            b5f_b6s += 1
            f2c_cell = "b5f_b6s"
        else:
            b5s_b6s += 1
            f2c_cell = "b5s_b6s"

        # Supplementary B1 × B5
        if not r1 and r5:
            b1f_b5s += 1
        if not r1 and not r5:
            b1f_b5f += 1

        # 3-way B1 × B5 × B6 cells (methodology line 347 disambiguation)
        if not r1:
            if not r5 and not r6:
                b1f_b5f_b6f += 1
            elif not r5 and r6:
                b1f_b5f_b6s += 1
            elif r5 and not r6:
                b1f_b5s_b6f += 1
            else:
                b1f_b5s_b6s += 1

        details.append({
            "fact_id": b1.get("fact_id", ""),
            "instance_id_b1": b1.get("instance_id", ""),
            "instance_id_b5": b5.get("instance_id", ""),
            "instance_id_b6": b6.get("instance_id", ""),
            "t_old": b1.get("t_old"),
            "t_new": b1.get("t_new"),
            "b1_success": r1,
            "b5_success": r5,
            "b6_success": r6,
            "b5_generated": gen5,
            "b6_generated": gen6,
            "f2c_cell": f2c_cell,
        })
        bar.set_postfix(
            b1=f"{b1_ok}/{len(details)}",
            b5=f"{b5_ok}/{len(details)}",
            b6=f"{b6_ok}/{len(details)}",
            b5fb6f=b5f_b6f,
        )

    bar.close()

    n = len(b1_instances)
    n_b1_fail = n - b1_ok

    return BehavioralCrossResult(
        b1_accuracy=b1_ok / n if n else 0,
        b5_accuracy=b5_ok / n if n else 0,
        b6_accuracy=b6_ok / n if n else 0,
        n_b5_success_b6_fail=b5s_b6f,
        n_b5_fail_b6_fail=b5f_b6f,
        n_b5_fail_b6_success=b5f_b6s,
        n_b5_success_b6_success=b5s_b6s,
        n_b1_fail_b5_success=b1f_b5s,
        n_b1_fail_b5_fail=b1f_b5f,
        n_b1_fail_b5_fail_rate=b1f_b5f / n_b1_fail if n_b1_fail else 0.0,
        n_b1f_b5f_b6f=b1f_b5f_b6f,
        n_b1f_b5f_b6s=b1f_b5f_b6s,
        n_b1f_b5s_b6f=b1f_b5s_b6f,
        n_b1f_b5s_b6s=b1f_b5s_b6s,
        n_total=n,
        details=details,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Verdict assembly — methodology lines 282–294
# ═════════════════════════════════════════════════════════════════════════════


def load_f1_positive_map(
    f1a_path: str,
    *,
    percentile: int = 25,
) -> tuple[dict[str, bool], dict[tuple, bool]]:
    """Read F1-a Step 5 output and return per-instance F1-positive maps.

    Methodology F1-a Step 5: instance ``i`` is **F1-positive** iff its
    $\\bar{A}^{\\mathcal{H}_T}_i$ scalar lies below the ``percentile``-th
    percentile of the B1-success distribution.  F1-positive ⇒ year not
    read at $\\mathcal{H}_T$ ⇒ F1, and F2 is **ruled out** on that
    instance.  F1-negative ⇒ year **was** read ⇒ F2 / F3 still possible.

    Two indices are returned because F1 is typically run on B5 instances
    (the recommended dual-span temporal testbed) while F2 runs on B1
    REVERTS_OLD, so the ``instance_id`` namespaces differ.  Callers
    should attempt the ``instance_id`` lookup first and fall back to the
    ``(fact_id, t_old, t_new)`` lookup.

    Returns
    -------
    by_instance_id : ``{instance_id: is_F1_positive}``
    by_fact_key    : ``{(fact_id, t_old, t_new): is_F1_positive}``
    """
    import json
    from pathlib import Path

    path = Path(f1a_path)
    with path.open() as fh:
        data = json.load(fh)

    step5 = data.get("step5_f1_positive") or {}
    flags_by_pctile = step5.get("f1_positive_by_percentile") or {}
    flags = flags_by_pctile.get(str(percentile)) or flags_by_pctile.get(percentile)
    if not flags:
        raise ValueError(
            f"F1-a file {f1a_path} has no Step-5 verdict at percentile "
            f"{percentile}; rerun run_f1_diagnostic.py with the updated "
            "sat_probe.py that emits step5_f1_positive."
        )

    meta = data.get("instance_meta") or []
    if len(meta) != len(flags):
        raise ValueError(
            f"F1-a meta length ({len(meta)}) ≠ Step-5 flag length "
            f"({len(flags)}) in {f1a_path}; the file is inconsistent."
        )

    by_iid: dict[str, bool] = {}
    by_key: dict[tuple, bool] = {}
    for m, is_pos in zip(meta, flags):
        iid = str(m.get("instance_id", ""))
        if iid:
            by_iid[iid] = bool(is_pos)
        fact_id = m.get("fact_id")
        t_old   = m.get("t_old")
        t_new   = m.get("t_new")
        if fact_id is not None and t_old is not None and t_new is not None:
            by_key[(str(fact_id), int(t_old), int(t_new))] = bool(is_pos)
    return by_iid, by_key


def load_f1_consistency(
    f1b_path: str,
    *,
    alpha: float = 0.05,
) -> dict:
    """Read F1-b output and return a population-level consistency report.

    Methodology line 284 says the F2 verdict requires cross-referencing
    **F1-a/b** (note the slash).  F1-a yields per-instance ``F1-positive``
    verdicts (handled by :func:`load_f1_positive_map`); F1-b yields a
    population-level Mann-Whitney U test of whether B1-success attention
    to year tokens at $\\mathcal{H}_T$ is significantly higher than
    B1-failure attention.  A non-significant F1-b ⇒ the "Time Set"
    premise is shaky for *all* failures, in which case the F1-positive
    per-instance verdicts are themselves untrustworthy and the
    downstream F2 verdict assembly should be flagged.

    Returns
    -------
    dict with keys:
      * ``mann_whitney_p`` : F1-b's reported p-value (or None if absent).
      * ``alpha``          : significance threshold used here.
      * ``f1b_significant``: ``mann_whitney_p < alpha``.
      * ``raw``            : the full F1-b JSON for downstream inspection.
    """
    import json
    from pathlib import Path

    path = Path(f1b_path)
    with path.open() as fh:
        data = json.load(fh)

    p = data.get("mann_whitney_p")
    if p is None:
        # Older F1-b output schemas store the p-value under one of the
        # head-specific blocks; we try the top-level then bail with None.
        for k in ("p_value", "p"):
            if k in data:
                p = data[k]
                break

    p_float = float(p) if p is not None else None
    return {
        "mann_whitney_p": p_float,
        "alpha":          alpha,
        "f1b_significant": (p_float is not None) and (p_float < alpha),
        "raw":            data,
    }


def _f1_lookup(
    inst: dict,
    by_iid: dict[str, bool],
    by_key: dict[tuple, bool],
) -> Optional[bool]:
    iid = str(inst.get("instance_id", ""))
    if iid in by_iid:
        return by_iid[iid]
    fact_id = inst.get("fact_id")
    t_old   = inst.get("t_old")
    t_new   = inst.get("t_new")
    if fact_id is not None and t_old is not None and t_new is not None:
        return by_key.get((str(fact_id), int(t_old), int(t_new)))
    return None


def assign_f2_verdicts(
    instances: list[dict],
    route_results: list[RouteScoreResult],
    *,
    f1_by_iid: dict[str, bool] | None = None,
    f1_by_key: dict[tuple, bool] | None = None,
    f1b_consistency: dict | None = None,
) -> list[dict]:
    """Combine F2-b regime + F1-a/b status into per-instance F2 verdicts.

    Methodology lines 282–294: F2 is confirmed for an instance only if
    **both** conditions hold:

      1. **F1-a Step 5** is *F1-negative* on that instance (year was
         read at $\\mathcal{H}_T$ — per-instance criterion).
      2. **F1-b** is population-significant (Mann-Whitney B1-success vs
         B1-failure attention; population-level sanity check that
         $\\mathcal{H}_T$ really does read the year token at all).  If
         F1-b is *not* significant the entire F1-a Step-5 verdict layer
         is unreliable; we annotate the verdict suffix
         ``_f1b_nonsignif`` so downstream readers can see the warning.
      3. F2-b trajectory shows the routing pattern (regime == ``f2_strong``
         or ``f2_weak``).

    The ``f3`` regime is *detectable* here as the strong-mid-rise +
    low-final case but its **causal** confirmation belongs to F3-b/c; we
    therefore label it ``f3_candidate`` to flag the population that
    should be carried into F3.

    Returns
    -------
    list of dicts with fields:
      * ``instance_id`` / ``fact_id`` / ``t_old`` / ``t_new``
      * ``f1_positive``: True / False / None (None = no F1 file or
        instance not present in F1 cohort)
      * ``f2b_regime``: ``"not_f2"`` / ``"f2_strong"`` / ``"f2_weak"`` / ``"f3"``
      * ``verdict``: final per-instance verdict, one of
        ``"F1"`` / ``"F2_strong"`` / ``"F2_weak"`` / ``"F3_candidate"``
        / ``"not_routing_failure"`` / ``"undetermined"`` /
        ``"…_unverified"`` (no F1-a file) / ``"…_f1b_nonsignif"``
        (F1-b non-significant).
    """
    by_iid = f1_by_iid or {}
    by_key = f1_by_key or {}
    route_by_iid = {r.instance_id: r for r in route_results}

    # F1-b consistency: when provided AND non-significant, the entire F1-a
    # per-instance verdict layer is untrustworthy.  We don't *block* the F2
    # verdicts — we annotate them so downstream readers see the warning.
    f1b_block = (
        (f1b_consistency is not None)
        and (f1b_consistency.get("mann_whitney_p") is not None)
        and (not f1b_consistency.get("f1b_significant", False))
    )

    verdicts: list[dict] = []
    for inst in instances:
        iid = str(inst.get("instance_id", ""))
        r = route_by_iid.get(iid)
        is_f1_pos = _f1_lookup(inst, by_iid, by_key)

        if r is None:
            f2b_regime = None
            base_verdict = "undetermined"
        else:
            f2b_regime = r.f2_regime
            if f2b_regime == "not_f2":
                base_verdict = "not_routing_failure"
            elif f2b_regime == "f3":
                base_verdict = "F3_candidate"
            elif f2b_regime == "f2_weak":
                base_verdict = "F2_weak"
            elif f2b_regime == "f2_strong":
                base_verdict = "F2_strong"
            else:
                base_verdict = "undetermined"

        # F1 cross-reference (methodology line 284): F1-positive blocks
        # any F2 verdict — the failure is F1.  F3_candidate is *also*
        # blocked, because F3 likewise requires the year to have been
        # read (Part III's F3 row, and F3-a Method line 424 keying off
        # B1-failure with F3-trajectory).
        if is_f1_pos is True and base_verdict in (
            "F2_strong", "F2_weak", "F3_candidate"
        ):
            verdict = "F1"
        elif is_f1_pos is None and base_verdict in (
            "F2_strong", "F2_weak", "F3_candidate"
        ):
            # No F1 verdict available for this instance — report the F2
            # regime as ``_unverified`` so downstream readers know the
            # F1 ruling-out step did not happen.
            verdict = f"{base_verdict}_unverified"
        else:
            verdict = base_verdict

        # F1-b consistency: if F1-b is non-significant, every F2 (and F3)
        # verdict is annotated to signal that the "Time Set" population
        # premise itself is not validated.  We append the suffix rather
        # than overwriting so the underlying regime is still readable.
        if f1b_block and verdict in (
            "F2_strong", "F2_weak", "F3_candidate",
            "F2_strong_unverified", "F2_weak_unverified", "F3_candidate_unverified",
            "F1",
        ):
            verdict = f"{verdict}_f1b_nonsignif"

        verdicts.append({
            "instance_id": iid,
            "fact_id":     inst.get("fact_id"),
            "t_old":       inst.get("t_old"),
            "t_new":       inst.get("t_new"),
            "f1_positive": is_f1_pos,
            "f2b_regime":  f2b_regime,
            "verdict":     verdict,
        })
    return verdicts
