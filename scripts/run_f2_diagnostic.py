#!/usr/bin/env python3
"""F2 Diagnostic — "Time Set but Not Routed"

Standalone script that runs the three F2 sub-experiments from the TATM
methodology:

  F2-a  STR Activation Patching — causal confirmation that temporal heads
        mediate the year routing signal.  Requires temporal_heads from F1
        (pass via --temporal-heads).

  F2-b  RouteScore — Logit Lens trajectory on REVERTS_OLD instances.
        Measures temporal signal attenuation between temporal-head layers
        and the final layer.

  F2-c  B5 vs B6 Behavioral Cross-Analysis — cheapest F2 detector.
        Compares accuracy on dual-span context (B5, year intact) vs
        year-stripped (B6), cross-referenced with single-evidence B1.

Prerequisites
-------------
  1. Provide temporal heads via --temporal-heads or --temporal-heads-manual.
     For Phi-3, the default stage script uses the paper-derived list in
     data/external/temporal_heads/paper_temporal_heads.json.
  2. Layer-2 JSONL must contain B1, B5, and B6 instance types.
     Build with:  python scripts/build_wikidata_layer2.py --layers B1 B3 B5 B6

Usage
-----
    python scripts/run_f2_diagnostic.py \\
        --data  data/processed/wikidata_layer2.jsonl \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --temporal-heads data/external/temporal_heads/paper_temporal_heads.json \\
        --out   results/f2_diagnostic/

    # Specify temporal heads manually (layer,head pairs):
    python scripts/run_f2_diagnostic.py \\
        --data  data/processed/wikidata_layer2.jsonl \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --temporal-heads-manual 15,0 15,3 16,2 \\
        --out   results/f2_diagnostic/

    # Skip sub-experiments:
    python scripts/run_f2_diagnostic.py ... --skip f2a f2b
"""
from __future__ import annotations

import argparse
import inspect
import json
import random as _random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "source"))

from tatm.f2_diagnosis import (
    DLA_F1F2_TAU,
    F2_DEFAULT_P_FINAL_HIGH,
    F2_DEFAULT_PEAK_LOW_THR,
    F2_DEFAULT_RS_F3_THR,
    F2_DEFAULT_RS_STRONG_THR,
    assign_f2_verdicts,
    behavioral_cross_analysis,
    compute_route_scores,
    load_f1_consistency,
    load_f1_positive_map,
    mcnemar_test,
    run_str_patching,
    run_str_patching_resid_sweep,
)
from tatm.model import (
    build_prompt,
    check_match,
    generate_answer,
    load_model,
)

F2_OUTPUT_SCHEMA_VERSION = "f2_v2_dla_clean_verdict"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_b1_b5_b6(path: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Load B1, B5, B6 instances from a Layer-2 JSONL, aligned by key."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    b1 = [r for r in records if r.get("instance_id", "").startswith("B1")]
    b5 = [r for r in records if r.get("instance_id", "").startswith("B5")]
    b6 = [r for r in records if r.get("instance_id", "").startswith("B6")]

    if not b5:
        raise ValueError(
            "No B5 instances found.  Rebuild layer2 with --layers B1 B3 B5 B6."
        )

    # Align all three lists to the B5 set via (fact_id, t_old, t_new)
    def _index(lst: list[dict]) -> dict:
        return {(r["fact_id"], r["t_old"], r["t_new"]): r for r in lst}

    b1_map = _index(b1)
    b6_map = _index(b6)

    b5_aligned, b1_aligned, b6_aligned = [], [], []
    for r in b5:
        key = (r["fact_id"], r["t_old"], r["t_new"])
        if key in b1_map and key in b6_map:
            b5_aligned.append(r)
            b1_aligned.append(b1_map[key])
            b6_aligned.append(b6_map[key])

    return b1_aligned, b5_aligned, b6_aligned


def restrict_instances(
    b1_instances: list[dict],
    b5_instances: list[dict],
    b6_instances: list[dict],
    *,
    number: int | None = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Randomly sample Layer-1 facts and keep all aligned B1/B5/B6 triples."""
    if number is None:
        return b1_instances, b5_instances, b6_instances
    if number < 1:
        raise ValueError("--number must be >= 1")

    n_triples = min(len(b1_instances), len(b5_instances), len(b6_instances))
    b1_instances = b1_instances[:n_triples]
    b5_instances = b5_instances[:n_triples]
    b6_instances = b6_instances[:n_triples]

    fact_ids = sorted({
        str(inst.get("fact_id") or idx)
        for idx, inst in enumerate(b5_instances)
    })
    if number >= len(fact_ids):
        return (
            b1_instances[:n_triples],
            b5_instances[:n_triples],
            b6_instances[:n_triples],
        )

    rng = _random.Random(seed)
    selected_fact_ids = set(rng.sample(fact_ids, number))
    indices = [
        idx
        for idx, inst in enumerate(b5_instances)
        if str(inst.get("fact_id") or idx) in selected_fact_ids
    ]
    return (
        [b1_instances[i] for i in indices],
        [b5_instances[i] for i in indices],
        [b6_instances[i] for i in indices],
    )


def load_temporal_heads(
    probe_json: str,
    top_k: int = 10,
    model_name: str | None = None,
) -> list[tuple[int, int]]:
    """Read temporal heads from a JSON file.

    This supports both ``f1a_sat_probe.json`` with a top-level ``top_heads``
    list and paper-derived files grouped by model under ``models``.
    """
    with open(probe_json) as f:
        data = json.load(f)

    if "models" in data:
        if not model_name:
            raise ValueError(
                f"{probe_json} contains multiple models. Pass --model so the "
                "matching temporal-head entry can be selected."
            )
        model_key = _select_temporal_heads_model(data["models"], model_name, probe_json)
        data = data["models"][model_key]

    heads = []
    for entry in data.get("top_heads", [])[:top_k]:
        if entry.get("coef", "0") not in ("0", "0.0000e+00"):
            heads.append((int(entry["layer"]), int(entry["head"])))
    if not heads:
        raise ValueError(
            f"No non-zero temporal heads found in {probe_json}.  "
            "Pass --temporal-heads-manual instead."
        )
    return heads


def _select_temporal_heads_model(
    model_entries: dict,
    model_name: str,
    source_path: str,
) -> str:
    """Select the temporal-head entry matching the requested model."""
    requested = model_name.lower()
    requested_short = requested.rsplit("/", 1)[-1]

    for key, entry in model_entries.items():
        candidates = {
            key.lower(),
            str(entry.get("model", "")).lower(),
            str(entry.get("model", "")).lower().rsplit("/", 1)[-1],
        }
        if requested in candidates or requested_short in candidates:
            return key

    available = ", ".join(
        str(entry.get("model", key)) for key, entry in model_entries.items()
    )
    raise ValueError(
        f"No temporal heads for model {model_name!r} in {source_path}. "
        f"Available models: {available}"
    )


# ── REVERTS_OLD filter ────────────────────────────────────────────────────────

def run_reverts_old_filter(
    model,
    b1_instances: list[dict],
    template: str,
    out_dir: Path,
) -> tuple[list[dict], dict[str, bool]]:
    """Keep only instances where the model fails B1 AND outputs answer_old.

    REVERTS_OLD: B1-failure (output ≠ answer_new) AND output matches answer_old.
    These are the diagnostic targets for F2/F3 — genuine temporal conflict failures.

    REVERTS_OTHER: B1-failure where output ≠ answer_new AND output ≠ answer_old.
    NOTE: this does NOT mean the model has no parametric memory.  Typical cases:
      • Model knows a *third* answer (different time period's answer)
      • Name variant / transliteration mismatch (check_match false negative)
      • Refusal or hallucination
      • Unresolved Wikidata QID in expected answer (data bug)

    Returns
    -------
    reverts_old : list of B1 instance dicts that satisfy REVERTS_OLD
    b1_success_map : {instance_id → True if B1-success} — reusable by F2-c
        to avoid duplicate generation
    """
    import re as _re
    def _is_qid(s: str) -> bool:
        return bool(_re.fullmatch(r"Q\d+", s.strip()))

    print("\n" + "=" * 60)
    print("REVERTS_OLD Filter")
    print("=" * 60)
    print("Keeping: B1-failure instances where model outputs answer_old\n")

    reverts_old: list[dict] = []
    b1_success: list[dict] = []
    reverts_other: list[dict] = []
    b1_success_map: dict[str, bool] = {}
    log = []

    n_qid_answers = 0
    bar = tqdm(b1_instances, desc="REVERTS_OLD filter", unit="inst", dynamic_ncols=True)
    for inst in bar:
        ctx      = inst.get("context", inst.get("evidence_new", ""))
        question = inst.get("question", "")
        ans_new  = inst.get("answer_new", "")
        ans_old  = inst.get("answer_old", "")

        # Flag unresolved QID answers as bad data (check_match always returns
        # False for QIDs, so they will land in reverts_other if the model is
        # wrong, confusing the count).
        has_qid_answer = _is_qid(ans_new) or _is_qid(ans_old)
        if has_qid_answer:
            n_qid_answers += 1

        prompt    = build_prompt(ctx, question, template=template)
        generated = generate_answer(model, prompt)

        is_new = check_match(generated, ans_new)
        is_old = check_match(generated, ans_old)

        iid = inst.get("instance_id", "")
        b1_success_map[iid] = is_new

        entry: dict = {
            "instance_id": iid,
            "generated": generated,
            "answer_new": ans_new,
            "answer_old": ans_old,
            "b1_success": is_new,
            "reverts_old": (not is_new) and is_old,
            "bad_data_qid": has_qid_answer,
        }
        log.append(entry)

        if is_new:
            b1_success.append(inst)
        elif is_old:
            reverts_old.append(inst)
        else:
            reverts_other.append(inst)

        bar.set_postfix(
            success=len(b1_success),
            reverts_old=len(reverts_old),
            other=len(reverts_other),
            refresh=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = len(b1_instances)
    print(f"\n  B1-success      : {len(b1_success)}/{n}  ({100*len(b1_success)/n:.1f}%)")
    print(f"  REVERTS_OLD     : {len(reverts_old)}/{n}  ({100*len(reverts_old)/n:.1f}%)")
    print(f"  REVERTS_OTHER   : {len(reverts_other)}/{n}  ({100*len(reverts_other)/n:.1f}%)")
    if n_qid_answers:
        print(f"  [!] Unresolved QID answers found: {n_qid_answers} instances — "
              "fix data pipeline (eval_builder.py label resolution)")
    print(f"\n  Note: REVERTS_OTHER may include: (a) model knows a *third* answer "
          f"(different time-period parametric memory), (b) name-variant check_match "
          f"misses, (c) refusals.  It does NOT imply absence of parametric memory.")
    print(f"\n→ {len(reverts_old)} REVERTS_OLD instances passed to F2/F3 diagnosis")

    with open(out_dir / "reverts_old_filter.json", "w") as f:
        json.dump(
            {
                "n_total": n,
                "n_b1_success": len(b1_success),
                "n_reverts_old": len(reverts_old),
                "n_reverts_other": len(reverts_other),
                "n_qid_answers": n_qid_answers,
                "note_reverts_other": (
                    "Output ≠ answer_new AND output ≠ answer_old. Does NOT imply "
                    "no parametric memory. May include: (a) third-year parametric "
                    "memory, (b) name variants check_match cannot reconcile, "
                    "(c) refusals, (d) unresolved Wikidata QIDs in expected answer."
                ),
                "log": log,
            },
            f, indent=2, ensure_ascii=False,
        )

    return reverts_old, b1_success_map


# ── F2-a: STR Activation Patching ────────────────────────────────────────────

def run_f2a(
    model,
    b5_instances: list[dict],
    temporal_heads: list[tuple[int, int]],
    template: str,
    out_dir: Path,
) -> list[dict]:
    print("\n" + "=" * 60)
    print("F2-a: STR Activation Patching")
    print("=" * 60)
    print(f"Temporal heads to patch: {temporal_heads}")
    print("Behavioral filter: clean(t_new)→answer_new  &  corrupted(t_old)→answer_old\n")

    results = run_str_patching(
        model, b5_instances, temporal_heads, template=template, verbose=True,
    )

    if not results:
        print("  No instances passed the behavioral filter.")
        return []

    recoveries   = [r.recovery_fraction for r in results]
    p_recoveries = [r.p_new_recovery_fraction for r in results]
    n_shifted    = sum(1 for r in results if r.top1_shifted_to_new)
    n_flipped    = sum(1 for r in results if r.top1_flipped_from_corrupted)

    mean_rec   = float(np.mean(recoveries))
    med_rec    = float(np.median(recoveries))
    pct_50     = sum(1 for r in recoveries if r >= 0.5) / len(recoveries)
    mean_p_rec = float(np.mean(p_recoveries))
    med_p_rec  = float(np.median(p_recoveries))

    print(f"\n  Instances with valid patching trials: {len(results)}")
    print(f"  Mean  logit-diff recovery fraction: {mean_rec:+.3f}")
    print(f"  Median logit-diff recovery fraction: {med_rec:+.3f}")
    print(f"  Fraction ≥ 50% recovery (causal):    {pct_50:.1%}")
    print(f"  Mean  P(answer_new) recovery:        {mean_p_rec:+.3f}   "
          "(methodology line 311 'Does p(answer_new) recover?')")
    print(f"  Median P(answer_new) recovery:       {med_p_rec:+.3f}")
    print(f"  Top-1 shifted → answer_new:          {n_shifted}/{len(results)}  "
          f"({100*n_shifted/len(results):.1f}%)  (methodology line 307 'top-1 shift')")
    print(f"  Top-1 flipped away from corrupted:   {n_flipped}/{len(results)}  "
          f"({100*n_flipped/len(results):.1f}%)")

    output = {
        "n_patched": len(results),
        "mean_recovery": mean_rec,
        "median_recovery": med_rec,
        "pct_ge_50_recovery": pct_50,
        "mean_p_new_recovery": mean_p_rec,
        "median_p_new_recovery": med_p_rec,
        "n_top1_shifted_to_new": n_shifted,
        "n_top1_flipped_from_corrupted": n_flipped,
        "rate_top1_shifted_to_new": n_shifted / len(results),
        "per_instance": [
            {
                "instance_id": r.instance_id,
                "logit_diff_clean": r.logit_diff_clean,
                "logit_diff_corrupted": r.logit_diff_corrupted,
                "logit_diff_patched": r.logit_diff_patched,
                "recovery_fraction": r.recovery_fraction,
                "p_new_clean":     r.p_new_clean,
                "p_new_corrupted": r.p_new_corrupted,
                "p_new_patched":   r.p_new_patched,
                "p_new_recovery_fraction": r.p_new_recovery_fraction,
                "top1_clean":     r.top1_clean,
                "top1_corrupted": r.top1_corrupted,
                "top1_patched":   r.top1_patched,
                "top1_shifted_to_new":         r.top1_shifted_to_new,
                "top1_flipped_from_corrupted": r.top1_flipped_from_corrupted,
            }
            for r in results
        ],
    }
    with open(out_dir / "f2a_str_patching.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output["per_instance"]


# ── F2-b: RouteScore ─────────────────────────────────────────────────────────

def run_f2b(
    model,
    reverts_old: list[dict],
    temporal_heads: list[tuple[int, int]],
    template: str,
    out_dir: Path,
    *,
    population_label: str = "reverts_old",
    l_t_mode: str = "median",
    lens_kind: str = "raw",
    p_final_high: float = F2_DEFAULT_P_FINAL_HIGH,
    rs_strong_thr: float = F2_DEFAULT_RS_STRONG_THR,
    rs_f3_thr: float = F2_DEFAULT_RS_F3_THR,
    peak_low_thr: float = F2_DEFAULT_PEAK_LOW_THR,
    cached_trajectories: dict | None = None,
) -> list:
    """Run F2-b RouteScore + per-instance regime classification.

    Returns the list of ``RouteScoreResult`` so the main driver can feed
    them into the F1-cross-referenced verdict assembly.
    """
    print("\n" + "=" * 60)
    print("F2-b: RouteScore  (Temporal Signal Attenuation)")
    print("=" * 60)

    temporal_layers = sorted({l for l, _ in temporal_heads})
    if not temporal_layers:
        print("  No temporal head layers — skipping RouteScore.")
        return []

    print(f"  Population                  : {population_label}")
    print(f"  Temporal head layers        : {temporal_layers}")
    print(f"  L_T mode                    : {l_t_mode}  (methodology primary: median)")
    print(f"  Lens kind                   : {lens_kind}  (methodology primary: tuned)")
    print(f"  Regime thresholds           : "
          f"P^L≥{p_final_high}→not_f2, "
          f"peak_drop≥{rs_f3_thr:.2f}→F3, "
          f"peak_drop∈[{rs_strong_thr:.2f},{rs_f3_thr:.2f})→F2-weak\n"
          f"  Absolute peak gate          : "
          f"|P^l_peak|<{peak_low_thr:.2f} ⇒ F2-strong ('never rises')\n")

    results, l_t = compute_route_scores(
        model, reverts_old, temporal_layers,
        template=template,
        l_t_mode=l_t_mode,
        lens_kind=lens_kind,
        p_final_high=p_final_high,
        rs_strong_thr=rs_strong_thr,
        rs_f3_thr=rs_f3_thr,
        peak_low_thr=peak_low_thr,
        cached_trajectories=cached_trajectories,
        temporal_heads=temporal_heads,
        compute_dla=True,
        verbose=True,
    )

    if not results:
        print("  No RouteScore results (check answer token lookup).")
        return []

    print(f"  Resolved L_T                : {l_t}")

    route_scores      = [r.route_score for r in results]
    route_scores_peak = [r.route_score_peak for r in results]
    p_at_lt           = [r.p_new_at_temporal for r in results]
    p_at_peak         = [r.p_new_peak for r in results]
    p_at_final        = [r.p_new_at_final for r in results]
    peak_layers       = [r.peak_layer for r in results]
    regimes           = [r.f2_regime for r in results]

    regime_counts = {
        "not_f2":    regimes.count("not_f2"),
        "f2_strong": regimes.count("f2_strong"),
        "f2_weak":   regimes.count("f2_weak"),
        "f3":        regimes.count("f3"),
    }
    n_f2_pooled = regime_counts["f2_strong"] + regime_counts["f2_weak"]

    print(f"\n  n = {len(results)}")
    print(f"  Mean  P(answer_new) at L_T:    {np.mean(p_at_lt):.4f}")
    print(f"  Mean  P(answer_new) peak:      {np.mean(p_at_peak):.4f}  "
          f"(avg peak layer: {np.mean(peak_layers):.1f})")
    print(f"  Mean  P(answer_new) at L_final:{np.mean(p_at_final):.4f}")
    print(f"  Mean  RouteScore (L_T basis):  {np.mean(route_scores):+.4f}")
    print(f"  Mean  RouteScore (peak basis): {np.mean(route_scores_peak):+.4f}")
    print(f"\n  F2 regime distribution (per-instance, DESCRIPTIVE only):")
    for label, count in regime_counts.items():
        print(f"    {label:>10s}: {count:>4d} ({100*count/len(results):5.1f}%)")
    print(f"    {'F2 pooled':>10s}: {n_f2_pooled:>4d} "
          f"({100*n_f2_pooled/len(results):5.1f}%)  "
          "(F2-strong + F2-weak collapsed; absolute-threshold regime demoted)")

    # ── DLA readout (AUTHORITATIVE F1/F2 separator, Ortu 2024) ──────────────
    dla_vals   = [r.dla_ht_sum for r in results if r.dla_ht_sum is not None]
    dla_labels = [r.dla_f1_vs_f2 for r in results if r.dla_f1_vs_f2 is not None]
    n_dla      = len(dla_vals)
    n_dla_f2   = dla_labels.count("F2")
    n_dla_f1   = dla_labels.count("F1")
    if n_dla:
        print(f"\n  DLA readout (H_T direct attribution onto answer_new−answer_old):")
        print(f"    n with DLA          : {n_dla}/{len(results)}")
        print(f"    mean Σ DLA(H_T)     : {np.mean(dla_vals):+.4f}  (logit units)")
        print(f"    median Σ DLA(H_T)   : {np.median(dla_vals):+.4f}")
        print(f"    DLA-says-F2 (writes): {n_dla_f2:>4d} ({100*n_dla_f2/n_dla:5.1f}%)")
        print(f"    DLA-says-F1 (no-write): {n_dla_f1:>4d} ({100*n_dla_f1/n_dla:5.1f}%)")

    # ── Lens-decodable coverage (reuse first_rank_competitive_layer) ────────
    # An instance is "lens-decodable" iff answer_new became rank-competitive at
    # a NON-FINAL layer (high-crystallization models route only at the last
    # layer, so the lens legs are uninformative there).  No separate filter.
    def _decodable(r) -> bool:
        l_final = len(r.trajectory.probs_new) - 1
        return 0 <= r.first_rank_competitive_layer < l_final
    n_decodable = sum(1 for r in results if _decodable(r))
    coverage = n_decodable / len(results)
    print(f"\n  Lens-decodable coverage     : {n_decodable}/{len(results)} "
          f"({100*coverage:5.1f}%)  (rank-competitive before final layer)")

    output = {
        "schema_version": F2_OUTPUT_SCHEMA_VERSION,
        "tatm_f2_diagnosis_path": inspect.getsourcefile(compute_route_scores),
        "l_t": l_t,
        "l_t_mode": l_t_mode,
        "lens_kind": lens_kind,
        "population": population_label,
        "temporal_layers": temporal_layers,
        "regime_thresholds": {
            "p_final_high":  p_final_high,
            "rs_strong_thr": rs_strong_thr,
            "rs_f3_thr":     rs_f3_thr,
            "peak_low_thr":  peak_low_thr,
        },
        "n": len(results),
        "regime_counts": regime_counts,
        "regime_pooled_f2": n_f2_pooled,
        "regime_note": (
            "f2_regime is DESCRIPTIVE only (absolute-threshold, Phi-3-calibrated, "
            "non-portable); the verdict uses DLA (authoritative) + rank (cross-check)."
        ),
        "dla": {
            "n_with_dla": n_dla,
            "n_dla_f2_writes": n_dla_f2,
            "n_dla_f1_no_write": n_dla_f1,
            "mean_dla_ht_sum": float(np.mean(dla_vals)) if n_dla else None,
            "median_dla_ht_sum": float(np.median(dla_vals)) if n_dla else None,
            "tau": DLA_F1F2_TAU,
        },
        "lens_decodable_coverage": {
            "n_decodable": n_decodable,
            "n_total": len(results),
            "coverage": coverage,
            "criterion": "first_rank_competitive_layer in [0, L_final)",
        },
        "mean_p_new_at_lt": float(np.mean(p_at_lt)),
        "mean_p_new_peak": float(np.mean(p_at_peak)),
        "mean_peak_layer": float(np.mean(peak_layers)),
        "mean_p_new_at_final": float(np.mean(p_at_final)),
        "mean_route_score": float(np.mean(route_scores)),
        "median_route_score": float(np.median(route_scores)),
        "mean_route_score_peak": float(np.mean(route_scores_peak)),
        "median_route_score_peak": float(np.median(route_scores_peak)),
        "pct_above_0.02_lt": float(sum(1 for s in route_scores if s > 0.02) / len(route_scores)),
        "pct_above_0.05_peak": float(
            sum(1 for s in route_scores_peak if s > 0.05) / len(route_scores_peak)
        ),
        "per_instance": [
            {
                "instance_id": r.instance_id,
                "p_new_at_temporal": r.p_new_at_temporal,
                "p_new_peak": r.p_new_peak,
                "peak_layer": r.peak_layer,
                "p_new_at_final": r.p_new_at_final,
                "route_score": r.route_score,
                "route_score_peak": r.route_score_peak,
                "f2_regime": r.f2_regime,           # descriptive only (demoted)
                "first_rank_competitive_layer": r.first_rank_competitive_layer,
                "rank_new_final": r.rank_new_final,
                "routed": r.routed,
                "dla_ht_sum": r.dla_ht_sum,
                "dla_per_head": r.dla_per_head,
                "dla_f1_vs_f2": r.dla_f1_vs_f2,
                # Full trajectory arrays (useful for plotting)
                "probs_new": r.trajectory.probs_new.tolist(),
                "probs_old": r.trajectory.probs_old.tolist(),
            }
            for r in results
        ],
    }
    with open(out_dir / "f2b_route_score.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return results


# ── F2-c: B5 vs B6 Behavioral Cross-Analysis ─────────────────────────────────

def run_f2c(
    model,
    b1_instances: list[dict],
    b5_instances: list[dict],
    b6_instances: list[dict],
    template: str,
    out_dir: Path,
    b1_success_map: dict[str, bool] | None = None,
):
    """Run F2-c B5×B6 (primary) + B1×B5 (supplementary) per methodology lines 334–349.

    Returns the populated :class:`BehavioralCrossResult` so the main
    driver can feed the per-instance details into the final F2-verdict
    assembly.
    """
    print("\n" + "=" * 60)
    print("F2-c: B5 vs B6 Behavioral Cross-Analysis")
    print("=" * 60)
    print("B1: single evidence, year intact")
    print("B5: dual evidence (old+new), year intact  → model must use year to select")
    print("B6: dual evidence (old+new), year STRIPPED → year-free baseline\n")

    result = behavioral_cross_analysis(
        model, b1_instances, b5_instances, b6_instances,
        b1_success_map=b1_success_map,
        template=template, verbose=True,
    )

    n = result.n_total

    print(f"\n  n = {n}")
    print(f"  B1 accuracy : {result.b1_accuracy:.1%}")
    print(f"  B5 accuracy : {result.b5_accuracy:.1%}")
    print(f"  B6 accuracy : {result.b6_accuracy:.1%}")
    print(f"  ΔB5−B6      : {(result.b5_accuracy - result.b6_accuracy):+.1%}  "
          "(positive ⇒ year token is a critical routing signal)")

    print("\n  ── PRIMARY F2 detector: per-instance B5 × B6 ───────────────────")
    print(f"    B5-success ∩ B6-fail    : {result.n_b5_success_b6_fail:>4d} "
          "  (year-dependent routing works; ANTI-F2 evidence)")
    print(f"    B5-fail    ∩ B6-fail    : {result.n_b5_fail_b6_fail:>4d} "
          "  (CANDIDATE F2 — year-based routing broken)")
    print(f"    B5-fail    ∩ B6-success : {result.n_b5_fail_b6_success:>4d} "
          "  (PARADOXICAL — year tokens hurt performance)")
    print(f"    B5-success ∩ B6-success : {result.n_b5_success_b6_success:>4d} "
          "  (year not necessary)")

    print("\n  ── SUPPLEMENTARY: B1 × B5 cross-reference ──────────────────────")
    print(f"    B1-fail    ∩ B5-success : {result.n_b1_fail_b5_success:>4d} "
          "  (single-passage persuasion too weak; rules out F2)")
    print(f"    B1-fail    ∩ B5-fail    : {result.n_b1_fail_b5_fail:>4d} "
          f"  (ambiguous; {result.n_b1_fail_b5_fail_rate:.1%} of B1-failures)")

    print("\n  ── 3-WAY DISAMBIGUATION: B1 × B5 × B6 (methodology line 347) ─")
    print(f"    B1-fail ∩ B5-fail    ∩ B6-fail    : "
          f"{result.n_b1f_b5f_b6f:>4d}   (PARAMETRIC DOMINANCE — likely not F2)")
    print(f"    B1-fail ∩ B5-fail    ∩ B6-success : "
          f"{result.n_b1f_b5f_b6s:>4d}   (year NOT required; year hurts)")
    print(f"    B1-fail ∩ B5-success ∩ B6-fail    : "
          f"{result.n_b1f_b5s_b6f:>4d}   (YEAR-DRIVEN RESCUE — rules out F2)")
    print(f"    B1-fail ∩ B5-success ∩ B6-success : "
          f"{result.n_b1f_b5s_b6s:>4d}   (dual-evidence rescue regardless of year)")

    # ── PRIMARY STATISTIC: McNemar paired test on the discordant cells ──────
    # B5 vs B6 are the SAME instances; the correct test is McNemar on the
    # discordant pairs (b = B5✓B6✗, c = B5✗B6✓), not the unpaired +ΔB5−B6 gap.
    mcnemar = mcnemar_test(
        result.n_b5_success_b6_fail,   # b: year helped
        result.n_b5_fail_b6_success,   # c: year hurt
    )
    print("\n  ── PRIMARY STATISTIC: McNemar paired test (B5 vs B6) ───────────")
    print(f"    Discordant: b(B5✓B6✗)={mcnemar['b_b5success_b6fail']}  "
          f"c(B5✗B6✓)={mcnemar['c_b5fail_b6success']}  "
          f"(n_disc={mcnemar['n_discordant']}, odds b/c={mcnemar['odds_b_over_c']:.1f})")
    if mcnemar.get("chi2") is not None:
        print(f"    χ²(cont.)={mcnemar['chi2']:.2f}  "
              f"p_chi2={mcnemar['p_chi2_continuity']:.3g}  "
              f"p_exact={mcnemar['p_exact']:.3g}  "
              f"(prefer: {mcnemar['prefer']})")

    if result.b5_accuracy > result.b6_accuracy + 0.05:
        print("\n  → B5 >> B6: year in evidence is a critical routing signal.")
    elif result.b5_accuracy <= result.b6_accuracy + 0.05:
        print("\n  → B5 ≈ B6: model does NOT use year token in evidence to route.")

    output = {
        "schema_version": F2_OUTPUT_SCHEMA_VERSION,
        "tatm_f2_diagnosis_path": inspect.getsourcefile(behavioral_cross_analysis),
        "n_total": n,
        "b1_accuracy": result.b1_accuracy,
        "b5_accuracy": result.b5_accuracy,
        "b6_accuracy": result.b6_accuracy,
        # PRIMARY statistic for F2-c (plan: mcnemar)
        "mcnemar_b5_vs_b6": mcnemar,
        # Primary (B5 × B6) — methodology lines 340–344
        "b5xb6_primary": {
            "n_b5_success_b6_fail":     result.n_b5_success_b6_fail,
            "n_b5_fail_b6_fail":        result.n_b5_fail_b6_fail,
            "n_b5_fail_b6_success":     result.n_b5_fail_b6_success,
            "n_b5_success_b6_success":  result.n_b5_success_b6_success,
            "rate_b5_fail_b6_fail":     result.n_b5_fail_b6_fail / n if n else 0.0,
            "rate_b5_success_b6_fail":  result.n_b5_success_b6_fail / n if n else 0.0,
        },
        # Supplementary (B1 × B5) — methodology lines 345–349
        "b1xb5_supplementary": {
            "n_b1_fail_b5_success":   result.n_b1_fail_b5_success,
            "n_b1_fail_b5_fail":      result.n_b1_fail_b5_fail,
            "n_b1_fail_b5_fail_rate": result.n_b1_fail_b5_fail_rate,
        },
        # 3-way disambiguation cells (methodology line 347)
        "b1xb5xb6_disambiguation": {
            "n_b1f_b5f_b6f":  result.n_b1f_b5f_b6f,   # parametric dominance
            "n_b1f_b5f_b6s":  result.n_b1f_b5f_b6s,   # year not required
            "n_b1f_b5s_b6f":  result.n_b1f_b5s_b6f,   # year-driven rescue
            "n_b1f_b5s_b6s":  result.n_b1f_b5s_b6s,   # robust dual-evidence rescue
            "rate_parametric_dominance_in_b1f_b5f": (
                result.n_b1f_b5f_b6f / result.n_b1_fail_b5_fail
                if result.n_b1_fail_b5_fail else 0.0
            ),
        },
        # Back-compat aliases (used by plot_f2_results.py and earlier readers)
        "n_b1_fail_b5_success": result.n_b1_fail_b5_success,
        "n_b1_fail_b5_fail":    result.n_b1_fail_b5_fail,
        "n_b1_fail_b5_fail_rate": result.n_b1_fail_b5_fail_rate,
        "details": result.details,
    }
    with open(out_dir / "f2c_b5_vs_b6.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F2 Diagnostic — Time Set but Not Routed")
    parser.add_argument("--data",  required=True, help="Path to JSONL (must contain B1, B5, B6)")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf",
                        help="HuggingFace model name (TransformerLens-compatible)")
    parser.add_argument(
        "--temporal-heads",
        help="Path to JSON file with top_heads or models, e.g. data/external/temporal_heads/paper_temporal_heads.json",
    )
    parser.add_argument(
        "--temporal-heads-manual", nargs="+", metavar="L,H",
        help="Manual temporal heads as layer,head pairs, e.g. 15,0 16,2",
    )
    parser.add_argument(
        "--temporal-heads-top-k", type=int, default=10,
        help="How many top heads to use from probe JSON (default: 10)",
    )
    parser.add_argument("--template", default="plain",
                        choices=["plain", "llama2", "llama3", "phi3", "qwen"])
    parser.add_argument("--out", default="results/f2_diagnostic", help="Output directory")
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"],
    )
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Limit number of instances (quick testing)")
    parser.add_argument(
        "-n", "--number", type=int, default=None,
        help="Randomly sample N Layer-1 facts and keep all aligned B1/B5/B6 Layer-2 triples",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=42,
        help="Random seed used with --number (default: 42)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=["f2a", "f2b", "f2c"],
        help="Skip sub-experiments",
    )
    parser.add_argument(
        "--no-reverts-filter", action="store_true",
        help=(
            "[deprecated, kept for back-compat] Equivalent to "
            "``--f2b-population all_b1``: skip REVERTS_OLD filter and run "
            "F2-b on all B1 instances."
        ),
    )
    parser.add_argument(
        "--f2b-population",
        default="reverts_old",
        choices=["reverts_old", "b1_failure", "all_b1"],
        help=(
            "Which population to run F2-b on. "
            "``reverts_old`` (default): B1-failure ∩ output=answer_old — "
            "the strictest behavioral proxy for methodology Part-II's "
            "PARAM_OLD panel.  "
            "``b1_failure``: all B1-failure instances (output ≠ answer_new) "
            "— the methodology-direct setting per line 332.  "
            "``all_b1``: every B1 instance (legacy; ``--no-reverts-filter`` "
            "alias)."
        ),
    )
    parser.add_argument(
        "--f2a-resid-sweep", action="store_true",
        help=(
            "[OPTIONAL] Also run the residual-stream layer-sweep STR patcher "
            "(broad→granular localization).  Only needed when head-level STR "
            "recovery is weak on a lens-decodable model; off by default."
        ),
    )
    parser.add_argument(
        "--f2a-resid-hook", default="resid_pre", choices=["resid_pre", "resid_post"],
        help="Residual hook point for --f2a-resid-sweep (default resid_pre).",
    )
    parser.add_argument(
        "--lens-kind", default="raw", choices=["raw", "tuned"],
        help=(
            "Logit-Lens flavour for F2-b's trajectory.  Methodology F3-a "
            "Step 3 specifies ``tuned`` as primary; ``tuned`` is not yet "
            "implemented and currently falls back to ``raw`` with a "
            "one-time warning."
        ),
    )
    parser.add_argument(
        "--trajectory-cache",
        help=(
            "Optional path to a previously saved ``f2b_route_score.json`` "
            "(or a compatible trajectory cache).  When provided, F2-b "
            "reuses ``probs_new`` / ``probs_old`` arrays per ``instance_id`` "
            "instead of re-running the lens forward pass — implements the "
            "methodology line 325 'trajectories already cached in F3-a' "
            "directive."
        ),
    )
    parser.add_argument(
        "--f1-results",
        help=(
            "Optional path to F1-a output JSON (typically "
            "results2/f1_diagnostic_1000_<tag>/f1a_sat_probe.json).  When provided, "
            "every F2 verdict is cross-referenced against F1-a Step 5's "
            "per-instance F1-positive verdict (methodology line 284): "
            "F1-positive ⇒ year not read ⇒ classified F1, not F2."
        ),
    )
    parser.add_argument(
        "--f1-percentile", type=int, default=25, choices=[20, 25, 33],
        help=(
            "Which F1-a Step 5 percentile to use for the F1 cross-"
            "reference (methodology default: 25)."
        ),
    )
    parser.add_argument(
        "--f1b-results",
        help=(
            "Optional path to F1-b output JSON (typically "
            "results2/f1_diagnostic_1000_<tag>/f1b_attention_comparison.json).  When "
            "provided, the population-level Mann-Whitney p-value is read "
            "and recorded as the boolean ``f1b_nonsignif`` if F1-b fails "
            "to confirm the old population premise. It does not alter the "
            "clean verdict label."
        ),
    )
    parser.add_argument(
        "--f1b-alpha", type=float, default=0.05,
        help="Significance threshold for F1-b's Mann-Whitney p (default 0.05).",
    )
    parser.add_argument(
        "--l-t-mode", default="median", choices=["median", "max", "min"],
        help=(
            "How to collapse multi-layer H_T into L_T for F2-b's "
            "RouteScore (methodology primary: median = $\\ell_{H_T}$)."
        ),
    )
    parser.add_argument(
        "--p-final-high", type=float, default=F2_DEFAULT_P_FINAL_HIGH,
        help="Final-P^L threshold above which the signal is considered to have survived (not_f2).",
    )
    parser.add_argument(
        "--rs-strong-thr", type=float, default=F2_DEFAULT_RS_STRONG_THR,
        help="Minimum peak RouteScore for a 'mid-rise to have occurred' (else F2-strong).",
    )
    parser.add_argument(
        "--rs-f3-thr", type=float, default=F2_DEFAULT_RS_F3_THR,
        help="Peak RouteScore above which the trajectory is classified as F3 candidate.",
    )
    parser.add_argument(
        "--peak-low-thr", type=float, default=F2_DEFAULT_PEAK_LOW_THR,
        help=(
            "Absolute P^l_peak threshold below which the trajectory is "
            "treated as 'never rises' → F2-strong (methodology line 290, "
            "gates the F2-strong classification on the *peak value* in "
            "addition to the peak-minus-final delta)."
        ),
    )
    args = parser.parse_args()

    # ── Resolve temporal heads ─────────────────────────────────────────────
    temporal_heads: list[tuple[int, int]] = []

    if args.temporal_heads_manual:
        for pair in args.temporal_heads_manual:
            l_str, h_str = pair.split(",")
            temporal_heads.append((int(l_str), int(h_str)))
        print(f"Temporal heads (manual): {temporal_heads}")
    elif args.temporal_heads:
        temporal_heads = load_temporal_heads(
            args.temporal_heads,
            top_k=args.temporal_heads_top_k,
            model_name=args.model,
        )
        print(f"Temporal heads (from probe, top-{args.temporal_heads_top_k}): {temporal_heads}")
    else:
        if "f2a" not in args.skip or "f2b" not in args.skip:
            parser.error(
                "F2-a and F2-b require temporal heads.  "
                "Provide --temporal-heads or --temporal-heads-manual, "
                "or skip both with --skip f2a f2b."
            )

    # ── Dtype ──────────────────────────────────────────────────────────────
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if args.dtype == "auto":
        _dev = args.device
        if _dev == "auto":
            _dev = (
                "cuda" if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        resolved_dtype = torch.float16 if _dev == "cuda" else torch.float32
    else:
        resolved_dtype = dtype_map[args.dtype]

    # ── Setup ──────────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model    : {args.model}")
    print(f"Data     : {args.data}")
    print(f"Template : {args.template}")
    print(f"Output   : {out_dir}")

    # ── Load data ──────────────────────────────────────────────────────────
    b1_instances, b5_instances, b6_instances = load_b1_b5_b6(args.data)

    if args.max_instances:
        b1_instances = b1_instances[:args.max_instances]
        b5_instances = b5_instances[:args.max_instances]
        b6_instances = b6_instances[:args.max_instances]
    if args.number:
        b1_instances, b5_instances, b6_instances = restrict_instances(
            b1_instances,
            b5_instances,
            b6_instances,
            number=args.number,
            seed=args.sample_seed,
        )

    print(f"\nLoaded (aligned): B1={len(b1_instances)}, B5={len(b5_instances)}, "
          f"B6={len(b6_instances)}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model {args.model} …")
    with tqdm(total=1, desc="Loading weights", unit="model",
              bar_format="{desc}: {elapsed} elapsed {postfix}") as pbar:
        model = load_model(args.model, device=args.device, dtype=resolved_dtype)
        pbar.set_postfix_str("done")
        pbar.update(1)
    print(f"  {model.cfg.n_layers} layers × {model.cfg.n_heads} heads  "
          f"d_model={model.cfg.d_model}")

    # ── Resolve F2-b population ────────────────────────────────────────────
    # Back-compat: --no-reverts-filter overrides --f2b-population to all_b1.
    if args.no_reverts_filter:
        f2b_population = "all_b1"
    else:
        f2b_population = args.f2b_population

    b1_success_map: dict[str, bool] | None = None
    reverts_old: list[dict]

    if f2b_population == "all_b1":
        reverts_old = b1_instances
        print(f"\n[F2-b population] all_b1: using all {len(reverts_old)} B1 instances.")
    else:
        # Run the REVERTS_OLD pass first (we need it for both reverts_old and
        # b1_failure populations, since both filter on B1 generation outcome).
        reverts_old_only, b1_success_map = run_reverts_old_filter(
            model, b1_instances, args.template, out_dir,
        )
        if f2b_population == "reverts_old":
            reverts_old = reverts_old_only
        else:  # b1_failure
            # Methodology line 332 says "B1-failure instances" generically.
            # Take every instance where the model did not output answer_new
            # on B1.  REVERTS_OTHER is included; REVERTS_OLD is included.
            reverts_old = [
                inst for inst in b1_instances
                if not b1_success_map.get(inst.get("instance_id", ""), False)
            ]
        if not reverts_old:
            print(f"\nNo instances in population '{f2b_population}'. Exiting.")
            return
        print(f"\n[F2-b population] {f2b_population}: "
              f"{len(reverts_old)} instances "
              f"(out of {len(b1_instances)} B1).")

    # ── Optional cross-script trajectory cache (methodology line 325) ──────
    cached_trajectories = None
    if args.trajectory_cache:
        try:
            from tatm.logit_lens import LogitTrajectory
            import numpy as _np

            with open(args.trajectory_cache) as fh:
                cache_data = json.load(fh)
            cache_rows = cache_data.get("per_instance", [])
            cached_trajectories = {}
            for row in cache_rows:
                iid = row.get("instance_id", "")
                if not iid or "probs_new" not in row or "probs_old" not in row:
                    continue
                cached_trajectories[iid] = LogitTrajectory(
                    probs_new=_np.asarray(row["probs_new"], dtype=float),
                    probs_old=_np.asarray(row["probs_old"], dtype=float),
                    logits_new=_np.zeros(len(row["probs_new"])),
                    logits_old=_np.zeros(len(row["probs_old"])),
                )
            print(f"[trajectory-cache] Loaded {len(cached_trajectories)} "
                  f"trajectories from {args.trajectory_cache}.")
        except Exception as exc:
            print(f"[trajectory-cache] WARNING: could not load "
                  f"{args.trajectory_cache} ({exc}); F2-b will recompute.")
            cached_trajectories = None

    # ── F2-a ───────────────────────────────────────────────────────────────
    if "f2a" not in args.skip:
        run_f2a(model, b5_instances, temporal_heads, args.template, out_dir)
        if args.f2a_resid_sweep:
            print("\n  [F2-a OPTIONAL] residual-stream layer sweep "
                  f"({args.f2a_resid_hook}) ...")
            resid_out = run_str_patching_resid_sweep(
                model, b5_instances, template=args.template,
                hook_point=args.f2a_resid_hook, verbose=True,
            )
            with open(out_dir / "f2a_resid_sweep.json", "w") as f:
                json.dump(resid_out, f, indent=2, ensure_ascii=False)
            if resid_out["per_layer"]:
                best = max(resid_out["per_layer"], key=lambda d: d["mean_recovery"])
                print(f"    Peak residual recovery at layer {best['layer']} "
                      f"(mean={best['mean_recovery']:+.3f}); wrote f2a_resid_sweep.json")

    # ── F2-b ───────────────────────────────────────────────────────────────
    route_results = []
    if "f2b" not in args.skip:
        route_results = run_f2b(
            model, reverts_old, temporal_heads, args.template, out_dir,
            population_label=f2b_population,
            l_t_mode=args.l_t_mode,
            lens_kind=args.lens_kind,
            p_final_high=args.p_final_high,
            rs_strong_thr=args.rs_strong_thr,
            rs_f3_thr=args.rs_f3_thr,
            peak_low_thr=args.peak_low_thr,
            cached_trajectories=cached_trajectories,
        )

    # ── F2-c ───────────────────────────────────────────────────────────────
    f2c_result = None
    if "f2c" not in args.skip:
        f2c_result = run_f2c(
            model, b1_instances, b5_instances, b6_instances,
            args.template, out_dir, b1_success_map=b1_success_map,
        )

    # ── F2 verdict assembly (methodology lines 282–294) ────────────────────
    # Per-instance verdict combining F2-b regime + F1-a Step-5 status.
    # The F1 cross-reference is *optional* — without it we report each
    # instance's regime with an ``_unverified`` suffix so downstream
    # readers know F1 ruling-out has not been applied.
    if route_results:
        f1_by_iid: dict[str, bool] = {}
        f1_by_key: dict[tuple, bool] = {}
        if args.f1_results:
            try:
                f1_by_iid, f1_by_key = load_f1_positive_map(
                    args.f1_results, percentile=args.f1_percentile,
                )
                print(f"\n[F1-a cross-reference] Loaded {len(f1_by_iid)} F1-a Step-5 "
                      f"verdicts from {args.f1_results} (percentile={args.f1_percentile}).")
            except Exception as exc:
                print(f"\n[F1-a cross-reference] WARNING: failed to load "
                      f"{args.f1_results} — proceeding without F1-a cross-reference. "
                      f"Reason: {exc}")

        # F1-b population consistency: if F1-b is non-significant, keep a soft
        # boolean annotation. Do not alter the clean verdict label.
        f1b_consistency: dict | None = None
        if args.f1b_results:
            try:
                f1b_consistency = load_f1_consistency(
                    args.f1b_results, alpha=args.f1b_alpha,
                )
                p = f1b_consistency.get("mann_whitney_p")
                if p is None:
                    print(f"\n[F1-b consistency] WARNING: {args.f1b_results} has "
                          "no Mann-Whitney p; consistency check skipped.")
                elif f1b_consistency["f1b_significant"]:
                    print(f"\n[F1-b consistency] OK: Mann-Whitney p={p:.4g} "
                          f"< α={args.f1b_alpha} — 'Time Set' premise holds.")
                else:
                    print(f"\n[F1-b consistency] WARNING: Mann-Whitney p={p:.4g} "
                          f"≥ α={args.f1b_alpha} — F1-b does NOT confirm the "
                          "'Time Set' population premise; recording soft "
                          "f1b_nonsignif=True without changing verdict labels.")
            except Exception as exc:
                print(f"\n[F1-b consistency] WARNING: failed to load "
                      f"{args.f1b_results} ({exc}); consistency check skipped.")
                f1b_consistency = None

        verdicts = assign_f2_verdicts(
            reverts_old, route_results,
            f1_by_iid=f1_by_iid, f1_by_key=f1_by_key,
            f1b_consistency=f1b_consistency,
        )

        verdict_counts: dict[str, int] = {}
        for v in verdicts:
            verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1

        print("\n  ── Final F2 verdict (per-instance, F1-cross-referenced) ─────")
        # Verdict labels collapsed: F2-strong/F2-weak → "F2"; f1b is now a
        # separate boolean field, not a suffix.  Display order: clean → unverified.
        verdict_label_order = []
        for base in [
            "F1", "F2", "F3_candidate",
            "not_routing_failure", "undetermined",
        ]:
            verdict_label_order.append(base)
            verdict_label_order.append(f"{base}_unverified")
        # Catch anything else we didn't enumerate
        for k in sorted(verdict_counts.keys()):
            if k not in verdict_label_order:
                verdict_label_order.append(k)
        for label in verdict_label_order:
            count = verdict_counts.get(label, 0)
            if count:
                print(f"    {label:>36s}: {count:>4d} "
                      f"({100*count/len(verdicts):5.1f}%)")
        n_f1b_nonsignif = sum(1 for v in verdicts if v.get("f1b_nonsignif"))
        if n_f1b_nonsignif:
            print(f"    [soft annotation] f1b_nonsignif=True on "
                  f"{n_f1b_nonsignif}/{len(verdicts)} instances "
                  "(premise now carried by F2-c McNemar, not F1-b).")

        verdict_output = {
            "schema_version": F2_OUTPUT_SCHEMA_VERSION,
            "tatm_f2_diagnosis_path": inspect.getsourcefile(assign_f2_verdicts),
            "verdict_label_policy": (
                "clean labels only; f1b_nonsignif is a separate boolean field"
            ),
            "n": len(verdicts),
            "f1_results_path":  args.f1_results,
            "f1_percentile":    args.f1_percentile if args.f1_results else None,
            "f1b_results_path": args.f1b_results,
            "f1b_alpha":        args.f1b_alpha if args.f1b_results else None,
            "f1b_consistency": (
                {
                    "mann_whitney_p":  f1b_consistency.get("mann_whitney_p"),
                    "alpha":           f1b_consistency.get("alpha"),
                    "f1b_significant": f1b_consistency.get("f1b_significant"),
                }
                if f1b_consistency is not None else None
            ),
            "l_t_mode":         args.l_t_mode,
            "lens_kind":        args.lens_kind,
            "f2b_population":   f2b_population,
            "regime_thresholds": {
                "p_final_high":  args.p_final_high,
                "rs_strong_thr": args.rs_strong_thr,
                "rs_f3_thr":     args.rs_f3_thr,
                "peak_low_thr":  args.peak_low_thr,
            },
            "verdict_counts": verdict_counts,
            "per_instance":   verdicts,
        }
        with open(out_dir / "f2_verdicts.json", "w") as f:
            json.dump(verdict_output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("F2 Diagnostic complete. Results saved to:", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
