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
  1. Run run_f1_diagnostic.py first to identify temporal_heads.
  2. Layer-2 JSONL must contain B1, B5, and B6 instance types.
     Build with:  python scripts/build_wikidata_layer2.py --layers B1 B3 B5 B6

Usage
-----
    python scripts/run_f2_diagnostic.py \\
        --data  data/processed/wikidata_layer2.jsonl \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --temporal-heads results/f1_diagnostic/f1a_sat_probe.json \\
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
    behavioral_cross_analysis,
    compute_route_scores,
    run_str_patching,
)
from tatm.model import (
    build_prompt,
    check_match,
    generate_answer,
    load_model,
)


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
) -> list[tuple[int, int]]:
    """Read temporal heads from F1 SAT probe output JSON.

    The file is ``f1a_sat_probe.json`` produced by ``run_f1_diagnostic.py``.
    """
    with open(probe_json) as f:
        data = json.load(f)
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

    recoveries = [r.recovery_fraction for r in results]
    mean_rec   = float(np.mean(recoveries))
    med_rec    = float(np.median(recoveries))
    pct_50     = sum(1 for r in recoveries if r >= 0.5) / len(recoveries)

    print(f"\n  Instances with valid patching trials: {len(results)}")
    print(f"  Mean  recovery fraction: {mean_rec:+.3f}")
    print(f"  Median recovery fraction: {med_rec:+.3f}")
    print(f"  Fraction ≥ 50% recovery (causal): {pct_50:.1%}")

    output = {
        "n_patched": len(results),
        "mean_recovery": mean_rec,
        "median_recovery": med_rec,
        "pct_ge_50_recovery": pct_50,
        "per_instance": [
            {
                "instance_id": r.instance_id,
                "logit_diff_clean": r.logit_diff_clean,
                "logit_diff_corrupted": r.logit_diff_corrupted,
                "logit_diff_patched": r.logit_diff_patched,
                "recovery_fraction": r.recovery_fraction,
                "top1_clean": r.top1_clean,
                "top1_corrupted": r.top1_corrupted,
                "top1_patched": r.top1_patched,
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
) -> None:
    print("\n" + "=" * 60)
    print("F2-b: RouteScore  (Temporal Signal Attenuation)")
    print("=" * 60)

    temporal_layers = sorted({l for l, _ in temporal_heads})
    if not temporal_layers:
        print("  No temporal head layers — skipping RouteScore.")
        return

    print(f"  Temporal head layers: {temporal_layers}")
    print(f"  L_T = {max(temporal_layers)}  (layer where P^L_T is measured)\n")

    results = compute_route_scores(
        model, reverts_old, temporal_layers, template=template, verbose=True,
    )

    if not results:
        print("  No RouteScore results (check answer token lookup).")
        return

    route_scores      = [r.route_score for r in results]
    route_scores_peak = [r.route_score_peak for r in results]
    p_at_lt           = [r.p_new_at_temporal for r in results]
    p_at_peak         = [r.p_new_peak for r in results]
    p_at_final        = [r.p_new_at_final for r in results]
    peak_layers       = [r.peak_layer for r in results]

    print(f"\n  n = {len(results)}")
    print(f"  Mean  P(answer_new) at L_T:    {np.mean(p_at_lt):.4f}")
    print(f"  Mean  P(answer_new) peak:      {np.mean(p_at_peak):.4f}  "
          f"(avg peak layer: {np.mean(peak_layers):.1f})")
    print(f"  Mean  P(answer_new) at L_final:{np.mean(p_at_final):.4f}")
    print(f"  Mean  RouteScore (L_T basis):  {np.mean(route_scores):+.4f}")
    print(f"  Mean  RouteScore (peak basis):  {np.mean(route_scores_peak):+.4f}")
    print(f"  Fraction peak RouteScore > 0.05 (F2 signal): "
          f"{sum(1 for s in route_scores_peak if s > 0.05)/len(route_scores_peak):.1%}")

    output = {
        "l_t": max(temporal_layers),
        "temporal_layers": temporal_layers,
        "n": len(results),
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
                # Full trajectory arrays (useful for plotting)
                "probs_new": r.trajectory.probs_new.tolist(),
                "probs_old": r.trajectory.probs_old.tolist(),
            }
            for r in results
        ],
    }
    with open(out_dir / "f2b_route_score.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ── F2-c: B5 vs B6 Behavioral Cross-Analysis ─────────────────────────────────

def run_f2c(
    model,
    b1_instances: list[dict],
    b5_instances: list[dict],
    b6_instances: list[dict],
    template: str,
    out_dir: Path,
    b1_success_map: dict[str, bool] | None = None,
) -> None:
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

    print(f"\n  n = {result.n_total}")
    print(f"  B1 accuracy : {result.b1_accuracy:.1%}")
    print(f"  B5 accuracy : {result.b5_accuracy:.1%}")
    print(f"  B6 accuracy : {result.b6_accuracy:.1%}")
    print(f"\n  B1-fail ∩ B5-success : {result.n_b1_fail_b5_success}  "
          f"(persuasion too weak, not F2)")
    print(f"  B1-fail ∩ B5-fail    : {result.n_b1_fail_b5_fail}  "
          f"(strong F2 signal — {result.n_b1_fail_b5_fail_rate:.1%} of B1-failures)")

    if result.b5_accuracy > result.b6_accuracy + 0.05:
        print("\n  → B5 >> B6: year in evidence is a critical routing signal.")
    elif result.b5_accuracy <= result.b6_accuracy + 0.05:
        print("\n  → B5 ≈ B6: model does NOT use year token in evidence to route.")

    output = {
        "n_total": result.n_total,
        "b1_accuracy": result.b1_accuracy,
        "b5_accuracy": result.b5_accuracy,
        "b6_accuracy": result.b6_accuracy,
        "n_b1_fail_b5_success": result.n_b1_fail_b5_success,
        "n_b1_fail_b5_fail": result.n_b1_fail_b5_fail,
        "n_b1_fail_b5_fail_rate": result.n_b1_fail_b5_fail_rate,
        "details": result.details,
    }
    with open(out_dir / "f2c_b5_vs_b6.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F2 Diagnostic — Time Set but Not Routed")
    parser.add_argument("--data",  required=True, help="Path to JSONL (must contain B1, B5, B6)")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf",
                        help="HuggingFace model name (TransformerLens-compatible)")
    parser.add_argument(
        "--temporal-heads",
        help="Path to f1a_sat_probe.json from run_f1_diagnostic.py",
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
        help="Skip REVERTS_OLD filter; run F2-b on all B1 instances",
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
            args.temporal_heads, top_k=args.temporal_heads_top_k,
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

    # ── REVERTS_OLD filter ─────────────────────────────────────────────────
    b1_success_map: dict[str, bool] | None = None
    if args.no_reverts_filter:
        reverts_old = b1_instances
        print(f"\n[--no-reverts-filter] Using all {len(reverts_old)} B1 instances for F2-b.")
    else:
        reverts_old, b1_success_map = run_reverts_old_filter(
            model, b1_instances, args.template, out_dir,
        )
        if not reverts_old:
            print("\nNo REVERTS_OLD instances found. Exiting.")
            return

    # ── F2-a ───────────────────────────────────────────────────────────────
    if "f2a" not in args.skip:
        run_f2a(model, b5_instances, temporal_heads, args.template, out_dir)

    # ── F2-b ───────────────────────────────────────────────────────────────
    if "f2b" not in args.skip:
        run_f2b(model, reverts_old, temporal_heads, args.template, out_dir)

    # ── F2-c ───────────────────────────────────────────────────────────────
    if "f2c" not in args.skip:
        run_f2c(model, b1_instances, b5_instances, b6_instances,
                args.template, out_dir, b1_success_map=b1_success_map)

    print("\n" + "=" * 60)
    print("F2 Diagnostic complete. Results saved to:", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
