#!/usr/bin/env python3
"""Phase 1 post-hoc reanalysis of F1-c attention-knockout results.

The shipped F1-c metric ``p_new_drop_relative`` tracks only ``p(answer_new)``.
On the B5 dual-evidence population that hides the disambiguation effect: when the
year is knocked out the model often reverts to ``answer_old`` (``p_old`` rises)
while ``p_new`` barely moves, so a ``p_new``-only readout reports the wrong sign.

The fix is the logit difference.  Because softmax shares one log-partition term,
it cancels exactly:

    logit_new - logit_old == log p_new - log p_old

so the corrected per-instance metric can be recomputed *directly from the stored
probabilities* without re-running the model:

    ld_drop = (log p_new_clean - log p_old_clean)
            - (log p_new_ko    - log p_old_ko)

``ld_drop > 0`` means the year knockout reduced answer_new's advantage over
answer_old, i.e. the year *mattered* (the F1-correct direction).

Usage
-----
    python scripts/recompute_f1c_logit_diff.py \
        --f1c results2/f1_diagnostic_1000_phi3/f1c_attention_knockout.json \
        --out results2/f1_diagnostic_1000_phi3/f1c_logit_diff_recompute.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EPS = 1e-12

# Bucket thresholds (absolute prob change at the prediction position).
DISAMBIG_TAU = 0.05   # p_old rose by >= this -> year was disambiguating
RENORM_TAU = 0.05     # p_new rose by >= this -> renormalisation-to-context
CEILING_TAU = 0.01    # p_old_clean < this -> answer_old is not a competitor


def _log(p: float) -> float:
    return math.log(max(float(p), EPS))


def _ld_drop(e: dict) -> float | None:
    """logit-diff drop = (log p_new - log p_old)_clean - (..)_knockout."""
    keys = ("p_new_clean", "p_old_clean", "p_new_knockout", "p_old_knockout")
    if not all(k in e for k in keys):
        return None
    ld_clean = _log(e["p_new_clean"]) - _log(e["p_old_clean"])
    ld_ko = _log(e["p_new_knockout"]) - _log(e["p_old_knockout"])
    return ld_clean - ld_ko


def _bucket(e: dict) -> str:
    """Classify why the p_new-only metric is misleading for this instance."""
    p_old_c = float(e.get("p_old_clean", 0.0))
    p_old_k = float(e.get("p_old_knockout", 0.0))
    p_new_c = float(e.get("p_new_clean", 0.0))
    p_new_k = float(e.get("p_new_knockout", 0.0))
    if p_old_c < CEILING_TAU:
        return "ceiling_old_absent"
    if (p_old_k - p_old_c) >= DISAMBIG_TAU:
        return "disambiguation_old_rose"
    if (p_new_k - p_new_c) >= RENORM_TAU:
        return "renorm_new_rose"
    return "other"


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _competitor_subset(entries: list[dict]) -> list[dict]:
    """Instances where answer_old is a genuine competitor (clean p_old above floor)."""
    return [e for e in entries if float(e.get("p_old_clean", 0.0)) >= CEILING_TAU]


def analyze_population(entries: list[dict]) -> dict:
    ld_all: list[float] = []
    old_metric: list[float] = []
    buckets: dict[str, int] = {}
    for e in entries:
        ld = _ld_drop(e)
        if ld is not None:
            ld_all.append(ld)
        if "p_new_drop_relative" in e:
            old_metric.append(float(e["p_new_drop_relative"]))
        buckets[_bucket(e)] = buckets.get(_bucket(e), 0) + 1

    comp = _competitor_subset(entries)
    ld_comp = [v for e in comp if (v := _ld_drop(e)) is not None]

    return {
        "n": len(entries),
        "old_metric_p_new_drop_relative": {
            "mean": _mean(old_metric),
            "median": _median(old_metric),
            "note": "shipped (misleading) metric; negative = year-KO raised p_new",
        },
        "new_metric_logit_diff_drop": {
            "n": len(ld_all),
            "mean": _mean(ld_all),
            "median": _median(ld_all),
            "frac_positive": _mean([1.0 if v > 0 else 0.0 for v in ld_all]),
            "note": "positive = year-KO reduced new-over-old advantage = year matters",
        },
        "new_metric_competitor_restricted": {
            "n": len(ld_comp),
            "mean": _mean(ld_comp),
            "median": _median(ld_comp),
            "frac_positive": _mean([1.0 if v > 0 else 0.0 for v in ld_comp]),
            "note": f"only instances with p_old_clean >= {CEILING_TAU}",
        },
        "buckets": buckets,
    }


def _load_populations(data: dict) -> dict[str, list[dict]]:
    pops = data.get("populations", {})
    out: dict[str, list[dict]] = {}
    for label in ("b1_success", "b1_failure"):
        block = pops.get(label, {})
        entries = block.get("per_instance_full", [])
        if entries:
            out[label] = entries
    if not out and data.get("per_instance_full"):
        out["b1_success"] = data["per_instance_full"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1c", required=True, help="f1c_attention_knockout.json path")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    data = json.loads(Path(args.f1c).read_text(encoding="utf-8"))
    populations = _load_populations(data)
    if not populations:
        raise SystemExit("[ERROR] no per_instance_full populations found in F1-c JSON")

    report = {label: analyze_population(entries) for label, entries in populations.items()}

    print("=" * 72)
    print("F1-c logit-difference recompute  (log p_new - log p_old)")
    print("=" * 72)
    for label, r in report.items():
        old = r["old_metric_p_new_drop_relative"]
        new = r["new_metric_logit_diff_drop"]
        comp = r["new_metric_competitor_restricted"]
        print(f"\n[{label}]  n={r['n']}")
        print(f"  OLD  p_new_drop_relative : mean={old['mean']:+.3f}  median={old['median']:+.3f}")
        print(f"  NEW  logit_diff_drop     : mean={new['mean']:+.3f}  median={new['median']:+.3f}"
              f"  frac>0={new['frac_positive']:.2f}  (n={new['n']})")
        print(f"  NEW  competitor-restr.   : mean={comp['mean']:+.3f}  median={comp['median']:+.3f}"
              f"  frac>0={comp['frac_positive']:.2f}  (n={comp['n']})")
        print(f"  buckets: {r['buckets']}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
