#!/usr/bin/env python3
"""Per-model F1/F2/F3 *propensity* scores (graded, non-exclusive).

Design (see chat 2026-06-18): the three failure mechanisms are sequential gates
on one causal chain (F1 read -> F2 route -> F3 override), so we model them as
SOFT gates calibrated against the B1-success population (the "working pipeline"
null) and assign each failing instance a membership vector that sums to 1:

    g_read   = sigmoid((read_score   - read0  )/s_read )   # year attended?
    g_route  = sigmoid((route_score  - route0 )/s_route)   # answer_new written?
    g_over   = sigmoid((over_score   - over0  )/s_over )   # written-then-suppressed?

    P(F1)      = (1 - g_read)                      # not read
    P(F2)      = g_read * (1 - g_route)            # read but not routed/written
    P(F3)      = g_read * g_route * g_over         # routed then overridden
    P(through) = g_read * g_route * (1 - g_over)   # signal survived (other/uncaught)

Two reported scores per model per mechanism:
  * conditional propensity  pi_F  = mean P(F | B1-failure)         (composition)
  * absolute propensity     rho_F = pi_F * failure_rate            (per all-B1)

Plus an Override-Dominance Index ODI = pi_F3 / (pi_F1+pi_F2+pi_F3) and the
mean membership entropy (a scalar measure of NON-exclusivity / superposition).

Signals (all already stored; no model rerun):
  read_score  = mean H_T attention to year tokens   (f1a_sat_probe: step5 scalar)
  route_score = peak P^l(answer_new) over layers     (f3a_trajectory: p_new_peak_all)
  over_score  = suppression drop peak->final         (f3a_trajectory: suppression_drop)

CAVEATS (report alongside): the read axis (attention) separates success/failure
only weakly, so the F1 vs F2 split is the least reliable; F3 (override) is the
robust, causally-corroborated axis (F3 head-ablation Delta). DLA is direct-effects
only. Gates are calibrated to B1-success percentiles (transparent, not learned).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(p: Path):
    with open(p) as fh:
        return json.load(fh)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _robust_scale(x: np.ndarray) -> float:
    """1.4826 * MAD (fallback to std), floored to avoid divide-by-zero."""
    if x.size == 0:
        return 1.0
    mad = float(np.median(np.abs(x - np.median(x))))
    s = 1.4826 * mad if mad > 0 else float(np.std(x))
    return s if s > 1e-12 else 1e-12


def build_instances(f1a: dict, f3a: dict):
    """Join f1a attention + f3a trajectory by fact_id.

    The two stages use different prompts (B5 closed-book SAT probe vs B1
    conflict trajectory), so instance_id hashes differ; fact_id is the stable
    shared key (unique per stage).

    Returns (failures, success_ref) where each is a list of dicts with keys
    read/route/over (read may be None when attention is unavailable).
    """
    attn = {}
    scal = (f1a.get("step5_f1_positive") or {}).get("scalar_per_instance") or []
    meta = f1a.get("instance_meta") or []
    for m, s in zip(meta, scal):
        attn[m["fact_id"]] = float(s)

    failures, success = [], []
    for r in f3a.get("per_instance", []):
        rec = {
            "instance_id": r.get("instance_id"),
            "fact_id": r.get("fact_id"),
            "read": attn.get(r.get("fact_id")),              # may be None
            "route": float(r.get("p_new_peak_all", 0.0)),    # peak P(answer_new)
            "over": float(r.get("suppression_drop", 0.0)),   # peak - final
            "b1_success": bool(r.get("b1_success")),
        }
        (success if rec["b1_success"] else failures).append(rec)
    return failures, success


def calibrate(success: list[dict], failures: list[dict]):
    """Gate midpoints from the B1-success ('working pipeline') null; gate SCALES
    from the pooled (success+failure) spread so a degenerate null population
    (e.g. success peaks all ~0.99) cannot collapse the scale to ~0."""
    def _pool(key, src):
        return np.array([s[key] for s in src if s[key] is not None], float)

    reads_s = _pool("read", success)
    reads_all = np.concatenate([reads_s, _pool("read", failures)]) if reads_s.size else reads_s
    routes_s = _pool("route", success)
    routes_all = np.concatenate([routes_s, _pool("route", failures)])
    overs_s = _pool("over", success)
    overs_all = np.concatenate([overs_s, _pool("over", failures)])
    return {
        # "not read" if attention below the 25th pct of working cases
        "read0": float(np.percentile(reads_s, 25)) if reads_s.size else 0.0,
        "s_read": _robust_scale(reads_all),
        # "written" if peak below the 25th pct of working peaks
        "route0": float(np.percentile(routes_s, 25)) if routes_s.size else 0.0,
        "s_route": _robust_scale(routes_all),
        # "overridden" if suppressed MORE than the 75th pct of working cases
        "over0": float(np.percentile(overs_s, 75)) if overs_s.size else 0.0,
        "s_over": _robust_scale(overs_all),
        "n_read_ref": int(reads_s.size),
    }


def membership(failures: list[dict], cal: dict):
    rows = []
    read_default = 0.5  # neutral when attention missing (flagged via coverage)
    for r in failures:
        if r["read"] is None:
            g_read = read_default
        else:
            g_read = float(_sigmoid((r["read"] - cal["read0"]) / cal["s_read"]))
        g_route = float(_sigmoid((r["route"] - cal["route0"]) / cal["s_route"]))
        g_over = float(_sigmoid((r["over"] - cal["over0"]) / cal["s_over"]))
        p_f1 = (1 - g_read)
        p_f2 = g_read * (1 - g_route)
        p_f3 = g_read * g_route * g_over
        p_through = g_read * g_route * (1 - g_over)
        rows.append({
            "instance_id": r["instance_id"],
            "read_available": r["read"] is not None,
            "g_read": g_read, "g_route": g_route, "g_over": g_over,
            "P_F1": p_f1, "P_F2": p_f2, "P_F3": p_f3, "P_through": p_through,
        })
    return rows


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def summarize(rows: list[dict], n_b1_total: int, model_tag: str):
    M = np.array([[r["P_F1"], r["P_F2"], r["P_F3"], r["P_through"]] for r in rows])
    n_fail = len(rows)
    pi = M.mean(axis=0) if n_fail else np.zeros(4)
    fail_rate = n_fail / n_b1_total if n_b1_total else 0.0
    rho = pi * fail_rate
    tri = pi[:3]
    odi = float(pi[2] / tri.sum()) if tri.sum() > 0 else 0.0
    ent = float(np.mean([_entropy(row[:3] / row[:3].sum() if row[:3].sum() > 0 else row[:3])
                         for row in M])) if n_fail else 0.0
    max_ent = np.log(3)
    dominant = {f: int((M[:, :3].argmax(axis=1) == i).sum())
                for i, f in enumerate(["F1", "F2", "F3"])}
    mixed = int((M[:, :3].max(axis=1) < 0.5).sum())
    return {
        "model_tag": model_tag,
        "n_b1_total": n_b1_total,
        "n_failures": n_fail,
        "failure_rate": fail_rate,
        "conditional_propensity_pi": {
            "F1": float(pi[0]), "F2": float(pi[1]),
            "F3": float(pi[2]), "through": float(pi[3])},
        "absolute_propensity_rho": {
            "F1": float(rho[0]), "F2": float(rho[1]),
            "F3": float(rho[2]), "through": float(rho[3])},
        "override_dominance_index": odi,
        "mean_membership_entropy": ent,
        "mean_membership_entropy_normalized": ent / max_ent if max_ent else 0.0,
        "dominant_label_counts": dominant,
        "n_genuinely_mixed_no_majority": mixed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f1-dir", required=True)
    ap.add_argument("--f3-dir", required=True)
    ap.add_argument("--model-tag", default="model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    f1a = _load(Path(args.f1_dir) / "f1a_sat_probe.json")
    f3a = _load(Path(args.f3_dir) / "f3a_trajectory.json")

    failures, success = build_instances(f1a, f3a)
    n_b1_total = len(failures) + len(success)
    cal = calibrate(success, failures)
    rows = membership(failures, cal)
    summ = summarize(rows, n_b1_total, args.model_tag)
    summ["calibration"] = cal
    cov = np.mean([r["read_available"] for r in rows]) if rows else 0.0
    summ["read_axis_coverage"] = float(cov)

    pi = summ["conditional_propensity_pi"]
    rho = summ["absolute_propensity_rho"]
    print(f"\n=== Mechanism propensity — {args.model_tag} "
          f"(B1: {summ['n_failures']} fail / {n_b1_total} total, "
          f"fail-rate {summ['failure_rate']:.1%}) ===")
    print(f"{'':12s}{'pi (|fail)':>12s}{'rho (|all)':>12s}")
    for f in ("F1", "F2", "F3", "through"):
        print(f"{f:12s}{pi[f]:>12.3f}{rho[f]:>12.3f}")
    print(f"\nOverride-Dominance Index (F3 / (F1+F2+F3)): {summ['override_dominance_index']:.3f}")
    print(f"Mean membership entropy (norm, 0=pure 1=maximally mixed): "
          f"{summ['mean_membership_entropy_normalized']:.3f}")
    print(f"Dominant-label counts: {summ['dominant_label_counts']}  "
          f"(genuinely mixed, no >0.5: {summ['n_genuinely_mixed_no_majority']})")
    print(f"Read-axis (attention) coverage: {cov:.1%}  "
          f"[F1 vs F2 split reliability scales with this]")

    out = Path(args.out) if args.out else (Path(args.f3_dir) / "mechanism_propensity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"summary": summ, "per_instance": rows}, fh, indent=1)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
