#!/usr/bin/env python3
"""Causal mechanism profile — comparable, standardized causal effect sizes for
F1 / F2 / F3, the rigorous replacement for the lens-based propensity ODI.

Rationale (chat 2026-06-18). The three mechanisms are measured by different
instruments, so they are made comparable via **standardized effect sizes**
(Cohen's d family) — the textbook way to compare effects on different metrics
(Cohen; Hedges; Hopkins 2024) — and, where the field-native scale applies, a
**normalized recovered-logit-difference fraction** (Wang et al. 2023 IOI;
Zhang et al. 2023; Heimersheim & Nanda 2024).

Each axis is a *causal* effect on the answer_new − answer_old logit difference,
standardized against that axis's natural null:

  F1  read/use   = year-attention knockout LD-drop vs random-layer control
                   (paired d).  Large +d ⇒ year IS causally used ⇒ argues
                   AGAINST F1; F1 is the *absence* of this effect (a null
                   mechanism, reported as a lower bound — see note).
  F2  route/write= H_T direct logit attribution onto (new − old) vs 0
                   (one-sample d; DLA's natural null is 0 = no contribution,
                   so no success cohort is required).
  F3  override   = F3-head ablation effect, failure vs B1-success null
                   (two-sample d; the success null removes the late-
                   crystallization baseline).

THE RED LINE (Heimersheim & Nanda 2024; learnmechinterp): recovered/standardized
causal effects are **comparable and rankable but NOT a partition of credit** —
they need not sum to 1 and must not be turned into a share like ODI=F3/(F1+F2+F3).
We therefore report **ranked pairwise contrasts with bootstrap CIs**
(d_override − d_write, etc.) instead of a normalized share.

The gap-normalized recovered-LD fraction (effect / [LD_success − LD_failure])
is emitted ONLY when a B1-success raw logit-diff is available; it is currently
left null (no model logs LD on B1-success — a cheap rerun would fill it).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path):
    with open(path) as fh:
        return json.load(fh)


def _pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float(np.std(np.concatenate([a, b]))) or 1e-9
    s = ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    return float(np.sqrt(s)) or 1e-9


def cohens_d(effect: np.ndarray, control: np.ndarray | None,
             paired: bool) -> float:
    """One-sample (control=None, null=0), paired, or two-sample Cohen's d."""
    effect = np.asarray(effect, float)
    if control is None:                      # one-sample vs 0
        sd = float(np.std(effect, ddof=1)) or 1e-9
        return float(np.mean(effect) / sd)
    control = np.asarray(control, float)
    if paired:                               # paired difference
        diff = effect - control
        sd = float(np.std(diff, ddof=1)) or 1e-9
        return float(np.mean(diff) / sd)
    sd = _pooled_sd(effect, control)         # two-sample
    return float((np.mean(effect) - np.mean(control)) / sd)


def boot_d(effect: np.ndarray, control: np.ndarray | None, paired: bool,
           n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of Cohen's d (returns the n_boot draws)."""
    effect = np.asarray(effect, float)
    out = np.empty(n_boot)
    n_e = len(effect)
    if control is not None and paired:
        diff = effect - np.asarray(control, float)
        for i in range(n_boot):
            s = diff[rng.integers(0, n_e, n_e)]
            sd = np.std(s, ddof=1) or 1e-9
            out[i] = np.mean(s) / sd
        return out
    if control is None:
        for i in range(n_boot):
            s = effect[rng.integers(0, n_e, n_e)]
            sd = np.std(s, ddof=1) or 1e-9
            out[i] = np.mean(s) / sd
        return out
    control = np.asarray(control, float); n_c = len(control)
    for i in range(n_boot):
        e = effect[rng.integers(0, n_e, n_e)]
        c = control[rng.integers(0, n_c, n_c)]
        out[i] = (np.mean(e) - np.mean(c)) / _pooled_sd(e, c)
    return out


def _ci(draws: np.ndarray, alpha: float = 0.05) -> list[float]:
    return [float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2)))]


def _axis(name: str, label: str, effect: np.ndarray, control, paired: bool,
          interp: str, rng, n_boot: int) -> dict:
    draws = boot_d(effect, control, paired, n_boot, rng)
    d = cohens_d(effect, control, paired)
    ci = _ci(draws)
    return {
        "name": name, "label": label, "d": d, "d_CI": ci,
        "n_effect": int(len(effect)),
        "raw_effect_mean": float(np.mean(effect)),
        "control_mean": (None if control is None else float(np.mean(control))),
        "null": ("0 (no direct contribution)" if control is None
                 else "random-layer control" if paired else "B1-success null"),
        "interpretation": interp,
        "gap_fraction": None,   # pending: needs LD on B1-success (cheap rerun)
        "_draws": draws,        # internal, popped before save
    }


def build_profile(f1_dir: Path, f2_dir: Path, f3_dir: Path,
                  model_tag: str, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    axes: list[dict] = []

    # ── F1: year-attention knockout LD-drop vs random-layer control (paired) ──
    f1c = _load(f1_dir / "f1c_attention_knockout.json")
    pf = f1c.get("per_instance_full", [])

    def _rand_ld(rc: dict | None):
        # per-instance random-layer control = mean ld_drop over its random samples
        if not isinstance(rc, dict):
            return None
        d = [s.get("ld_drop") for s in rc.get("details", [])
             if s.get("ld_drop") is not None]
        return float(np.mean(d)) if d else None

    ld, rc = [], []
    for r in pf:
        c = _rand_ld(r.get("rand_control"))
        if r.get("ld_drop") is not None and c is not None:
            ld.append(float(r["ld_drop"])); rc.append(c)
    ld = np.array(ld, float); rc = np.array(rc, float)
    if len(ld) and len(rc) and len(ld) == len(rc):
        axes.append(_axis(
            "F1", "F1 read/use\n(year-KO LD-drop vs rand)", ld, rc, True,
            "causal necessity of year attention; ~0/neg d ⇒ year not a clean "
            "causal lever for the new-old margin (weak read evidence). F1 is a "
            "null/absence mechanism — report as a lower bound, not magnitude-"
            "compared to F3", rng, n_boot))

    # ── F2: H_T direct logit attribution onto (new-old) vs 0 (one-sample) ──
    f2 = _load(f2_dir / "f2_verdicts.json")
    dla = np.array([r["dla_ht_sum"] for r in f2.get("per_instance", [])
                    if r.get("dla_ht_sum") is not None], float)
    if len(dla):
        axes.append(_axis(
            "F2", "F2 route/write\n(H_T DLA new-old vs 0)", dla, None, False,
            "+d ⇒ heads write the new-answer direction (set & routed)",
            rng, n_boot))

    # ── F3: head-ablation effect, failure vs B1-success null (two-sample) ──
    fa = _load(f3_dir / "f3_head_ablation.json")
    fe = np.array(fa.get("failure_effects", []), float)
    se = np.array(fa.get("success_effects", []), float)
    if len(fe) and len(se):
        axes.append(_axis(
            "F3", "F3 override\n(ablation Δ fail vs success)", fe, se, False,
            "+d ⇒ ablating F3 heads moves failures more than successes "
            "(parametric override present)", rng, n_boot))

    # ── recovered-LD fractions from the clean success<->failure gaps ──
    gap_path = f3_dir / "f3_ld_gap.json"
    gaps = _load(gap_path) if gap_path.exists() else None
    if gaps:
        g_no = gaps.get("gap_new_old")
        g_pn = gaps.get("gap_param_new")
        for a in axes:
            if a["name"] in ("F1", "F2") and g_no:
                a["gap_fraction"] = float(a["raw_effect_mean"] / g_no)
                a["gap_contrast"] = "new_old"
            elif a["name"] == "F3" and g_pn:
                num = a["raw_effect_mean"] - (a["control_mean"] or 0.0)
                a["gap_fraction"] = float(num / g_pn)
                a["gap_contrast"] = "param_new"

    # ── ranked pairwise contrasts (NOT a share; comparable magnitudes) ──
    contrasts = []
    by = {a["name"]: a for a in axes}
    for hi, lo in (("F3", "F2"), ("F3", "F1"), ("F2", "F1")):
        if hi in by and lo in by:
            diff = by[hi]["_draws"] - by[lo]["_draws"]
            ci = _ci(diff)
            contrasts.append({
                "contrast": f"d_{hi} - d_{lo}",
                "estimate": float(by[hi]["d"] - by[lo]["d"]),
                "CI": ci,
                "significant": bool(ci[0] > 0 or ci[1] < 0),
            })

    # ranking by |d| (magnitude of causal footprint; sign kept in each axis)
    ranking = sorted([(a["name"], abs(a["d"])) for a in axes],
                     key=lambda t: t[1], reverse=True)

    for a in axes:
        a.pop("_draws", None)
    return {
        "model_tag": model_tag,
        "scale": "Cohen's d vs natural null (paired/one-sample/two-sample)",
        "n_boot": n_boot,
        "ld_gaps": (None if not gaps else
                    {"gap_new_old": gaps.get("gap_new_old"),
                     "gap_param_new": gaps.get("gap_param_new"),
                     "n_success": gaps.get("n_success"),
                     "n_failure": gaps.get("n_failure")}),
        "axes": axes,
        "pairwise_contrasts": contrasts,
        "ranking_by_abs_d": [{"axis": n, "abs_d": v} for n, v in ranking],
        "note": ("Comparable & rankable, NOT a partition: effects need not sum "
                 "to 1 (Heimersheim & Nanda 2024). No ODI share is reported; "
                 "dominance is the F3-vs-F2 contrast CI."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f1-dir", required=True)
    ap.add_argument("--f2-dir", required=True)
    ap.add_argument("--f3-dir", required=True)
    ap.add_argument("--model-tag", default="model")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prof = build_profile(Path(args.f1_dir), Path(args.f2_dir), Path(args.f3_dir),
                         args.model_tag, args.n_boot, args.seed)

    print(f"\n=== Causal mechanism profile — {args.model_tag} "
          f"(scale: {prof['scale']}) ===")
    print(f"{'axis':6s}{'Cohen d':>10s}{'95% CI':>22s}{'gap_frac':>10s}{'n':>7s}")
    for a in prof["axes"]:
        ci = a["d_CI"]
        gf = a.get("gap_fraction")
        gf_s = f"{gf:+.2f}" if isinstance(gf, (int, float)) else "n/a"
        print(f"{a['name']:6s}{a['d']:>10.3f}"
              f"{f'[{ci[0]:+.2f},{ci[1]:+.2f}]':>22s}{gf_s:>10s}{a['n_effect']:>7d}")
    g = prof.get("ld_gaps")
    if g:
        print(f"LD gaps (success->failure): new_old={g['gap_new_old']}, "
              f"param_new={g['gap_param_new']}")
    print("\nRanked pairwise contrasts (dominance, NOT a share):")
    for c in prof["pairwise_contrasts"]:
        sig = "SIG" if c["significant"] else "n.s."
        print(f"  {c['contrast']:14s} = {c['estimate']:+.3f}  "
              f"CI=[{c['CI'][0]:+.2f},{c['CI'][1]:+.2f}]  [{sig}]")
    print(f"\nRanking by |d|: "
          f"{' > '.join(r['axis'] for r in prof['ranking_by_abs_d'])}")
    if not prof.get("ld_gaps"):
        print("gap-normalized recovered-LD fraction: pending "
              "(run F3 with the LD-gap probe to emit f3_ld_gap.json)")

    out = Path(args.out) if args.out else (Path(args.f3_dir) / "causal_profile.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(prof, fh, indent=1)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
