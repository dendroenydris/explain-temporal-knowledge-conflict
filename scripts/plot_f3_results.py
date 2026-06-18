#!/usr/bin/env python3
"""Research-ready figures for F3 Diagnostic results (spine = head-ablation verdict).

The default figure is the crystallization-robust SPINE (Changes 5/8):

  Panel A  —  F3-a data-driven suppression rate per param_class
              (TEMPORAL_STALE_CONFIRMED / PARAM_AMBIGUOUS / PARAM_NEW) with the
              rank-competitive overlay. The legacy fixed mid-window trajectory
              rate is ≈0 under late crystallization and is shown only as a
              footnote (the peak lands outside [L/4, 2L/3]).
  Panel B  —  F3-a Logit-Lens trajectories ``P^l(answer_new)`` — drawn ONLY when
              ``not lens_na`` (Finding 12.4); a placeholder otherwise.
  Panel C  —  Combined DLA localization + causal head-ablation Δ (full width):
              per-head DLA magnitudes (top-k) and the population Δ
              (failure − B1-success) with CI.

The cross-cutting failure-mode taxonomy (F1/F2/F3) is deliberately NOT shown
here — it is an F3-internal heuristic that conflicts with the authoritative
per-diagnostic verdicts. The single authoritative taxonomy is rendered by
``scripts/plot_integrated_results.py`` after F1/F2/F3 have all run.

The legacy F3-0.5/b/c lattice figure is rendered separately only when the
corresponding ``--hardening`` artifacts exist.

Usage
-----
    python scripts/plot_f3_results.py \\
        --results results/f3_diagnostic \\
        --out     results/f3_diagnostic/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Wong (2011) colour-blind-safe palette ─────────────────────────────
C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_PURPLE = "#CC79A7"
C_SKY    = "#56B4E9"
C_YELLOW = "#F0E442"
C_GRAY   = "#999999"

CLASS_COLOR = {
    "TEMPORAL_STALE_CONFIRMED": C_RED,
    "PARAM_AMBIGUOUS":          C_ORANGE,
    "PARAM_NEW":                C_BLUE,
}
CLASS_SHORT = {
    "TEMPORAL_STALE_CONFIRMED": "STALE_CONF",
    "PARAM_AMBIGUOUS":          "AMBIG",
    "PARAM_NEW":                "NEW",
}
PARAM_CLASSES = ["TEMPORAL_STALE_CONFIRMED", "PARAM_AMBIGUOUS", "PARAM_NEW"]
LENS_NA_THRESHOLD = 0.10

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})


# ─────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _panel_label(ax, letter: str, x: float = -0.10, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")


def _lens_na(f3a: dict | None) -> bool:
    if not f3a:
        return True
    frac = (f3a.get("summary", {}) or {}).get("lens_decodable_fraction", 0.0)
    return float(frac) < LENS_NA_THRESHOLD


# ─────────────────────────────────────────────────────────────────────
# Panel A — F3-a trajectory rates (descriptive) + spine annotations
# ─────────────────────────────────────────────────────────────────────


def _rate_by_class(f3a: dict, field: str, predicate) -> dict[str, float]:
    """Fallback: compute a per-class rate from per_instance when the summary
    lacks the field (older result JSONs)."""
    per = f3a.get("per_instance", []) or []
    out: dict[str, float] = {}
    for c in PARAM_CLASSES:
        rows = [r for r in per if r.get("param_class") == c]
        out[c] = (float(np.mean([int(bool(predicate(r))) for r in rows]))
                  if rows else 0.0)
    return out


def panel_a(ax: plt.Axes, f3a: dict | None, modes: dict | None) -> None:
    ax.set_title("F3-a suppression rate by param_class")
    if not f3a:
        ax.text(0.5, 0.5, "(F3-a not run)", ha="center", va="center",
                transform=ax.transAxes, color=C_GRAY)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "A")
        return

    summary = f3a.get("summary", {}) or {}

    # Primary signal = data-driven suppression rate (peak-over-all-layers → final
    # drop), which is late-crystallization-robust. The legacy mid-window
    # f3_traj_rate is ≈0 on late-crystallizing models (peak falls outside [L/4,2L/3])
    # and is shown only as a footnote.
    supp = summary.get("suppression_rate")
    if not supp:
        supp = _rate_by_class(f3a, "is_suppression", lambda r: r.get("is_suppression"))
    rankc = summary.get("rank_competitive_rate")
    if not rankc:
        rankc = _rate_by_class(
            f3a, "first_rank_competitive_layer",
            lambda r: r.get("first_rank_competitive_layer", -1) >= 0)

    rates = [supp.get(c, 0.0) for c in PARAM_CLASSES]
    rc_rates = [rankc.get(c, 0.0) for c in PARAM_CLASSES]
    counts = [summary.get("counts_by_class", {}).get(c, 0) for c in PARAM_CLASSES]
    labels = [CLASS_SHORT[c] for c in PARAM_CLASSES]
    x = np.arange(len(PARAM_CLASSES))

    bars = ax.bar(x, rates, color=[CLASS_COLOR[c] for c in PARAM_CLASSES],
                  edgecolor="black", linewidth=0.6, label="suppression rate")
    # Overlay rank-competitive ("was routed/decodable") as open markers.
    ax.plot(x, rc_rates, "D", color="black", markersize=6,
            markerfacecolor="white", label="rank-competitive rate")
    for xi, bar, count, r in zip(x, bars, counts, rates):
        ax.text(xi, bar.get_height() + 0.02, f"{r:.0%}\nn={count}",
                ha="center", va="bottom", fontsize=8, color="black")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate over class (descriptive)")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7.5)

    legacy = summary.get("f3_traj_rate", {}) or {}
    legacy_max = max(legacy.values()) if legacy else 0.0
    lens_dec = summary.get("lens_decodable_fraction", 0.0)
    conf_stale = summary.get("confirmed_stale_fraction", 0.0)
    n_clean = summary.get("n_clean_f3", 0)
    raw = (modes or {}).get("counts_f3_raw", 0)
    note = (f"lens_decodable={lens_dec:.2f}  •  F3_clean={n_clean} (raw={raw})  •  "
            f"confirmed_stale\u2265{conf_stale:.2f}\n"
            f"legacy mid-window traj_rate(max)={legacy_max:.2f} "
            f"(\u22480 under late crystallization \u2014 not the F3 signal)")
    if _lens_na(f3a):
        note += "  •  LENS_NA"
    ax.set_xlabel(note, fontsize=7.5)
    _panel_label(ax, "A")


# ─────────────────────────────────────────────────────────────────────
# Panel B — F3-a Logit-Lens trajectories (only when not lens_na)
# ─────────────────────────────────────────────────────────────────────


def panel_b(ax: plt.Axes, f3a: dict | None) -> None:
    ax.set_title("F3-a Logit-Lens trajectories  P^l(answer_new)")
    if not f3a:
        ax.text(0.5, 0.5, "(F3-a not run)", ha="center", va="center",
                transform=ax.transAxes, color=C_GRAY)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "B")
        return
    if _lens_na(f3a):
        frac = (f3a.get("summary", {}) or {}).get("lens_decodable_fraction", 0.0)
        ax.text(0.5, 0.5,
                f"lens_na (decodable={frac:.2f} < {LENS_NA_THRESHOLD})\n"
                "trajectory readout suppressed (Finding 12.4)",
                ha="center", va="center", transform=ax.transAxes, color=C_GRAY,
                fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "B")
        return

    per = f3a.get("per_instance", [])
    summary = f3a.get("summary", {}) or {}
    n_layers = summary.get("n_layers", 32)
    mid = summary.get("mid_window", [None, None])
    if mid[0] is not None and mid[1] is not None:
        ax.axvspan(mid[0], mid[1], color=C_YELLOW, alpha=0.18, zorder=0)

    n_drawn = {c: 0 for c in PARAM_CLASSES}
    cap = 80
    median_frc = []
    for row in per:
        cls = row.get("param_class", "?")
        if cls not in CLASS_COLOR:
            continue
        if row.get("first_rank_competitive_layer", -1) >= 0:
            median_frc.append(row["first_rank_competitive_layer"])
        if n_drawn.get(cls, cap) >= cap:
            continue
        n_drawn[cls] += 1
        probs = row.get("p_new", [])
        # Highlight the data-driven suppression trajectories (late-crystallization
        # robust); the legacy is_f3_trajectory flag is ≈always False here.
        highlight = row.get("is_suppression") or row.get("is_f3_trajectory")
        alpha = 0.7 if highlight else 0.18
        lw = 1.0 if highlight else 0.5
        ax.plot(range(len(probs)), probs, color=CLASS_COLOR[cls], alpha=alpha, lw=lw)

    if median_frc:
        med = float(np.median(median_frc))
        ax.axvline(med, color=C_GREEN, ls=":", lw=1.5,
                   label=f"median first-rank-competitive={med:.0f}")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("P^l(answer_new)")
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(0, 1.0)
    legend_handles = [mpatches.Patch(color=CLASS_COLOR[c], label=CLASS_SHORT[c])
                      for c in PARAM_CLASSES]
    ax.add_artist(ax.legend(handles=legend_handles, loc="upper left",
                            framealpha=0.9, fontsize=7))
    _panel_label(ax, "B")


# NOTE: The cross-cutting failure-mode taxonomy (F1/F2/F3/MIXED) is intentionally
# NOT plotted inside the F3 figure. That classifier is an F3-internal heuristic and
# conflicts with the authoritative per-diagnostic verdicts (real F1-a, F2-DLA).
# The single authoritative taxonomy lives in scripts/plot_integrated_results.py,
# which joins the real F1/F2/F3 verdicts per instance after all stages have run.


# ─────────────────────────────────────────────────────────────────────
# Panel C — Combined DLA localization + causal head-ablation Δ
# ─────────────────────────────────────────────────────────────────────


def panel_dla_delta(ax: plt.Axes, ablation: dict | None) -> None:
    ax.set_title("DLA-localized heads + causal ablation Δ")
    if not ablation:
        ax.text(0.5, 0.5, "(no f3_head_ablation.json)", ha="center", va="center",
                transform=ax.transAxes, color=C_GRAY)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "D")
        return

    per_head = ablation.get("per_head_dla", {}) or {}
    items = sorted(per_head.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    xs = np.arange(len(labels))
    if labels:
        ax.bar(xs, vals, color=C_PURPLE, edgecolor="black", linewidth=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("cohort-mean DLA  (a_param − answer_new)")
    ax.axhline(0, color=C_GRAY, lw=0.8)

    delta = ablation.get("delta")
    ci = ablation.get("delta_CI") or [None, None]
    eff_f = ablation.get("effect_failure_mean")
    eff_s = ablation.get("effect_success_mean")
    n_f = ablation.get("n_clean_f3", 0)
    n_s = ablation.get("n_success_null", 0)
    txt = "ablation Δ (failure − B1-success):\n"
    if delta is not None:
        txt += f"Δ={delta:+.4f}"
        if ci[0] is not None:
            txt += f"  CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        txt += "\n"
    if eff_f is not None and eff_s is not None:
        txt += f"eff_fail={eff_f:+.3f} (n={n_f})  eff_succ={eff_s:+.3f} (n={n_s})"
    supported = ci[0] is not None and ci[0] > 0
    box_color = C_GREEN if supported else C_GRAY
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec=box_color, lw=1.4))
    _panel_label(ax, "C", x=-0.05)


# ─────────────────────────────────────────────────────────────────────
# Title strip — final F3 verdict
# ─────────────────────────────────────────────────────────────────────


def title_strip(fig, verdict: dict | None) -> None:
    if verdict is None:
        return
    title = verdict.get("title", "F3")
    code = verdict.get("verdict", "")
    fig.suptitle(f"F3 Diagnosis — {title}  [{code}]", fontsize=14, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────
# Legacy hardening panels (rendered only when --hardening artifacts exist)
# ─────────────────────────────────────────────────────────────────────


def _render_hardening(results: Path, out_dir: Path) -> None:
    f3half = _load(results / "f3_half_attribution.json")
    f3b = _load(results / "f3b_ablation.json")
    interv = _load(results / "f3_intervention.json")
    if f3half is None and f3b is None and interv is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1, ax2 = axes
    ax1.set_title("F3-0.5 routing set / F3-b (hardening, appendix)")
    if f3b:
        labels = ["δR/δrand-SL", "δR/δrand-DM"]
        ratios = [f3b.get("ratio_R_over_rand_same", 0.0),
                  f3b.get("ratio_R_over_rand_depth", 0.0)]
        ax1.bar(labels, ratios, color=C_BLUE, edgecolor="black", linewidth=0.6)
        ax1.axhline(1.0, color=C_GRAY, ls="--", lw=1.0)
        ax1.set_ylabel("specificity ratio")
    else:
        ax1.text(0.5, 0.5, "(F3-b not run)", ha="center", va="center",
                 transform=ax1.transAxes, color=C_GRAY)
    ax2.set_title("Appendix span intervention (not lens_na only)")
    summ = (interv or {}).get("summary") or []
    if summ:
        variants = [s["variant"] for s in summ]
        rec = [s["recovery_rate"] for s in summ]
        rnd = [s["random_recovery_rate"] for s in summ]
        x = np.arange(len(variants))
        w = 0.38
        ax2.bar(x - w / 2, rec, w, label="intervention", color=C_GREEN)
        ax2.bar(x + w / 2, rnd, w, label="random-layer", color=C_GRAY)
        ax2.set_xticks(x); ax2.set_xticklabels(variants, rotation=15, ha="right")
        ax2.set_ylabel("answer_new recovery"); ax2.set_ylim(0, 1); ax2.legend()
    else:
        ax2.text(0.5, 0.5, "(span KO skipped: lens_na or not run)", ha="center",
                 va="center", transform=ax2.transAxes, color=C_GRAY)
    fig.suptitle("F3 hardening (appendix lattice)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"f3_hardening.{ext}")
    plt.close(fig)
    print(f"Saved: {out_dir / 'f3_hardening.png'}")


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


def render(results: Path, out_dir: Path, figsize: tuple[float, float]) -> None:
    """Render the spine figure (+ hardening figure when artifacts exist)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    f3a = _load(results / "f3a_trajectory.json")
    modes = _load(results / "f3a_failure_modes.json")
    ablation = _load(results / "f3_head_ablation.json")
    verdict = _load(results / "f3_verdict.json")

    fig = plt.figure(figsize=tuple(figsize), constrained_layout=False)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
                           left=0.08, right=0.97, top=0.90, bottom=0.10)
    panel_a(fig.add_subplot(gs[0, 0]), f3a, modes)
    panel_b(fig.add_subplot(gs[0, 1]), f3a)
    panel_dla_delta(fig.add_subplot(gs[1, :]), ablation)
    title_strip(fig, verdict)

    pdf_path = out_dir / "f3_results.pdf"
    png_path = out_dir / "f3_results.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    _render_hardening(results, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--f3-dir", "--results", dest="results", default=None,
                        help="Directory containing f3*.json outputs")
    parser.add_argument("--f3-dir-m", dest="results_m", default=None,
                        help="Optional second results directory; figures go to <out>/M")
    parser.add_argument("--out", default=None,
                        help="Output directory for figures (defaults to <results>/figures)")
    parser.add_argument("--figsize", nargs=2, type=float, default=[13, 10])
    parser.add_argument("--model-tag", dest="model_tag", default=None,
                        help="(accepted for compatibility; unused)")
    parser.add_argument("--tau", type=float, default=None,
                        help="(accepted for compatibility; unused)")
    args = parser.parse_args()

    if not args.results:
        parser.error("one of --f3-dir / --results is required")

    results = Path(args.results)
    out_dir = Path(args.out) if args.out else (results / "figures")
    render(results, out_dir, tuple(args.figsize))

    if args.results_m:
        results_m = Path(args.results_m)
        render(results_m, out_dir / "M", tuple(args.figsize))


if __name__ == "__main__":
    main()
