#!/usr/bin/env python3
"""Research-ready figures for F2 Diagnostic results.

Generates a 4-panel figure (PDF + PNG) suitable for a paper:
  Panel A  —  B1 failure taxonomy (stacked bar)
  Panel B  —  F2-c: B1 / B5 / B6 accuracy + overlap breakdown
  Panel C  —  F2-a: STR patching recovery fraction distribution
  Panel D  —  F2-b: Logit Lens trajectories (P(answer_new) per layer)

Usage
-----
    python scripts/plot_f2_results.py \\
        --results results/f2_diagnostic \\
        --out     results/f2_diagnostic/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Colour palette (Wong, 2011 – colour-blind safe) ──────────────────────────
C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_PURPLE = "#CC79A7"
C_SKY    = "#56B4E9"
C_YELLOW = "#F0E442"
C_GRAY   = "#999999"

# ── Global style ──────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _panel_label(ax, letter: str, x: float = -0.12, y: float = 1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")


# ─────────────────────────────────────────────────────────────────────────────
# Panel A  –  B1 failure taxonomy
# ─────────────────────────────────────────────────────────────────────────────

def panel_a(ax: plt.Axes, data: dict) -> None:
    n_total   = data["n_total"]
    n_success = data["n_b1_success"]
    n_old     = data["n_reverts_old"]
    n_other   = data["n_reverts_other"]

    cats   = ["B1-Success", "REVERTS_OLD", "REVERTS_OTHER"]
    counts = [n_success, n_old, n_other]
    colors = [C_GREEN, C_RED, C_ORANGE]
    pcts   = [100 * c / n_total for c in counts]

    bars = ax.barh(cats, pcts, color=colors, edgecolor="white", linewidth=0.8, height=0.55)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(
            pct + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%  (n={cnt})",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Fraction of B1 instances (%)")
    ax.set_title("B1 Failure Taxonomy\n(Phi-3-mini, n=200)", pad=8)
    ax.axvline(pcts[0], color="black", linewidth=0.6, linestyle=":")
    ax.grid(axis="x")
    ax.grid(axis="y", alpha=0)
    _panel_label(ax, "A")


# ─────────────────────────────────────────────────────────────────────────────
# Panel B  –  F2-c Behavioral cross-analysis
# ─────────────────────────────────────────────────────────────────────────────

def panel_b(ax: plt.Axes, data: dict) -> None:
    b1_acc = data["b1_accuracy"] * 100
    b5_acc = data["b5_accuracy"] * 100
    b6_acc = data["b6_accuracy"] * 100

    n_total   = data["n_total"]
    n_b1_fail = n_total - round(n_total * data["b1_accuracy"])
    n_b1f_b5s = data["n_b1_fail_b5_success"]
    n_b1f_b5f = data["n_b1_fail_b5_fail"]
    rate_b1f_b5f = data["n_b1_fail_b5_fail_rate"] * 100

    # ── Accuracy bars ────────────────────────────────────────────────────────
    x    = np.arange(3)
    accs = [b1_acc, b5_acc, b6_acc]
    cols = [C_BLUE, C_PURPLE, C_GRAY]
    labs = ["B1\n(single evid.)", "B5\n(dual + year)", "B6\n(dual – year)"]

    bars = ax.bar(x, accs, color=cols, width=0.55, edgecolor="white", zorder=3)
    for xi, (acc, col) in enumerate(zip(accs, cols)):
        ax.text(xi, acc + 1.2, f"{acc:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("F2-c: B5 vs B6 Behavioral Cross-Analysis\n(Phi-3-mini, n=70)", pad=8)
    ax.grid(axis="y", alpha=0.35, linestyle="--")

    # ── B1-failure overlap annotation box ───────────────────────────────────
    ax.annotate(
        f"B1-fail & B5-fail:  {n_b1f_b5f}/{n_b1_fail}  ({rate_b1f_b5f:.0f}%)\n"
        f"B1-fail & B5-success: {n_b1f_b5s}/{n_b1_fail}",
        xy=(1, b5_acc), xytext=(1.6, 52),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"),
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF9C4", ec="#CCBB44", alpha=0.9),
    )
    ax.text(
        0.5, -0.18,
        f"{rate_b1f_b5f:.0f}% of B1 failures also fail B5  =>  strong F2 signal",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, style="italic", color="#444444",
    )
    _panel_label(ax, "B")


# ─────────────────────────────────────────────────────────────────────────────
# Panel C  –  F2-a STR patching recovery distribution
# ─────────────────────────────────────────────────────────────────────────────

def panel_c(ax: plt.Axes, data: dict) -> None:
    recs = [
        r["recovery_fraction"]
        for r in data["per_instance"]
        if "recovery_fraction" in r
    ]
    recs = np.array(recs)

    n_causal  = int((recs >= 0.5).sum())
    n_valid   = len(recs)
    pct_causal = 100 * n_causal / n_valid if n_valid else 0

    # Histogram
    bins = np.linspace(-0.5, 1.2, 30)
    n_vals, _, patches = ax.hist(
        recs, bins=bins, color=C_SKY, edgecolor="white",
        linewidth=0.6, zorder=3,
    )
    # Colour bars above 0.5 threshold differently
    for patch, left in zip(patches, bins[:-1]):
        if left >= 0.5:
            patch.set_facecolor(C_PURPLE)

    ax.axvline(0.0, color="black",  linewidth=1.2, linestyle="--", label="no recovery")
    ax.axvline(0.5, color=C_PURPLE, linewidth=1.5, linestyle="--",
               label=f"≥50% recovery  ({pct_causal:.1f}%,  n={n_causal})")

    ax.set_xlabel("Recovery fraction  (patched − corrupted) / (clean − corrupted)")
    ax.set_ylabel("# instances")
    ax.set_title(f"F2-a: STR Activation Patching Recovery\n"
                 f"(Phi-3-mini, n={n_valid} valid instances)", pad=8)
    ax.legend(frameon=False, fontsize=8.5)

    # Summary stats annotation
    ax.text(
        0.97, 0.95,
        f"mean = {recs.mean():+.3f}\nmedian = {np.median(recs):+.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8),
    )
    _panel_label(ax, "C")


# ─────────────────────────────────────────────────────────────────────────────
# Panel D  –  F2-b Logit Lens trajectories
# ─────────────────────────────────────────────────────────────────────────────

def panel_d(ax: plt.Axes, data: dict) -> None:
    instances = data["per_instance"]
    n_layers  = len(instances[0]["probs_new"]) if instances else 32
    layers    = np.arange(n_layers)
    l_t       = data.get("l_t", 10)

    # Compute peak for each instance and choose the top-2 most "F2-like"
    # (largest peak-to-final drop) to highlight; others in light gray
    peak_drops = []
    for inst in instances:
        p_new = np.array(inst["probs_new"])
        peak  = p_new[l_t:].max()
        final = p_new[-1]
        peak_drops.append(peak - final)

    ranked = np.argsort(peak_drops)[::-1]   # descending

    ax2 = ax.twinx()
    ax2.set_ylabel("P(answer_old)", color=C_RED, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C_RED, labelsize=8)
    ax2.spines["top"].set_visible(False)

    plotted_new = []
    plotted_old = []

    for rank, idx in enumerate(ranked):
        inst  = instances[idx]
        p_new = np.array(inst["probs_new"])
        p_old = np.array(inst["probs_old"])
        iid   = inst["instance_id"]

        is_highlight = rank < 2
        alpha_new = 0.85 if is_highlight else 0.22
        alpha_old = 0.55 if is_highlight else 0.12
        lw_new    = 2.2  if is_highlight else 0.8
        lw_old    = 1.4  if is_highlight else 0.6

        ln1, = ax.plot(layers, p_new, color=C_BLUE,
                       alpha=alpha_new, linewidth=lw_new, zorder=3 if is_highlight else 1)
        ln2, = ax2.plot(layers, p_old, color=C_RED,
                        alpha=alpha_old, linewidth=lw_old, linestyle="--",
                        zorder=2 if is_highlight else 0)
        if is_highlight:
            plotted_new.append(ln1)
            plotted_old.append(ln2)

            # annotate peak
            pk_layer = np.argmax(p_new[l_t:]) + l_t
            pk_val   = p_new[pk_layer]
            ax.annotate(
                f"peak L{pk_layer}\n({pk_val:.2f})",
                xy=(pk_layer, pk_val),
                xytext=(pk_layer - 4, pk_val + 0.06 * (1 + rank * 0.5)),
                fontsize=7, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_BLUE),
            )

    ax.axvline(l_t, color="#888888", linewidth=1.2, linestyle=":",
               label=f"$L_T$ = {l_t}  (temporal heads)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("P(answer_new)", color=C_BLUE)
    ax.tick_params(axis="y", labelcolor=C_BLUE, labelsize=8)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"F2-b: Logit Lens Trajectories (REVERTS_OLD, n={len(instances)})\n"
        "Solid = P(answer_new), dashed = P(answer_old)",
        pad=8,
    )

    # legend
    handles = [
        mpatches.Patch(color=C_BLUE,  label="P(answer_new) — top-2 instances (highlighted)"),
        mpatches.Patch(color=C_RED,   label="P(answer_old)"),
        mpatches.Patch(color=C_BLUE,  alpha=0.25, label="P(answer_new) — other instances"),
        plt.Line2D([0], [0], color="#888888", linestyle=":", label=f"$L_T$={l_t}"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.5)
    _panel_label(ax, "D")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus Panel  –  F2-b peak RouteScore per instance (small, below D)
# ─────────────────────────────────────────────────────────────────────────────

def panel_d_inset(ax: plt.Axes, data: dict) -> None:
    """Small bar chart showing peak-to-final drop per REVERTS_OLD instance."""
    instances = data["per_instance"]
    l_t = data.get("l_t", 10)

    iids, peak_drops, lt_drops = [], [], []
    for inst in instances:
        p_new = np.array(inst["probs_new"])
        peak_drops.append(float(p_new[l_t:].max() - p_new[-1]))
        lt_drops.append(float(p_new[l_t] - p_new[-1]))
        short_id = inst["instance_id"].replace("B1_", "")[:8]
        iids.append(short_id)

    x = np.arange(len(iids))
    w = 0.38
    ax.bar(x - w/2, peak_drops, width=w, color=C_BLUE,   label="RouteScore (peak)", alpha=0.85)
    ax.bar(x + w/2, lt_drops,   width=w, color=C_SKY,    label=f"RouteScore (L_T={l_t})", alpha=0.85)
    ax.axhline(0.05, color=C_RED, linewidth=1.2, linestyle="--", label="threshold 0.05")
    ax.axhline(0.0,  color="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(iids, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("RouteScore")
    ax.set_title("F2-b: Peak RouteScore per REVERTS_OLD Instance", pad=6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    _panel_label(ax, "E")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/f2_diagnostic",
                        help="Directory containing F2 JSON result files")
    parser.add_argument("--out",     default="results/f2_diagnostic/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    res_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    filter_data = _load(res_dir / "reverts_old_filter.json")
    f2a_data    = _load(res_dir / "f2a_str_patching.json")
    f2b_data    = _load(res_dir / "f2b_route_score.json")
    f2c_data    = _load(res_dir / "f2c_b5_vs_b6.json")

    # ── Figure 1: 4-panel summary ────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.55,
        wspace=0.38,
        left=0.07, right=0.97,
        top=0.91,  bottom=0.09,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, filter_data)
    panel_b(ax_b, f2c_data)
    panel_c(ax_c, f2a_data)
    panel_d(ax_d, f2b_data)

    fig.suptitle(
        'F2 Diagnostic: "Time Set but Not Routed"  —  Phi-3-mini-4k-instruct',
        fontsize=13, fontweight="bold", y=0.97,
    )

    for fmt in ("pdf", "png"):
        path = out_dir / f"f2_diagnostic_summary.{fmt}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    # ── Figure 2: Logit Lens detail (full-width, taller) ─────────────────────
    fig2, axes2 = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig2.subplots_adjust(wspace=0.4, left=0.07, right=0.97, top=0.88, bottom=0.12)
    panel_d(axes2[0], f2b_data)
    panel_d_inset(axes2[1], f2b_data)
    fig2.suptitle(
        "F2-b: Logit Lens  ·  Phi-3-mini-4k-instruct  ·  REVERTS_OLD instances",
        fontsize=12, fontweight="bold",
    )
    for fmt in ("pdf", "png"):
        path = out_dir / f"f2b_logit_lens_detail.{fmt}"
        fig2.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig2)

    # ── Figure 3: F2-a recovery scatter (logit_diff clean vs patched) ─────────
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    fig3.subplots_adjust(left=0.13, right=0.95, top=0.90, bottom=0.12)

    instances = f2a_data["per_instance"]
    ld_clean   = np.array([r["logit_diff_clean"]    for r in instances])
    ld_patch   = np.array([r["logit_diff_patched"]  for r in instances])
    ld_corrupt = np.array([r["logit_diff_corrupted"] for r in instances])
    rec        = np.array([r["recovery_fraction"]   for r in instances])

    sc = ax3.scatter(
        ld_corrupt, ld_patch,
        c=rec, cmap="RdYlGn",
        vmin=-0.5, vmax=1.0,
        s=40, alpha=0.80, edgecolors="white", linewidth=0.4,
    )
    # diagonal = no change from corrupted
    lim = max(abs(ld_corrupt).max(), abs(ld_patch).max()) * 1.05
    ax3.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.9, alpha=0.5, label="no change")
    ax3.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax3.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    cbar = fig3.colorbar(sc, ax=ax3, shrink=0.85)
    cbar.set_label("Recovery fraction", fontsize=9)
    ax3.set_xlabel("Logit diff (corrupted run:  clean year->t_old)")
    ax3.set_ylabel("Logit diff (patched run:  temporal heads restored)")
    ax3.set_title(
        "F2-a: STR Patching — Corrupted vs Patched Logit Diffs\n"
        f"(Phi-3-mini, n={len(instances)})", pad=8,
    )
    ax3.legend(frameon=False, fontsize=8.5)
    _panel_label(ax3, "F2-a", x=-0.14)

    for fmt in ("pdf", "png"):
        path = out_dir / f"f2a_patching_scatter.{fmt}"
        fig3.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig3)

    print(f"\nAll figures written to: {out_dir}/")


if __name__ == "__main__":
    main()
