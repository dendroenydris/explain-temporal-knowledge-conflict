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

def _f2c_get_primary(data: dict) -> dict:
    """Return the B5×B6 primary cells dict, regardless of JSON schema age."""
    primary = data.get("b5xb6_primary")
    if primary is not None:
        return primary
    # Back-compat: derive from `details` if pre-update JSON
    details = data.get("details", [])
    n_b5s_b6f = sum(1 for d in details if d.get("b5_success") and not d.get("b6_success"))
    n_b5f_b6f = sum(1 for d in details if not d.get("b5_success") and not d.get("b6_success"))
    n_b5f_b6s = sum(1 for d in details if not d.get("b5_success") and d.get("b6_success"))
    n_b5s_b6s = sum(1 for d in details if d.get("b5_success") and d.get("b6_success"))
    n = max(len(details), 1)
    return {
        "n_b5_success_b6_fail":    n_b5s_b6f,
        "n_b5_fail_b6_fail":       n_b5f_b6f,
        "n_b5_fail_b6_success":    n_b5f_b6s,
        "n_b5_success_b6_success": n_b5s_b6s,
        "rate_b5_fail_b6_fail":    n_b5f_b6f / n,
        "rate_b5_success_b6_fail": n_b5s_b6f / n,
    }


def panel_b(ax: plt.Axes, data: dict) -> None:
    """Panel B: F2-c B5 vs B6 — primary F2 behavioral detector.

    Methodology lines 334–349:
      * **Primary** F2 detector is the per-instance B5 × B6 contingency.
      * B1 × B5 is supplementary (separate annotation below the bars).
    """
    b1_acc = data["b1_accuracy"] * 100
    b5_acc = data["b5_accuracy"] * 100
    b6_acc = data["b6_accuracy"] * 100
    n_total = data["n_total"]

    primary = _f2c_get_primary(data)
    n_b5s_b6f = primary["n_b5_success_b6_fail"]
    n_b5f_b6f = primary["n_b5_fail_b6_fail"]
    n_b5f_b6s = primary["n_b5_fail_b6_success"]
    n_b5s_b6s = primary["n_b5_success_b6_success"]

    # ── Accuracy bars ────────────────────────────────────────────────────────
    x    = np.arange(3)
    accs = [b1_acc, b5_acc, b6_acc]
    cols = [C_BLUE, C_PURPLE, C_GRAY]
    labs = ["B1\n(single evid.)", "B5\n(dual + year)", "B6\n(dual − year)"]

    ax.bar(x, accs, color=cols, width=0.55, edgecolor="white", zorder=3)
    for xi, acc in enumerate(accs):
        ax.text(xi, acc + 1.2, f"{acc:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222222")

    # Highlight ΔB5−B6 (year contribution to routing)
    delta = b5_acc - b6_acc
    ax.annotate(
        "",
        xy=(2, b5_acc), xytext=(2, b6_acc),
        arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.4, mutation_scale=12),
    )
    ax.text(
        2.05, (b5_acc + b6_acc) / 2,
        f"ΔB5−B6 = {delta:+.1f}pp",
        fontsize=8.5, color=C_RED, va="center", fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        f"F2-c: B5 vs B6 Behavioral Cross-Analysis  (n={n_total})\n"
        "Primary F2 detector — per-instance B5×B6 cells below",
        pad=8,
    )
    ax.grid(axis="y", alpha=0.35, linestyle="--")

    # ── B5 × B6 cells (primary F2 detector) ────────────────────────────────
    cell_lines = [
        ("B5-success ∩ B6-fail",   n_b5s_b6f, "anti-F2"),
        ("B5-fail    ∩ B6-fail",   n_b5f_b6f, "candidate F2"),
        ("B5-fail    ∩ B6-success", n_b5f_b6s, "paradoxical"),
        ("B5-success ∩ B6-success", n_b5s_b6s, "year not needed"),
    ]
    box_text = "Per-instance B5×B6 cells (methodology lines 340–344):\n"
    for label, n, tag in cell_lines:
        pct = 100 * n / n_total if n_total else 0
        box_text += f"  {label:<26s}  {n:>3d}/{n_total}  ({pct:4.1f}%)  — {tag}\n"

    # 3-way disambiguation cells (methodology line 347)
    disamb = data.get("b1xb5xb6_disambiguation") or {}
    if disamb:
        box_text += "\n3-way B1×B5×B6 disambiguation (methodology line 347):\n"
        rows = [
            ("B1f ∩ B5f ∩ B6f", disamb.get("n_b1f_b5f_b6f", 0), "PARAMETRIC DOMINANCE — not F2"),
            ("B1f ∩ B5f ∩ B6s", disamb.get("n_b1f_b5f_b6s", 0), "year NOT required"),
            ("B1f ∩ B5s ∩ B6f", disamb.get("n_b1f_b5s_b6f", 0), "YEAR-DRIVEN RESCUE — rules out F2"),
            ("B1f ∩ B5s ∩ B6s", disamb.get("n_b1f_b5s_b6s", 0), "dual-ev. rescue regardless of year"),
        ]
        for label, n, tag in rows:
            pct = 100 * n / n_total if n_total else 0
            box_text += f"  {label:<26s}  {n:>3d}/{n_total}  ({pct:4.1f}%)  — {tag}\n"

    box_text = box_text.rstrip()
    ax.text(
        -0.18, -0.45, box_text,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, family="monospace", color="#222222",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F4F4F8", ec="#CCCCCC", alpha=0.95),
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

# Colour map for F2 regimes (methodology lines 288–292)
_REGIME_COLORS = {
    "not_f2":    C_GREEN,
    "f2_strong": C_RED,
    "f2_weak":   C_ORANGE,
    "f3":        C_PURPLE,
    None:        C_GRAY,
}


def panel_d(ax: plt.Axes, data: dict) -> None:
    """Panel D: F2-b Logit Lens trajectories, coloured by F2 regime.

    Methodology lines 288–292: each trajectory is per-instance classified
    into ``not_f2`` / ``f2_strong`` / ``f2_weak`` / ``f3`` from the
    RouteScore peak + final-P^L pattern.  We use that label (when
    present) to colour the trajectory so the regime composition of the
    population is visible at a glance.
    """
    instances = data["per_instance"]
    n_layers  = len(instances[0]["probs_new"]) if instances else 32
    layers    = np.arange(n_layers)
    l_t       = data.get("l_t", 10)
    l_t_mode  = data.get("l_t_mode", "?")

    ax2 = ax.twinx()
    ax2.set_ylabel("P(answer_old)", color=C_RED, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C_RED, labelsize=8)
    ax2.spines["top"].set_visible(False)

    regime_seen: set = set()
    for inst in instances:
        p_new = np.array(inst["probs_new"])
        p_old = np.array(inst["probs_old"])
        regime = inst.get("f2_regime")
        regime_seen.add(regime)
        color = _REGIME_COLORS.get(regime, C_GRAY)

        is_highlight = regime in ("f2_weak", "f3")
        alpha_new = 0.75 if is_highlight else 0.35
        lw_new    = 1.6  if is_highlight else 0.7
        ax.plot(layers, p_new, color=color, alpha=alpha_new, linewidth=lw_new,
                zorder=3 if is_highlight else 1)
        ax2.plot(layers, p_old, color=C_RED, alpha=0.15, linewidth=0.5,
                 linestyle="--", zorder=0)

    ax.axvline(l_t, color="#444444", linewidth=1.2, linestyle=":",
               label=f"$L_T$ = {l_t} ({l_t_mode})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("P(answer_new)", color="#222222")
    ax.tick_params(axis="y", labelcolor="#222222", labelsize=8)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"F2-b: Logit Lens Trajectories  (n={len(instances)}, coloured by F2 regime)\n"
        f"Solid = P(answer_new), dashed = P(answer_old).  L_T={l_t} ({l_t_mode}).",
        pad=8,
    )

    # Legend: only show regimes that actually appear in the data
    legend_entries = [
        ("not_f2",    "signal survived (not F2)"),
        ("f2_strong", "F2-strong (no mid-rise)"),
        ("f2_weak",   "F2-weak (moderate mid-rise)"),
        ("f3",        "F3 candidate (strong mid-rise)"),
    ]
    handles = [
        mpatches.Patch(color=_REGIME_COLORS[k], label=label)
        for k, label in legend_entries if k in regime_seen
    ]
    handles.append(plt.Line2D([0], [0], color="#444444", linestyle=":",
                              label=f"$L_T$={l_t}"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.5)
    _panel_label(ax, "D")


# ─────────────────────────────────────────────────────────────────────────────
# Panel E  –  F2 regime distribution (population histogram)
# ─────────────────────────────────────────────────────────────────────────────

def panel_regime(ax: plt.Axes, f2b_data: dict, verdict_data: dict | None) -> None:
    """Stacked bar of per-instance F2 regimes (and final F2 verdicts if available).

    Top bar: F2-b regime alone (methodology lines 288–292 table).
    Bottom bar: final F2 verdict after F1 cross-reference (methodology
    line 284) — present only when ``f2_verdicts.json`` was produced.
    """
    instances = f2b_data.get("per_instance", [])
    regime_counts = {
        "not_f2":    0,
        "f2_strong": 0,
        "f2_weak":   0,
        "f3":        0,
        "other":     0,
    }
    for inst in instances:
        r = inst.get("f2_regime")
        if r in regime_counts:
            regime_counts[r] += 1
        else:
            regime_counts["other"] += 1
    n_b = sum(regime_counts.values()) or 1

    # Order: not_f2 → f2_strong → f2_weak → f3
    order = ["not_f2", "f2_strong", "f2_weak", "f3", "other"]
    labels = {
        "not_f2":    "not_f2 (signal survived)",
        "f2_strong": "F2-strong (no mid-rise)",
        "f2_weak":   "F2-weak (moderate)",
        "f3":        "F3 candidate (strong)",
        "other":     "other",
    }
    colors = {**_REGIME_COLORS, "other": "#888888"}

    bars_y = ["F2-b regime"]
    left = 0
    for key in order:
        n = regime_counts[key]
        if n == 0:
            continue
        ax.barh(0, n / n_b * 100, left=left,
                color=colors[key], edgecolor="white", height=0.55,
                label=f"{labels[key]} ({n})")
        if n / n_b > 0.04:
            ax.text(left + n / n_b * 100 / 2, 0,
                    f"{n}\n{100*n/n_b:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
        left += n / n_b * 100

    # Final verdict bar (F1-cross-referenced)
    if verdict_data is not None and verdict_data.get("verdict_counts"):
        bars_y.append("F2 verdict\n(F1-crossref)")
        v_counts = verdict_data["verdict_counts"]
        n_v = sum(v_counts.values()) or 1
        # Order: not_routing_failure → F1 → F2_strong → F2_weak → F3 → ..._unverified → ..._f1b_nonsignif → undetermined
        verdict_order = [
            ("not_routing_failure",            C_GREEN,   "not_routing_failure"),
            ("F1",                             C_BLUE,    "F1 (year not read)"),
            ("F2_strong",                      C_RED,     "F2_strong"),
            ("F2_weak",                        C_ORANGE,  "F2_weak"),
            ("F3_candidate",                   C_PURPLE,  "F3_candidate"),
            # _unverified — no F1-a per-instance lookup available
            ("F2_strong_unverified",           "#FFAAAA", "F2_strong (no F1-a ref)"),
            ("F2_weak_unverified",             "#FFCC99", "F2_weak (no F1-a ref)"),
            ("F3_candidate_unverified",        "#DDAACC", "F3_cand. (no F1-a ref)"),
            # _f1b_nonsignif — F1-b Mann-Whitney non-significant
            ("F1_f1b_nonsignif",                       "#7799CC", "F1 (F1-b non-signif)"),
            ("F2_strong_f1b_nonsignif",                "#AA5555", "F2_strong (F1-b non-signif)"),
            ("F2_weak_f1b_nonsignif",                  "#CC8855", "F2_weak (F1-b non-signif)"),
            ("F3_candidate_f1b_nonsignif",             "#9966AA", "F3_cand. (F1-b non-signif)"),
            ("F2_strong_unverified_f1b_nonsignif",     "#DD9999", "F2_strong (no F1-a, F1-b NS)"),
            ("F2_weak_unverified_f1b_nonsignif",       "#DDAA88", "F2_weak (no F1-a, F1-b NS)"),
            ("F3_candidate_unverified_f1b_nonsignif",  "#BB99CC", "F3_cand. (no F1-a, F1-b NS)"),
            ("undetermined",                   C_GRAY,    "undetermined"),
        ]
        left = 0
        for key, color, label in verdict_order:
            n = v_counts.get(key, 0)
            if n == 0:
                continue
            ax.barh(1, n / n_v * 100, left=left,
                    color=color, edgecolor="white", height=0.55,
                    label=f"{label} ({n})")
            if n / n_v > 0.04:
                ax.text(left + n / n_v * 100 / 2, 1,
                        f"{n}\n{100*n/n_v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
            left += n / n_v * 100

    ax.set_yticks(range(len(bars_y)))
    ax.set_yticklabels(bars_y, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of F2-b instances")
    title_n = f2b_data.get("n", n_b)
    ax.set_title(
        f"F2 regime distribution  (n={title_n})\n"
        "Top: F2-b regime alone (lines 288–292).  "
        "Bottom: final verdict after F1-a Step-5 cross-reference (line 284).",
        pad=8,
    )
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    ax.grid(axis="y", alpha=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              frameon=False, fontsize=7.5)
    _panel_label(ax, "E")


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
    _panel_label(ax, "F")


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

    # Optional: F2 verdict file (only present when F2-b was run AND
    # ``run_f2_diagnostic.py`` finished its verdict assembly step).
    verdict_path = res_dir / "f2_verdicts.json"
    verdict_data = _load(verdict_path) if verdict_path.exists() else None

    # ── Figure 1: 5-panel summary  (A | B over C | D, with E full-width) ─────
    fig = plt.figure(figsize=(14, 13))
    gs  = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.65,
        wspace=0.38,
        left=0.07, right=0.97,
        top=0.93,  bottom=0.08,
        height_ratios=[1.0, 1.0, 0.7],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])     # full-width F2 regime panel

    panel_a(ax_a, filter_data)
    panel_b(ax_b, f2c_data)
    panel_c(ax_c, f2a_data)
    panel_d(ax_d, f2b_data)
    panel_regime(ax_e, f2b_data, verdict_data)

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
