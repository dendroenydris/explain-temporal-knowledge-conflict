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
    ax.set_title(f"B1 Failure Taxonomy\n(n={n_total})", pad=8)
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

    ax.text(
        0.02, 0.03,
        (f"B5+B6-={n_b5s_b6f}   B5-B6+={n_b5f_b6s}   "
         f"B5-B6-={n_b5f_b6f}"),
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=8, color="#222222",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.90),
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
                 f"(n={n_valid} valid instances)", pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Summary stats annotation (upper-left to avoid the legend)
    ax.text(
        0.03, 0.95,
        f"mean = {recs.mean():+.3f}\nmedian = {np.median(recs):+.3f}",
        transform=ax.transAxes, ha="left", va="top",
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
        f"F2-b: Logit Lens Trajectories — DESCRIPTIVE  (n={len(instances)})\n"
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
        bars_y.append("F2 verdict\n(DLA + F1-crossref)")
        raw_counts = verdict_data["verdict_counts"]
        v_counts = {k: 0 for k in [
            "not_routing_failure", "F1", "F2", "F3_candidate",
            "F2_unverified", "F3_candidate_unverified", "undetermined",
        ]}
        legacy_suffix_seen = False
        for key, n in raw_counts.items():
            norm = key
            if norm.endswith("_f1b_nonsignif"):
                legacy_suffix_seen = True
                norm = norm.removesuffix("_f1b_nonsignif")
            if norm.startswith(("F2_strong_unverified", "F2_weak_unverified")):
                norm = "F2_unverified"
            elif norm.startswith(("F2_strong", "F2_weak")):
                norm = "F2"
            elif norm.startswith("F3_candidate_unverified"):
                norm = "F3_candidate_unverified"
            elif norm.startswith("F3_candidate"):
                norm = "F3_candidate"
            v_counts[norm if norm in v_counts else "undetermined"] += n
        n_v = sum(v_counts.values()) or 1
        # Collapsed labels: F2-strong/F2-weak → "F2"; f1b is a separate boolean
        # (no longer a verdict suffix).  Order: not_routing_failure → F1 → F2 → F3.
        verdict_order = [
            ("not_routing_failure",     C_GREEN,   "not_routing_failure"),
            ("F1",                      C_BLUE,    "F1 (year not written)"),
            ("F2",                      C_RED,     "F2 (set, not routed)"),
            ("F3_candidate",            C_PURPLE,  "F3_candidate"),
            # _unverified — no F1-a per-instance lookup available
            ("F2_unverified",           "#FFAAAA", "F2 (no F1-a ref)"),
            ("F3_candidate_unverified", "#DDAACC", "F3_cand. (no F1-a ref)"),
            ("undetermined",            C_GRAY,    "undetermined"),
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
        if legacy_suffix_seen:
            ax.text(
                0.99, 0.02,
                "Note: legacy _f1b_nonsignif suffixes collapsed for plotting",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color=C_GRAY,
            )

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
# Panel DLA  –  Direct Logit Attribution (PRIMARY internal F1/F2 separator)
# ─────────────────────────────────────────────────────────────────────────────

def panel_dla(ax: plt.Axes, f2b_data: dict) -> None:
    """Per-instance Σ DLA(H_T) onto (answer_new − answer_old).

    DLA is the AUTHORITATIVE, late-crystallization-robust F1/F2 separator
    (Ortu 2024): >0 ⇒ H_T writes the answer direction (F2 "set but not
    routed"); ≤0 ⇒ H_T does not write it (F1 "not set").
    """
    instances = f2b_data.get("per_instance", [])
    vals = [i.get("dla_ht_sum") for i in instances if i.get("dla_ht_sum") is not None]
    if not vals:
        ax.text(0.5, 0.5, "No DLA data\n(rerun F2-b with temporal heads)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color=C_GRAY)
        ax.set_title("DLA readout (unavailable)", pad=8)
        _panel_label(ax, "DLA")
        return

    vals = np.asarray(vals, dtype=float)
    n_f2 = int((vals > 0).sum())
    n_f1 = int((vals <= 0).sum())
    lim = float(np.abs(vals).max()) * 1.05 or 1.0
    bins = np.linspace(-lim, lim, 30)
    _, _, patches = ax.hist(vals, bins=bins, color=C_BLUE, edgecolor="white",
                            linewidth=0.6, zorder=3)
    for patch, left in zip(patches, bins[:-1]):
        if left >= 0:
            patch.set_facecolor(C_ORANGE)
    ax.axvline(0.0, color="black", linewidth=1.4, linestyle="--",
               label=r"$\tau=0$  (F1 $\leq 0 <$ F2)")
    ax.set_xlabel(r"$\sum_{h \in H_T}$ DLA onto (answer_new − answer_old)  [logits]")
    ax.set_ylabel("# instances")
    ax.set_title("DLA readout — authoritative F1/F2 separator\n"
                 f"writes/F2 (>0): {n_f2}   ·   no-write/F1 (≤0): {n_f1}", pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.text(0.03, 0.95,
            f"mean = {vals.mean():+.3f}\nmedian = {np.median(vals):+.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))
    _panel_label(ax, "DLA")


# ─────────────────────────────────────────────────────────────────────────────
# Panel Summary  –  coverage + DLA split + McNemar (text panel)
# ─────────────────────────────────────────────────────────────────────────────

def panel_summary(ax: plt.Axes, f2b_data: dict, f2c_data: dict) -> None:
    """Compact text panel: lens-decodable coverage, DLA split, F2-c McNemar."""
    ax.axis("off")
    lines = ["F2 summary readouts", ""]

    cov = f2b_data.get("lens_decodable_coverage")
    if cov:
        lines.append("Lens-decodable coverage (rank-competitive before final layer):")
        lines.append(f"  {cov.get('n_decodable','?')}/{cov.get('n_total','?')}"
                     f"  ({100*cov.get('coverage',0):.1f}%)")
        lines.append("")

    dla = f2b_data.get("dla")
    if dla and dla.get("n_with_dla"):
        lines.append("DLA F1/F2 separation (authoritative):")
        lines.append(f"  writes/F2: {dla.get('n_dla_f2_writes','?')}   "
                     f"no-write/F1: {dla.get('n_dla_f1_no_write','?')}")
        md = dla.get("median_dla_ht_sum")
        if md is not None:
            lines.append(f"  median Σ DLA(H_T) = {md:+.3f}")
        lines.append("")

    mc = f2c_data.get("mcnemar_b5_vs_b6")
    if mc:
        lines.append("F2-c McNemar (PRIMARY; B5 vs B6, paired):")
        lines.append(f"  b(B5+B6-)={mc.get('b_b5success_b6fail','?')}  "
                     f"c(B5-B6+)={mc.get('c_b5fail_b6success','?')}  "
                     f"(odds {mc.get('odds_b_over_c',float('nan')):.1f}×)")
        if mc.get("chi2") is not None:
            lines.append(f"  χ²(cont.)={mc['chi2']:.2f}  p_exact={mc['p_exact']:.3g}  "
                         f"(prefer {mc.get('prefer','?')})")
    else:
        primary = f2c_data.get("b5xb6_primary")
        if primary:
            lines.append("F2-c paired behavior (legacy schema):")
            lines.append(f"  B5+B6-={primary.get('n_b5_success_b6_fail','?')}  "
                         f"B5-B6+={primary.get('n_b5_fail_b6_success','?')}  "
                         f"B5-B6-={primary.get('n_b5_fail_b6_fail','?')}")
            lines.append(f"  B5 accuracy={100*f2c_data.get('b5_accuracy',0):.1f}%  "
                         f"B6 accuracy={100*f2c_data.get('b6_accuracy',0):.1f}%")
            disamb = f2c_data.get("b1xb5xb6_disambiguation") or {}
            if disamb:
                lines.append(f"  B1f∩B5f∩B6f among B1f∩B5f: "
                             f"{100*disamb.get('rate_parametric_dominance_in_b1f_b5f',0):.1f}%")

    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=9, family="monospace", color="#222222",
            bbox=dict(boxstyle="round,pad=0.5", fc="#F4F4F8", ec="#CCCCCC", alpha=0.95))
    _panel_label(ax, "S")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/f2_diagnostic",
                        help="Directory containing F2 JSON result files")
    parser.add_argument("--out",     default="results/f2_diagnostic/figures",
                        help="Output directory for figures")
    parser.add_argument("--model-tag", default="phi3",
                        help="Model label for figure titles (phi3 / llama2 / mistral)")
    args = parser.parse_args()

    model_label = args.model_tag
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

    # ── Figure 1: 7-panel summary ────────────────────────────────────────────
    #   row0: A taxonomy   | B F2-c behavioral
    #   row1: C STR recov. | D trajectories (descriptive)
    #   row2: DLA (primary)| S summary (coverage/DLA/McNemar)
    #   row3: E verdict distribution (full-width)
    fig = plt.figure(figsize=(14, 19))
    gs  = gridspec.GridSpec(
        4, 2,
        figure=fig,
        hspace=1.00,
        wspace=0.38,
        left=0.07, right=0.97,
        top=0.94,  bottom=0.08,
        height_ratios=[1.0, 1.0, 1.0, 0.7],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_dla = fig.add_subplot(gs[2, 0])
    ax_s = fig.add_subplot(gs[2, 1])
    ax_e = fig.add_subplot(gs[3, :])     # full-width F2 verdict distribution

    panel_a(ax_a, filter_data)
    panel_b(ax_b, f2c_data)
    panel_c(ax_c, f2a_data)
    panel_d(ax_d, f2b_data)
    panel_dla(ax_dla, f2b_data)
    panel_summary(ax_s, f2b_data, f2c_data)
    panel_regime(ax_e, f2b_data, verdict_data)

    fig.suptitle(
        f'F2 Diagnostic: "Time Set but Not Routed"  —  {model_label}',
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
        f"F2-b: Logit Lens (descriptive)  ·  {model_label}  ·  failure cohort",
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
        f"(n={len(instances)})", pad=8,
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
