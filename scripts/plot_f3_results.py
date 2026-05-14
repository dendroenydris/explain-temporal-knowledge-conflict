#!/usr/bin/env python3
"""Research-ready figures for F3 Diagnostic results.

Generates a 6-panel figure matching the methodology section *F3 Diagnosis*
(L364–L728):

  Panel A  —  F3-a trajectory rates per param_class (PARAM_OLD /
              PARAM_OTHER / PARAM_NEW), with positive-control gates.
  Panel B  —  F3-a Logit Lens trajectories ``P^l(answer_new)`` per layer,
              coloured by trajectory presence; ℓ_HT / ℓ_R marks overlaid.
  Panel C  —  F3-0.5 routing-set heads (per-head attribution heat-map);
              top-decile heads highlighted plus ω overlap statistics.
  Panel D  —  F3-b dual (M)/(Z) specificity ratios with bootstrap CIs.
  Panel E  —  F3-c Step 2–3 Override (ΔD), Chain (I^σ), and flip-rate
              bar charts per (σ, panel).
  Panel F  —  F3-c Step 4 Content containment ``r_conflict /
              r_closed-book / r_random,late / r_random,mid`` with
              ``0.8 · r_closed-book`` anchor.

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
    "PARAM_OLD":   C_RED,
    "PARAM_OTHER": C_ORANGE,
    "PARAM_NEW":   C_BLUE,
}

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


# ─────────────────────────────────────────────────────────────────────
# Panel A — F3-a trajectory rates
# ─────────────────────────────────────────────────────────────────────


def panel_a(ax: plt.Axes, f3a: dict | None) -> None:
    ax.set_title("F3-a Trajectory rates by param_class")
    if not f3a:
        ax.text(0.5, 0.5, "(F3-a not run)", ha="center", va="center",
                transform=ax.transAxes, color=C_GRAY)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "A")
        return

    summary = f3a.get("summary", {})
    classes = ["PARAM_OLD", "PARAM_OTHER", "PARAM_NEW"]
    rates = [summary.get("f3_traj_rate", {}).get(c, 0.0) for c in classes]
    counts = [summary.get("counts_by_class", {}).get(c, 0) for c in classes]
    bars = ax.bar(classes, rates,
                  color=[CLASS_COLOR[c] for c in classes],
                  edgecolor="black", linewidth=0.6)

    # Annotate with counts.
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f"n={count}",
                ha="center", va="bottom", fontsize=8, color="black")

    # Reference lines from positive-control gates.
    ax.axhline(0.40, ls="--", color=C_RED, alpha=0.6,
               label="PARAM_OLD ≥ 0.40 (positive control)")
    ax.axhline(0.20, ls="--", color=C_BLUE, alpha=0.6,
               label="PARAM_NEW ≤ 0.20")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F3-trajectory rate")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    reading = summary.get("reading", "")
    ax.set_xlabel(f"reading: {reading}", fontsize=8)
    _panel_label(ax, "A")


# ─────────────────────────────────────────────────────────────────────
# Panel B — F3-a Logit Lens trajectories
# ─────────────────────────────────────────────────────────────────────


def panel_b(ax: plt.Axes, f3a: dict | None) -> None:
    ax.set_title("F3-a Logit-Lens trajectories  P^l(answer_new)")
    if not f3a:
        ax.text(0.5, 0.5, "(F3-a not run)", ha="center", va="center",
                transform=ax.transAxes, color=C_GRAY)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "B")
        return

    per = f3a.get("per_instance", [])
    summary = f3a.get("summary", {}) or {}
    n_layers = summary.get("n_layers", 32)
    ell_HT = f3a.get("ell_HT")
    ell_R = f3a.get("ell_R")
    mid = summary.get("mid_window", [None, None])

    # Mid-window shaded band.
    if mid[0] is not None and mid[1] is not None:
        ax.axvspan(mid[0], mid[1], color=C_YELLOW, alpha=0.18, zorder=0,
                   label=f"mid-window [L/4,2L/3]")

    # Trajectories grouped by class + trajectory-bearing.
    n_drawn = {"PARAM_OLD": 0, "PARAM_OTHER": 0, "PARAM_NEW": 0}
    cap = 80   # avoid drawing thousands of lines
    for row in per:
        cls = row.get("param_class", "?")
        if cls not in CLASS_COLOR:
            continue
        if n_drawn.get(cls, cap) >= cap:
            continue
        n_drawn[cls] = n_drawn.get(cls, 0) + 1
        probs = row.get("p_new", [])
        color = CLASS_COLOR[cls]
        alpha = 0.7 if row.get("is_f3_trajectory") else 0.18
        lw = 1.0 if row.get("is_f3_trajectory") else 0.5
        ax.plot(range(len(probs)), probs, color=color, alpha=alpha, lw=lw)

    if ell_HT is not None:
        ax.axvline(ell_HT, color=C_GREEN, ls=":", lw=1.5, label=f"ℓ_HT={ell_HT}")
    if ell_R is not None:
        ax.axvline(ell_R, color=C_PURPLE, ls=":", lw=1.5, label=f"ℓ_R={ell_R}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("P^l(answer_new)")
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(0, 1.0)

    legend_handles = [
        mpatches.Patch(color=C_RED,    label="PARAM_OLD"),
        mpatches.Patch(color=C_ORANGE, label="PARAM_OTHER"),
        mpatches.Patch(color=C_BLUE,   label="PARAM_NEW"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9,
              fontsize=7)
    _panel_label(ax, "B")


# ─────────────────────────────────────────────────────────────────────
# Panel C — F3-0.5 attribution heat-map (Stage 1 minus Stage 2)
# ─────────────────────────────────────────────────────────────────────


def panel_c(ax: plt.Axes, f3half: dict | None) -> None:
    ax.set_title("F3-0.5 Routing-set attribution (Stage 1 + Stage 2 overlap)")
    if not f3half:
        ax.text(0.5, 0.5, "(F3-0.5 not run)", ha="center", va="center",
                transform=ax.transAxes, color=C_GRAY)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "C")
        return

    stage1 = np.asarray(f3half["stage1"]["per_head_score"])
    stage2 = np.asarray(f3half["stage2_pooled"]["per_head_score"])
    if stage1.size == 0:
        ax.text(0.5, 0.5, "(empty attribution matrix)", ha="center", va="center",
                transform=ax.transAxes, color=C_GRAY)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "C")
        return

    diff = stage1 - stage2  # positive ⇒ Stage 1 specific (i.e. R_succ-leaning)
    vmax = max(abs(diff.min()), abs(diff.max()), 1e-9)
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    # Mark R_pooled.
    R = f3half.get("R_pooled", [])
    if R:
        ys = [int(l) for l, _ in R]
        xs = [int(h) for _, h in R]
        ax.scatter(xs, ys, s=22, facecolors="none", edgecolors="black",
                   linewidths=1.2, label=f"R_pooled (|R|={len(R)})")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=7)

    note = (
        f"ω(pooled)={f3half['omega_pooled']:.2f}, "
        f"ω(OLD)={f3half['omega_param_old']:.2f}, "
        f"ω(OTHER)={f3half['omega_param_other']:.2f}; rule={f3half['selection_rule']}"
    )
    ax.set_title(f"F3-0.5 Routing-set\n{note}", fontsize=9)
    _panel_label(ax, "C")


# ─────────────────────────────────────────────────────────────────────
# Panel D — F3-b specificity ratios
# ─────────────────────────────────────────────────────────────────────


def panel_d(ax: plt.Axes, f3b: dict | None) -> None:
    ax.set_title("F3-b Specificity ratios (95% CI)")
    if not f3b:
        ax.text(0.5, 0.5, "(F3-b not run)", ha="center", va="center",
                transform=ax.transAxes, color=C_GRAY)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "D")
        return

    labels = ["δR/δrand-SL", "δR/δrand-DM"]
    ratios = [f3b["ratio_R_over_rand_same"],
              f3b["ratio_R_over_rand_depth"]]
    cis = [f3b["ratio_R_over_rand_same_CI"],
           f3b["ratio_R_over_rand_depth_CI"]]
    xs = np.arange(len(labels))
    yerr_lo = [r - c[0] for r, c in zip(ratios, cis)]
    yerr_hi = [c[1] - r for r, c in zip(ratios, cis)]
    ax.bar(xs, ratios, color=C_BLUE, edgecolor="black", linewidth=0.6,
           yerr=[yerr_lo, yerr_hi], capsize=4)
    ax.axhline(1.0, color=C_GRAY, ls="--", lw=1.0, label="ratio = 1")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("Ratio")
    rho = f3b.get("rho_HT")
    rho_text = f"ρ_HT = {rho:.2f}" if isinstance(rho, (float, int)) else "ρ_HT = n/a"
    ax.set_xlabel(f"primary={f3b.get('primary_protocol', '?')}  •  {rho_text}\n"
                  f"{f3b.get('routed_verdict', '')}",
                  fontsize=8)
    _panel_label(ax, "D")


# ─────────────────────────────────────────────────────────────────────
# Panel E — F3-c Step 2–3 Override + Chain
# ─────────────────────────────────────────────────────────────────────


def panel_e(ax: plt.Axes, f3c_files: list[tuple[str, str, dict]]) -> None:
    """``f3c_files`` is a list of ``(sigma, panel, json_dict)``."""
    ax.set_title("F3-c Step 2–3 Override (ΔD) + Chain (I^σ)")
    if not f3c_files:
        ax.text(0.5, 0.5, "(F3-c Step 2-3 not run)", ha="center", va="center",
                transform=ax.transAxes, color=C_GRAY)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "E")
        return

    labels = [f"{s}/{p}" for s, p, _ in f3c_files]
    xs = np.arange(len(labels))
    width = 0.35
    dD = [d.get("override_dD_mean", 0.0) for _, _, d in f3c_files]
    dD_CI = [d.get("override_dD_CI", (0.0, 0.0)) for _, _, d in f3c_files]
    chain = [d.get("chain_I_sigma_mean") or 0.0 for _, _, d in f3c_files]
    chain_CI = [d.get("chain_I_sigma_CI") or (0.0, 0.0) for _, _, d in f3c_files]

    dD_lo = [v - c[0] for v, c in zip(dD, dD_CI)]
    dD_hi = [c[1] - v for v, c in zip(dD, dD_CI)]
    ch_lo = [v - c[0] for v, c in zip(chain, chain_CI)]
    ch_hi = [c[1] - v for v, c in zip(chain, chain_CI)]

    ax.bar(xs - width / 2, dD, width=width, color=C_PURPLE,
           label="ΔD = D(3)−D(1)", edgecolor="black", linewidth=0.6,
           yerr=[dD_lo, dD_hi], capsize=3)
    ax.bar(xs + width / 2, chain, width=width, color=C_GREEN,
           label="I^σ (logit space)", edgecolor="black", linewidth=0.6,
           yerr=[ch_lo, ch_hi], capsize=3)
    ax.axhline(0, color=C_GRAY, ls="-", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Effect size")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    _panel_label(ax, "E")


# ─────────────────────────────────────────────────────────────────────
# Panel F — F3-c Step 4 Content containment
# ─────────────────────────────────────────────────────────────────────


def panel_f(ax: plt.Axes, content_files: list[tuple[str, str, dict]]) -> None:
    ax.set_title("F3-c Step 4 Content containment (r)")
    if not content_files:
        ax.text(0.5, 0.5, "(F3-c Step 4 not run)", ha="center", va="center",
                transform=ax.transAxes, color=C_GRAY)
        ax.set_xticks([]); ax.set_yticks([])
        _panel_label(ax, "F")
        return

    labels = [f"{s}/{p}" for s, p, _ in content_files]
    xs = np.arange(len(labels))
    width = 0.20

    r_conf  = [d.get("r_conflict", 0.0)     for _, _, d in content_files]
    r_cb    = [d.get("r_closed_book", 0.0)  for _, _, d in content_files]
    r_late  = [d.get("r_random_late", 0.0)  for _, _, d in content_files]
    r_mid   = [d.get("r_random_mid", 0.0)   for _, _, d in content_files]

    ax.bar(xs - 1.5 * width, r_conf, width=width, color=C_PURPLE,
           label="r_conflict", edgecolor="black", linewidth=0.6)
    ax.bar(xs - 0.5 * width, r_cb, width=width, color=C_BLUE,
           label="r_closed_book", edgecolor="black", linewidth=0.6)
    ax.bar(xs + 0.5 * width, r_late, width=width, color=C_ORANGE,
           label="r_random_late", edgecolor="black", linewidth=0.6)
    ax.bar(xs + 1.5 * width, r_mid, width=width, color=C_GRAY,
           label="r_random_mid", edgecolor="black", linewidth=0.6)

    # 0.8 anchor markers.
    for i, v in enumerate(r_cb):
        ax.hlines(0.8 * v, xs[i] - 0.45, xs[i] + 0.45, colors=C_RED,
                  linewidth=1.2, linestyles="--",
                  label="0.8 · r_closed_book" if i == 0 else None)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Top-k containment fraction")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    _panel_label(ax, "F")


# ─────────────────────────────────────────────────────────────────────
# Title strip — final F3 verdict
# ─────────────────────────────────────────────────────────────────────


def title_strip(fig, verdict: dict | None) -> None:
    if verdict is None:
        return
    title = verdict.get("title", "F3")
    fig.suptitle(f"F3 Diagnosis — {title}", fontsize=14, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


def collect_f3c_arm_panel(results_dir: Path, kind: str) -> list[tuple[str, str, dict]]:
    """``kind`` is ``"step2_3"`` or ``"step4"``.

    Returns a sorted list of ``(sigma, panel, data)`` triples loaded from
    ``f3c_{step}_{sigma}_{panel}.json`` files.
    """
    pattern = "f3c_step2_3_*.json" if kind == "step2_3" else "f3c_step4_*.json"
    out: list[tuple[str, str, dict]] = []
    for path in sorted(results_dir.glob(pattern)):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        sigma = parts[-2]
        panel = parts[-1]
        data = _load(path)
        if data is not None:
            out.append((sigma, panel, data))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True,
                        help="Directory containing f3*.json outputs")
    parser.add_argument("--out", default=None,
                        help="Output directory for figures (defaults to results/figures)")
    parser.add_argument("--figsize", nargs=2, type=float, default=[14, 13])
    args = parser.parse_args()

    results = Path(args.results)
    out_dir = Path(args.out) if args.out else (results / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    f3a    = _load(results / "f3a_trajectory.json")
    f3half = _load(results / "f3_half_attribution.json")
    f3b    = _load(results / "f3b_ablation.json")
    verdict = _load(results / "f3_verdict.json")
    f3c23  = collect_f3c_arm_panel(results, "step2_3")
    f3c4   = collect_f3c_arm_panel(results, "step4")

    fig = plt.figure(figsize=tuple(args.figsize), constrained_layout=False)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.32,
                            left=0.07, right=0.97, top=0.93, bottom=0.07)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    panel_a(ax_a, f3a)
    panel_b(ax_b, f3a)
    panel_c(ax_c, f3half)
    panel_d(ax_d, f3b)
    panel_e(ax_e, f3c23)
    panel_f(ax_f, f3c4)
    title_strip(fig, verdict)

    pdf_path = out_dir / "f3_results.pdf"
    png_path = out_dir / "f3_results.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
