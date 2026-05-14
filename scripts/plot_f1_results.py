#!/usr/bin/env python3
"""Generate publication-quality figures for F1 diagnostic results.

Usage:
    python scripts/plot_f1_results.py \
        --results results/f1_phi3 \
        --out     results/f1_phi3/figures
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── colour palette (colour-blind friendly) ───────────────────────────────────
C_SUCCESS = "#2196F3"   # blue
C_FAILURE = "#F44336"   # red
C_WEAK    = "#9E9E9E"   # grey (B6)
C_POS     = "#4CAF50"   # green  (↑ success coefficient)
C_NEG     = "#E53935"   # red    (↓ success coefficient)
C_FULL    = "#5C6BC0"   # indigo (full knockout)
C_DEEP    = "#EF6C00"   # orange (deep knockout)


def load(path: Path):
    with open(path) as f:
        return json.load(f)


# ── Figure 1 – F1-a: SAT Probe coefficients ──────────────────────────────────
def plot_f1a(data: dict, out_dir: Path):
    top = data["top_heads"]
    labels = [f"L{t['layer']}.H{t['head']}" for t in top]
    coefs  = [float(t["coef"]) for t in top]
    colors = [C_POS if c > 0 else C_NEG for c in coefs]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels[::-1], coefs[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Probe Coefficient (original-space units)")
    ax.set_title(
        f"F1-a  SAT Probe — Top-10 Heads by |Coefficient|\n"
        f"AUROC = {data['auroc']:.3f} ± {data['auroc_std']:.3f}  "
        f"  n = {data['n_samples']} ({data['n_positive']}+ / {data['n_negative']}−)"
    )

    pos_patch = mpatches.Patch(color=C_POS, label="↑ success  (higher year-attn → success)")
    neg_patch = mpatches.Patch(color=C_NEG, label="↓ success  (higher year-attn → failure)")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "f1a_probe_coefficients.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "f1a_probe_coefficients.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved] f1a_probe_coefficients")

    # ── F1-a Step 5: per-instance H_T-attention scalar & percentile sweep ───
    step5 = data.get("step5_f1_positive")
    if not step5 or not step5.get("scalar_per_instance"):
        return
    meta = data.get("instance_meta") or []
    scalars = np.array(step5["scalar_per_instance"], dtype=float)
    labels_succ = np.array([
        bool(m.get("success", False)) for m in meta
    ]) if meta else np.zeros(len(scalars), dtype=bool)
    if labels_succ.size != scalars.size:
        labels_succ = np.zeros(scalars.size, dtype=bool)

    thresholds = step5["threshold_by_percentile"]
    primary_p  = step5["primary_percentile"]
    f1_pos_mask = np.array(step5["f1_positive_by_percentile"][str(primary_p)], dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    succ_vals = scalars[labels_succ]
    fail_vals = scalars[~labels_succ]
    bins = np.linspace(scalars.min(), scalars.max(), 25) if scalars.size else 10
    ax.hist(succ_vals, bins=bins, alpha=0.6, color=C_SUCCESS,
            label=f"B1-success (n={succ_vals.size})", edgecolor="white")
    ax.hist(fail_vals, bins=bins, alpha=0.6, color=C_FAILURE,
            label=f"B1-failure (n={fail_vals.size})", edgecolor="white")

    palette = {20: "#9E9E9E", 25: "#212121", 33: "#616161"}
    for p_str, thr in thresholds.items():
        p = int(p_str)
        style = "-" if p == primary_p else "--"
        ax.axvline(
            float(thr), color=palette.get(p, "#212121"), linewidth=1.4, linestyle=style,
            label=f"p{p} threshold = {float(thr):.2e}"
                  + ("  (primary)" if p == primary_p else ""),
        )
    n_f1_pos = int(f1_pos_mask.sum())
    is_fb = step5.get("is_fallback")
    ht_label = (
        "top-3 by |coef| (FALLBACK)" if is_fb
        else "H_T (prerequisite)"
    )
    ax.set_xlabel(r"$\bar{A}^{\mathcal{H}_T}_i$ — mean attention to year tokens at $\mathcal{H}_T$")
    ax.set_ylabel("Instance count")
    ax.set_title(
        f"F1-a Step 5  Per-Instance H_T Scalar  ({ht_label})\n"
        f"F1-positive @ p{primary_p}: {n_f1_pos}/{scalars.size} "
        f"({100*n_f1_pos/max(scalars.size, 1):.1f}%)"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "f1a_step5_f1_positive.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "f1a_step5_f1_positive.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved] f1a_step5_f1_positive")


# ── Figure 2 – F1-b: Attention heatmaps + mean comparison ───────────────────
def plot_f1b(data: dict, out_dir: Path):
    succ = data.get("b1_success") or {}
    fail = data.get("b1_failure") or {}
    weak_label = data.get("weak_group_label", "B3")
    weak_key   = weak_label.lower()
    weak       = data.get(weak_key) or data.get("b3") or {}
    if not succ.get("mean_attn_all") or not fail.get("mean_attn_all"):
        return

    mat_s = np.array(succ["mean_attn_all"], dtype=float)
    mat_f = np.array(fail["mean_attn_all"], dtype=float)
    n_layers, n_heads = mat_s.shape
    mat_diff = mat_s - mat_f          # positive = success attends more

    # ── 2a: side-by-side heatmaps ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    vmax = max(mat_s.max(), mat_f.max())

    for ax, mat, title in zip(axes, [mat_s, mat_f, mat_diff],
                               [f"B1-success  (n={succ.get('count', '?')})",
                                f"B1-failure  (n={fail.get('count', '?')})",
                                "Δ  (success − failure)"]):
        if "Δ" in title:
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                           vmin=-np.abs(mat_diff).max(),
                           vmax= np.abs(mat_diff).max())
        else:
            im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Attention weight")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

        # mark top-3 H_T-fallback heads (methodology F1-a Step 5 fallback)
        head_ref = succ.get("per_head") or fail.get("per_head") or []
        for h_info in head_ref[:3]:
            l, h = h_info["layer"], h_info["head"]
            ax.add_patch(plt.Rectangle((h - 0.5, l - 0.5), 1, 1,
                                       fill=False, edgecolor="gold",
                                       linewidth=2.0))

    fig.suptitle(
        f"F1-b  Attention to Year / <YEAR>-placeholder per (Layer, Head)\n"
        "Gold boxes = top-3 H_T heads (fallback selection from F1-a probe)",
        fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out_dir / "f1b_attention_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "f1b_attention_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved] f1b_attention_heatmap")

    # ── 2b: per-layer mean attention (averaged over heads) ───────────────────
    layer_mean_s = mat_s.mean(axis=1)
    layer_mean_f = mat_f.mean(axis=1)
    layers = np.arange(n_layers)
    deep_lo, deep_hi = (n_layers * 3) // 4, n_layers - 1

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, layer_mean_s, color=C_SUCCESS, linewidth=2.0,
            marker="o", markersize=3, label=f"B1-success (n={succ.get('count', '?')})")
    ax.plot(layers, layer_mean_f, color=C_FAILURE, linewidth=2.0,
            marker="o", markersize=3, linestyle="--",
            label=f"B1-failure (n={fail.get('count', '?')})")
    if weak.get("mean_attn_all"):
        mat_w = np.array(weak["mean_attn_all"], dtype=float)
        ax.plot(layers, mat_w.mean(axis=1), color=C_WEAK, linewidth=1.6,
                marker="s", markersize=3, linestyle=":",
                label=f"{weak_label} (<YEAR>, n={weak.get('count', '?')})")
    ax.fill_between(layers, layer_mean_s, layer_mean_f,
                    where=layer_mean_s > layer_mean_f,
                    alpha=0.15, color=C_SUCCESS, label="success > failure")
    ax.axvspan(deep_lo, deep_hi, alpha=0.08, color=C_DEEP,
               label=f"Deep knockout window (L{deep_lo}–{deep_hi})")
    p_value = data.get("mann_whitney_p")
    pval_str = f"p = {p_value:.4f}" if isinstance(p_value, (int, float)) else "n/a"
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean attention to year / <YEAR> tokens")
    ax.set_title("F1-b  Mean Year-Token Attention per Layer\n"
                 f"Mann-Whitney U (success > failure): {pval_str}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "f1b_layer_profile.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "f1b_layer_profile.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved] f1b_layer_profile")


# ── Figure 3 – F1-c: Knockout effect distributions ──────────────────────────
def _f1c_drops(records: list[dict]) -> list[float]:
    return [r["p_new_drop_relative"] for r in (records or [])
            if "p_new_drop_relative" in r]


def _f1c_get_populations(data: dict) -> dict[str, dict]:
    """Return ``{"b1_success": ..., "b1_failure": ...}``, gracefully handling
    the legacy single-population (success only) layout."""
    pops = data.get("populations")
    if isinstance(pops, dict) and ("b1_success" in pops or "b1_failure" in pops):
        return pops
    # Legacy: top-level per_instance_* arrays were B1-success only.
    return {
        "b1_success": {
            "per_instance_full": data.get("per_instance_full", []),
            "per_instance_deep": data.get("per_instance_deep", []),
            "n_instances": data.get("n_instances", 0),
        },
        "b1_failure": {
            "per_instance_full": [], "per_instance_deep": [], "n_instances": 0,
        },
    }


def plot_f1c(data: dict, out_dir: Path):
    pops = _f1c_get_populations(data)
    succ = pops.get("b1_success", {})
    fail = pops.get("b1_failure", {})
    succ_full = _f1c_drops(succ.get("per_instance_full"))
    succ_deep = _f1c_drops(succ.get("per_instance_deep"))
    fail_full = _f1c_drops(fail.get("per_instance_full"))
    fail_deep = _f1c_drops(fail.get("per_instance_deep"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    rng = np.random.default_rng(0)
    jitter = 0.08

    # ── 3a: strip / scatter — success vs failure × full vs deep ─────────────
    ax = axes[0]
    column_specs = [
        (1, "B1-succ\nfull",  succ_full, C_SUCCESS),
        (2, "B1-succ\ndeep",  succ_deep, C_SUCCESS),
        (3, "B1-fail\nfull",  fail_full, C_FAILURE),
        (4, "B1-fail\ndeep",  fail_deep, C_FAILURE),
    ]
    for x, _, drops, color in column_specs:
        if not drops:
            continue
        xs = rng.uniform(-jitter, jitter, len(drops)) + x
        ax.scatter(xs, drops, color=color, alpha=0.7, s=36, zorder=3)
        med = np.median(drops); mn = np.mean(drops)
        ax.hlines(med, x - 0.27, x + 0.27, colors=color, linewidths=2.4, zorder=4)
        ax.plot(x, mn, marker="D", color=color, markersize=8, zorder=5,
                markeredgecolor="white")

    ax.axhline(0,    color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.10, color="grey",  linewidth=0.6, linestyle=":", alpha=0.7,
               label=">10% threshold")
    ax.set_xticks([s[0] for s in column_specs])
    ax.set_xticklabels([s[1] for s in column_specs])
    ax.set_ylabel("Relative p(answer_new) drop")
    ax.set_title("F1-c  Year-Token Knockout × Population\n"
                 "Line = median, Diamond = mean")
    ax.legend(fontsize=9, loc="upper right")

    # ── 3b: CDF — full knockout, success vs failure ──────────────────────────
    ax = axes[1]
    for drops, label, color in [
        (succ_full, f"B1-success  full  (n={len(succ_full)}, mean={np.mean(succ_full):.3f})"
                    if succ_full else "B1-success  full  (n=0)", C_SUCCESS),
        (fail_full, f"B1-failure  full  (n={len(fail_full)}, mean={np.mean(fail_full):.3f})"
                    if fail_full else "B1-failure  full  (n=0)", C_FAILURE),
    ]:
        if not drops:
            continue
        sorted_d = np.sort(drops)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.step(sorted_d, cdf, where="post", color=color, linewidth=2.0, label=label)

    ax.axvline(0.10, color="grey",  linewidth=0.8, linestyle=":", label=">10% threshold")
    ax.axvline(0.0,  color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Relative p(answer_new) drop")
    ax.set_ylabel("CDF")
    ax.set_title("F1-c  Full-Network Knockout CDF\nB1-success vs B1-failure")
    ax.legend(fontsize=9)

    # ── specificity ratios in the suptitle ──────────────────────────────────
    def _spec(pop: dict) -> Optional[float]:
        full = pop.get("full_stats") or {}
        v = full.get("mean_specificity_ratio")
        return float(v) if isinstance(v, (int, float)) else None
    spec_s = _spec(succ)
    spec_f = _spec(fail)
    spec_str = []
    if spec_s is not None: spec_str.append(f"succ year/random = {spec_s:.2f}x")
    if spec_f is not None: spec_str.append(f"fail year/random = {spec_f:.2f}x")
    sub = "   |   ".join(spec_str) if spec_str else ""

    fig.suptitle(
        f"F1-c  Attention Knockout — Year vs Random-Token Control\n{sub}",
        fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out_dir / "f1c_knockout_distribution.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "f1c_knockout_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved] f1c_knockout_distribution")


# ── Figure 4 – F1 Summary dashboard ─────────────────────────────────────────
def plot_summary(f1a, f1b, f1c, out_dir: Path):
    pops = _f1c_get_populations(f1c)
    succ_full = _f1c_drops(pops.get("b1_success", {}).get("per_instance_full"))
    fail_full = _f1c_drops(pops.get("b1_failure", {}).get("per_instance_full"))

    fig = plt.figure(figsize=(15, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)

    # — panel A: AUROC bar ————————————————————————————————————————————————————
    ax1 = fig.add_subplot(gs[0])
    auroc = f1a["auroc"]
    std   = f1a["auroc_std"]
    ax1.bar(["SAT Probe\nAUROC"], [auroc], yerr=[std], color=C_SUCCESS,
            capsize=8, width=0.4, error_kw={"linewidth": 1.5})
    ax1.axhline(0.5, color="grey", linewidth=1.0, linestyle="--", label="chance")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("AUROC")
    ax1.set_title("F1-a\nSAT Probe")
    ax1.legend(fontsize=9)
    ax1.text(0, auroc + std + 0.03, f"{auroc:.3f}±{std:.3f}",
             ha="center", fontsize=10, color=C_SUCCESS)

    # — panel B: attention bar comparison ————————————————————————————————————
    ax2 = fig.add_subplot(gs[1])
    weak_label = f1b.get("weak_group_label", "B3")
    weak_key   = weak_label.lower()
    succ_node = f1b.get("b1_success", {})
    fail_node = f1b.get("b1_failure", {})
    weak_node = f1b.get(weak_key) or f1b.get("b3") or {}

    def _attn_top(node: dict) -> Optional[float]:
        v = node.get("mean_attn_top3", node.get("mean_attn_top5"))
        return float(v) if isinstance(v, (int, float)) else None

    s_attn  = _attn_top(succ_node)
    f_attn  = _attn_top(fail_node)
    w_attn  = _attn_top(weak_node)
    bar_lbls = ["B1-success", "B1-failure", f"{weak_label}\n(<YEAR>)"]
    bar_vals = [v if v is not None else 0.0 for v in (s_attn, f_attn, w_attn)]
    ax2.bar(bar_lbls, bar_vals,
            color=[C_SUCCESS, C_FAILURE, C_WEAK],
            edgecolor="white")
    ax2.set_ylabel("Mean attention to year / <YEAR>\n(top-3 H_T-fallback heads)")
    p_value = f1b.get("mann_whitney_p")
    pval_str = f"p = {p_value:.4f}" if isinstance(p_value, (int, float)) else ""
    ax2.set_title(f"F1-b\nAttention Comparison\n{pval_str}")
    for i, v in enumerate(bar_vals):
        if v:
            ax2.text(i, v + max(bar_vals) * 0.02, f"{v:.4f}",
                     ha="center", fontsize=9)

    # — panel C: knockout CDF — success vs failure on the methodology
    #            primary (full-network) knockout ——————————————————————————————
    ax3 = fig.add_subplot(gs[2])
    plotted = False
    for drops, label, color in [
        (succ_full, "B1-success  full", C_SUCCESS),
        (fail_full, "B1-failure  full", C_FAILURE),
    ]:
        if not drops:
            continue
        sorted_d = np.sort(drops)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax3.step(sorted_d, cdf, where="post", color=color, linewidth=2.0,
                 label=f"{label}  (n={len(drops)}, mean={np.mean(drops):.3f})")
        plotted = True
    ax3.axvline(0.10, color="grey", linewidth=0.8, linestyle=":", label=">10% threshold")
    ax3.axvline(0.0,  color="black", linewidth=0.6, linestyle="--")
    ax3.set_xlabel("Relative p(answer_new) drop")
    ax3.set_ylabel("CDF")
    ax3.set_title("F1-c  Full-Network Knockout\nB1-success vs B1-failure")
    if plotted:
        ax3.legend(fontsize=9)

    fig.suptitle(
        f"F1 Diagnostic Summary — methodology-aligned "
        f"(C={f1a.get('probe_C', 0.05)}, agg={f1a.get('feature_aggregation', 'mean')}, "
        f"H_T = top-3 fallback)",
        fontsize=12, y=1.01
    )
    fig.savefig(out_dir / "f1_summary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "f1_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved] f1_summary")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/f1_phi3")
    parser.add_argument("--out",     default=None)
    args = parser.parse_args()

    res_dir = Path(args.results)
    out_dir = Path(args.out) if args.out else res_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    f1a = load(res_dir / "f1a_sat_probe.json")
    f1b = load(res_dir / "f1b_attention_comparison.json")
    f1c = load(res_dir / "f1c_attention_knockout.json")

    plot_f1a(f1a, out_dir)
    plot_f1b(f1b, out_dir)
    plot_f1c(f1c, out_dir)
    plot_summary(f1a, f1b, f1c, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
