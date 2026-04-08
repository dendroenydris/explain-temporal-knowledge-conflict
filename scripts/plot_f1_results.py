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


# ── Figure 2 – F1-b: Attention heatmaps + mean comparison ───────────────────
def plot_f1b(data: dict, out_dir: Path):
    n_layers = 32
    n_heads  = 32

    def to_matrix(mean_attn_all):
        mat = np.zeros((n_layers, n_heads))
        for l, row in enumerate(mean_attn_all):
            for h, v in enumerate(row):
                mat[l, h] = v
        return mat

    mat_s = to_matrix(data["b1_success"]["mean_attn_all"])
    mat_f = to_matrix(data["b1_failure"]["mean_attn_all"])
    mat_diff = mat_s - mat_f          # positive = success attends more

    # ── 2a: side-by-side heatmaps ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    vmax = max(mat_s.max(), mat_f.max())

    for ax, mat, title in zip(axes, [mat_s, mat_f, mat_diff],
                               ["B5-success  (n=27)",
                                "B5-failure  (n=11)",
                                "Δ  (success − failure)"]):
        if "Δ" in title:
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                           vmin=-mat_diff.abs().max() if hasattr(mat_diff, "abs") else -abs(mat_diff).max(),
                           vmax=abs(mat_diff).max())
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                           vmin=-np.abs(mat_diff).max(),
                           vmax= np.abs(mat_diff).max())
        else:
            im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Attention weight")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

        # mark top-5 probe heads
        top5 = data.get("b1_success", {}).get("per_head", [])[:5]
        for h_info in top5:
            l, h = h_info["layer"], h_info["head"]
            ax.add_patch(plt.Rectangle((h - 0.5, l - 0.5), 1, 1,
                                       fill=False, edgecolor="gold",
                                       linewidth=2.0, label="probe head"))

    fig.suptitle(
        "F1-b  Attention to Year Tokens per (Layer, Head)\n"
        "Gold boxes = top-5 probe heads from F1-a",
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

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, layer_mean_s, color=C_SUCCESS, linewidth=2.0,
            marker="o", markersize=3, label="B5-success")
    ax.plot(layers, layer_mean_f, color=C_FAILURE, linewidth=2.0,
            marker="o", markersize=3, linestyle="--", label="B5-failure")
    ax.fill_between(layers, layer_mean_s, layer_mean_f,
                    where=layer_mean_s > layer_mean_f,
                    alpha=0.15, color=C_SUCCESS, label="success > failure")
    ax.axvspan(24, 31, alpha=0.08, color=C_DEEP, label="Deep knockout window (L24–31)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean attention to year tokens")
    ax.set_title("F1-b  Mean Year-Token Attention per Layer\n"
                 f"Mann-Whitney U (success > failure): p = 0.0038")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "f1b_layer_profile.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "f1b_layer_profile.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved] f1b_layer_profile")


# ── Figure 3 – F1-c: Knockout effect distributions ──────────────────────────
def plot_f1c(data: dict, out_dir: Path):
    drops_full = [r["p_new_drop_relative"]
                  for r in data["per_instance_full"]
                  if "p_new_drop_relative" in r]
    drops_deep = [r["p_new_drop_relative"]
                  for r in data["per_instance_deep"]
                  if "p_new_drop_relative" in r]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    # ── 3a: strip / scatter comparison ───────────────────────────────────────
    ax = axes[0]
    jitter = 0.08
    xs_full = np.random.default_rng(0).uniform(-jitter, jitter, len(drops_full))
    xs_deep = np.random.default_rng(1).uniform(-jitter, jitter, len(drops_deep))

    ax.scatter(1 + xs_full, drops_full, color=C_FULL, alpha=0.7, s=40, zorder=3,
               label=f"Full L0–31  (n={len(drops_full)})")
    ax.scatter(2 + xs_deep, drops_deep, color=C_DEEP, alpha=0.7, s=40, zorder=3,
               label=f"Deep L24–31  (n={len(drops_deep)})")

    for drops, x, c in [(drops_full, 1, C_FULL), (drops_deep, 2, C_DEEP)]:
        med = np.median(drops)
        mn  = np.mean(drops)
        ax.hlines(med, x - 0.25, x + 0.25, colors=c, linewidths=2.5, zorder=4)
        ax.plot(x, mn, marker="D", color=c, markersize=8, zorder=5,
                markeredgecolor="white")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.10, color="grey", linewidth=0.6, linestyle=":", alpha=0.7,
               label=">10% threshold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Full\nL0–31", "Deep\nL24–31"])
    ax.set_ylabel("Relative p(answer_new) drop")
    ax.set_title("F1-c  Per-Instance Knockout Effect\n"
                 "Line = median, Diamond = mean")
    ax.legend(fontsize=9)

    # ── 3b: CDF comparison ───────────────────────────────────────────────────
    ax = axes[1]
    for drops, label, color in [
        (drops_full, f"Full L0–31  (mean={np.mean(drops_full):.3f})", C_FULL),
        (drops_deep, f"Deep L24–31 (mean={np.mean(drops_deep):.3f})", C_DEEP),
    ]:
        sorted_d = np.sort(drops)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.step(sorted_d, cdf, where="post", color=color, linewidth=2.0, label=label)

    ax.axvline(0.10, color="grey", linewidth=0.8, linestyle=":", label=">10% threshold")
    ax.axvline(0.0,  color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Relative p(answer_new) drop")
    ax.set_ylabel("CDF")
    ax.set_title("F1-c  Cumulative Distribution of Dropout Effect")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"F1-c  Attention Knockout on B5-success Instances (n={len(drops_full)})\n"
        f"Full: {sum(d>0.1 for d in drops_full)/len(drops_full):.0%}  >10% drop   |   "
        f"Deep: {sum(d>0.1 for d in drops_deep)/len(drops_deep):.0%}  >10% drop",
        fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out_dir / "f1c_knockout_distribution.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "f1c_knockout_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved] f1c_knockout_distribution")


# ── Figure 4 – F1 Summary dashboard ─────────────────────────────────────────
def plot_summary(f1a, f1b, f1c, out_dir: Path):
    drops_deep = [r["p_new_drop_relative"]
                  for r in f1c["per_instance_deep"]
                  if "p_new_drop_relative" in r]

    fig = plt.figure(figsize=(14, 5))
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
    s_attn = f1b["b1_success"]["mean_attn_top5"]
    f_attn = f1b["b1_failure"]["mean_attn_top5"]
    b3_attn = f1b["b3"]["mean_attn_top5"]
    ax2.bar(["B5-success", "B5-failure", "B6\n(no year)"],
            [s_attn, f_attn, b3_attn],
            color=[C_SUCCESS, C_FAILURE, C_WEAK],
            edgecolor="white")
    ax2.set_ylabel("Mean attention to year tokens\n(top-5 probe heads)")
    ax2.set_title(f"F1-b\nAttention Comparison\np = 0.0038")
    for i, v in enumerate([s_attn, f_attn, b3_attn]):
        ax2.text(i, v + 2e-5, f"{v:.4f}", ha="center", fontsize=9)

    # — panel C: knockout CDF ————————————————————————————————————————————————
    ax3 = fig.add_subplot(gs[2])
    sorted_d = np.sort(drops_deep)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax3.step(sorted_d, cdf, where="post", color=C_DEEP, linewidth=2.0)
    ax3.fill_betweenx(cdf, 0, sorted_d,
                      where=np.array(sorted_d) > 0.10,
                      alpha=0.15, color=C_DEEP)
    ax3.axvline(0.10, color="grey", linewidth=0.8, linestyle=":", label=">10% threshold")
    ax3.axvline(0.0,  color="black", linewidth=0.6, linestyle="--")
    pct = sum(1 for d in drops_deep if d > 0.10) / len(drops_deep)
    ax3.set_xlabel("Relative p(answer_new) drop")
    ax3.set_ylabel("CDF")
    ax3.set_title(f"F1-c  Deep Knockout (L24–31)\n{pct:.0%} instances > 10% drop")
    ax3.legend(fontsize=9)

    fig.suptitle(
        "F1 Diagnostic Summary — Phi-3-mini-4k-instruct  |  B5 multi-span context  |  n = 38",
        fontsize=13, y=1.01
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
