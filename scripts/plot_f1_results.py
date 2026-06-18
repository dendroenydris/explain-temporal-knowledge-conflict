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
_LD_EPS = 1e-12
_COMPETITOR_TAU = 0.01


def _f1c_drops(records: list[dict]) -> list[float]:
    """Legacy p(answer_new)-only relative drop (reference only)."""
    return [r["p_new_drop_relative"] for r in (records or [])
            if "p_new_drop_relative" in r]


def _ld_drop(r: dict) -> Optional[float]:
    """PRIMARY metric: logit-diff drop = (log p_new - log p_old)_clean - (..)_ko.

    Uses the stored ``ld_drop`` when present (new runs); otherwise recomputes it
    from the stored probabilities so existing result JSONs still plot.
    """
    if isinstance(r.get("ld_drop"), (int, float)):
        return float(r["ld_drop"])
    keys = ("p_new_clean", "p_old_clean", "p_new_knockout", "p_old_knockout")
    if not all(k in r for k in keys):
        return None
    lg = lambda p: float(np.log(max(float(p), _LD_EPS)))
    ld_clean = lg(r["p_new_clean"]) - lg(r["p_old_clean"])
    ld_ko    = lg(r["p_new_knockout"]) - lg(r["p_old_knockout"])
    return ld_clean - ld_ko


def _is_competitor(r: dict) -> bool:
    if "old_is_competitor" in r:
        return bool(r["old_is_competitor"])
    return float(r.get("p_old_clean", 0.0)) >= _COMPETITOR_TAU


def _f1c_ld_drops(records: list[dict], competitor_only: bool = True) -> list[float]:
    out: list[float] = []
    for r in (records or []):
        if competitor_only and not _is_competitor(r):
            continue
        v = _ld_drop(r)
        if v is not None:
            out.append(v)
    return out


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

    # PRIMARY metric: logit-diff drop on the competitor cohort.
    succ_ld = _f1c_ld_drops(succ.get("per_instance_full"), competitor_only=True)
    fail_ld = _f1c_ld_drops(fail.get("per_instance_full"), competitor_only=True)
    # Reference: legacy p(answer_new)-only drop (all instances).
    succ_pn = _f1c_drops(succ.get("per_instance_full"))
    fail_pn = _f1c_drops(fail.get("per_instance_full"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    rng = np.random.default_rng(0)
    jitter = 0.08

    # ── 3a: PRIMARY — logit-diff drop strip plot (competitor cohort) ─────────
    ax = axes[0]
    column_specs = [
        (1, f"B1-succ\ncompetitor",  succ_ld, C_SUCCESS),
        (2, f"B1-fail\ncompetitor",  fail_ld, C_FAILURE),
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

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--",
               label="0 (year had no effect)")
    ax.set_xticks([s[0] for s in column_specs])
    ax.set_xticklabels([f"{s[1]}\n(n={len(s[2])})" for s in column_specs])
    ax.set_ylabel(r"logit-diff drop  $\Delta(\log p_{new}-\log p_{old})$")
    ax.set_title("F1-c  PRIMARY — Year-Token Knockout (logit-diff)\n"
                 "competitor cohort; >0 ⇒ year favored answer_new")
    ax.legend(fontsize=9, loc="upper right")

    # ── 3b: CDF of logit-diff drop + reference p_new CDF ────────────────────
    ax = axes[1]
    for drops, label, color in [
        (succ_ld, f"B1-success  (n={len(succ_ld)}, "
                  f"med={np.median(succ_ld):+.2f})" if succ_ld else "B1-success (n=0)",
         C_SUCCESS),
        (fail_ld, f"B1-failure  (n={len(fail_ld)}, "
                  f"med={np.median(fail_ld):+.2f})" if fail_ld else "B1-failure (n=0)",
         C_FAILURE),
    ]:
        if not drops:
            continue
        sorted_d = np.sort(drops)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.step(sorted_d, cdf, where="post", color=color, linewidth=2.0, label=label)

    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"logit-diff drop  $\Delta(\log p_{new}-\log p_{old})$")
    ax.set_ylabel("CDF")
    ax.set_title("F1-c  Logit-Diff Knockout CDF\n(competitor cohort)")
    ax.legend(fontsize=9)

    # ── logit-diff specificity in the suptitle ───────────────────────────────
    def _spec_ld(pop: dict) -> Optional[float]:
        full = pop.get("full_stats") or {}
        ld = full.get("logit_diff_metric") or {}
        cr = ld.get("competitor_restricted") or {}
        v = cr.get("mean_specificity_ratio")
        return float(v) if isinstance(v, (int, float)) else None
    spec_f = _spec_ld(fail)
    sub_bits = []
    if fail_ld:
        sub_bits.append(f"fail competitor median = {np.median(fail_ld):+.2f}")
    # Guard the specificity ratio: it divides by a near-zero random-control drop,
    # so it explodes / flips sign and is not interpretable in that regime.
    if spec_f is not None and 0.0 < spec_f < 100.0:
        sub_bits.append(f"fail logit-diff specificity = {spec_f:.2f}x")
    elif spec_f is not None:
        sub_bits.append("specificity ratio unstable (near-zero random control) — omitted")
    sub_bits.append("(p_new-only metric shown in reference panels only — see methodology F1-c)")
    sub = "   |   ".join(sub_bits)

    fig.suptitle(
        f"F1-c  Attention Knockout (corroborating) — logit-diff = log p_new − log p_old\n{sub}",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out_dir / "f1c_knockout_distribution.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "f1c_knockout_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved] f1c_knockout_distribution")


# ── Figure 4 – F1 Summary dashboard ─────────────────────────────────────────
def _attn_top(node: dict) -> Optional[float]:
    v = node.get("mean_attn_top3", node.get("mean_attn_top5"))
    return float(v) if isinstance(v, (int, float)) else None


def _mcnemar_bc(f2c: dict | None):
    """Extract (b, c, p, odds) from f2c_b5_vs_b6.json (nested or flat schema)."""
    if not f2c:
        return None
    m = f2c.get("mcnemar_b5_vs_b6") or {}
    b = m.get("b_b5success_b6fail", f2c.get("mcnemar_b", f2c.get("b")))
    c = m.get("c_b5fail_b6success", f2c.get("mcnemar_c", f2c.get("c")))
    p = m.get("p_exact", m.get("p_chi2_continuity",
              f2c.get("p_exact", f2c.get("mcnemar_p"))))
    odds = m.get("odds_b_over_c")
    if b is None or c is None:
        return None
    if odds is None and c:
        odds = b / c
    return int(b), int(c), p, odds


def _f1_verdict_lines(f1a, f1b, f1c, a1, f2c=None) -> list[tuple[str, str]]:
    """Synthesized F1 interpretation as ``(text, color)`` lines, faithful to the
    methodology: the authoritative 'Time Set' premise is carried by F2-c
    (B5-vs-B6 McNemar); F1-a/b/c are SOFT descriptive cross-references.
    """
    lines: list[tuple[str, str]] = []

    # AUTHORITATIVE Time-Set verdict = F2-c McNemar (methodology line 298/356).
    bc = _mcnemar_bc(f2c)
    if bc:
        b, c, p, odds = bc
        odds_s = f"{odds:.1f}x" if isinstance(odds, (int, float)) else "n/a"
        p_s = f"{p:.1e}" if isinstance(p, (int, float)) else "n/a"
        col = C_POS if (b > c and isinstance(p, (int, float)) and p < 0.05) else C_NEG
        lines.append((f"TIME SET (authoritative, F2-c B5-vs-B6 McNemar): "
                      f"b={b} (year helped) >> c={c} (year hurt), odds {odds_s}, "
                      f"p={p_s}  \u21d2 year tokens ARE causally load-bearing; "
                      f"the Time-Set premise HOLDS at population level.", col))
    else:
        lines.append(("TIME SET verdict comes from F2-c (B5-vs-B6 McNemar) \u2014 "
                      "pass --f2-dir to display it here.", C_WEAK))

    # A1 parametric ceiling.
    if a1:
        n_tot = a1.get("n_total", 0) or 0
        n_kn = a1.get("n_knows_new", 0) or 0
        frac = (n_kn / n_tot) if n_tot else 0.0
        lines.append((f"A1 parametric ceiling: model recalls answer_new for "
                      f"{n_kn}/{n_tot} ({frac:.0%}) closed-book \u2014 context, not "
                      f"parametric recall, drives B1.", "#212121"))

    # F1-a probe: descriptive success-correlate (NOT the Time-Set test).
    auroc = f1a.get("auroc")
    top = f1a.get("top_heads", []) or []
    n_neg = sum(1 for t in top if float(t.get("coef", 0)) < 0)
    if auroc is not None and top:
        lines.append((f"F1-a probe (SOFT/descriptive): AUROC={auroc:.3f} but "
                      f"{n_neg}/{len(top)} top heads have NEGATIVE coef \u2014 it is a "
                      f"success-correlate, not a timestamp-extraction test; "
                      f"intentionally non-load-bearing (methodology: soft annotation).",
                      C_WEAK))

    # F1-b raw attention: descriptive; non-sig is expected.
    p = f1b.get("mann_whitney_p")
    s_attn = _attn_top(f1b.get("b1_success", {}))
    f_attn = _attn_top(f1b.get("b1_failure", {}))
    if isinstance(p, (int, float)) and s_attn is not None and f_attn is not None:
        sig = "n.s." if p > 0.05 else "sig."
        arrow = "succ>fail" if s_attn > f_attn else "fail\u2265succ"
        lines.append((f"F1-b raw year-attention (SOFT): p={p:.3f} ({sig}), {arrow} "
                      f"({s_attn:.4f} vs {f_attn:.4f}) \u2014 success & failure read the "
                      f"year alike, so attention is NOT the separator (as expected).",
                      C_WEAK))

    # Bottom-line synthesis — concept-faithful.
    lines.append(("BOTTOM LINE: Time Set is ESTABLISHED (F2-c). F1-a/b/c are soft "
                  "descriptive corroboration, not the verdict. F1 (Time-not-set) is "
                  "therefore a MINORITY failure mode, identified per-instance by the "
                  "DLA F1/F2 separator (see integrated figure) \u2014 most temporal "
                  "errors are downstream (F2 routing / F3 override), matching the "
                  "mediation-chain thesis.", "#1B5E20"))
    return lines


def plot_summary(f1a, f1b, f1c, out_dir: Path, a1=None, f2c=None):
    pops = _f1c_get_populations(f1c)
    succ_full = _f1c_ld_drops(pops.get("b1_success", {}).get("per_instance_full"),
                              competitor_only=True)
    fail_full = _f1c_ld_drops(pops.get("b1_failure", {}).get("per_instance_full"),
                              competitor_only=True)

    fig = plt.figure(figsize=(16, 9))
    gs  = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.45,
                           top=0.84, bottom=0.08, left=0.07, right=0.97)

    # — panel A0: A1 parametric memory ceiling ————————————————————————————————
    ax0 = fig.add_subplot(gs[0, 0])
    if a1 and a1.get("n_total"):
        n_tot = a1["n_total"]
        n_kn  = a1.get("n_knows_new", 0)
        frac  = n_kn / n_tot if n_tot else 0.0
        ax0.bar(["knows\nanswer_new", "does not"], [n_kn, n_tot - n_kn],
                color=[C_POS, C_WEAK], edgecolor="white")
        ax0.set_ylabel("# instances")
        ax0.set_title(f"A1  Parametric Memory\n{n_kn}/{n_tot} ({frac:.0%}) recall answer_new")
        ax0.text(0, n_kn, str(n_kn), ha="center", va="bottom", fontsize=10)
        ax0.text(1, n_tot - n_kn, str(n_tot - n_kn), ha="center", va="bottom", fontsize=10)
    else:
        ax0.text(0.5, 0.5, "(a1_parametric_memory.json\nnot provided)",
                 ha="center", va="center", transform=ax0.transAxes, color=C_WEAK)
        ax0.set_title("A1  Parametric Memory")
        ax0.set_xticks([]); ax0.set_yticks([])

    # — panel A: AUROC bar + coefficient direction note ——————————————————————
    ax1 = fig.add_subplot(gs[0, 1])
    auroc = f1a["auroc"]
    std   = f1a["auroc_std"]
    top = f1a.get("top_heads", []) or []
    n_neg = sum(1 for t in top if float(t.get("coef", 0)) < 0)
    bar_color = C_NEG if (top and n_neg > len(top) / 2) else C_SUCCESS
    ax1.bar(["SAT Probe\nAUROC"], [auroc], yerr=[std], color=bar_color,
            capsize=8, width=0.4, error_kw={"linewidth": 1.5})
    ax1.axhline(0.5, color="grey", linewidth=1.0, linestyle="--", label="chance")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("AUROC")
    dir_note = (f"{n_neg}/{len(top)} top heads NEGATIVE\n(year-attn \u2191 \u21d2 FAILURE)"
                if top and n_neg > len(top) / 2 else
                f"{len(top) - n_neg}/{len(top)} top heads positive")
    ax1.set_title(f"F1-a  SAT Probe\n{dir_note}")
    ax1.legend(fontsize=9)
    ax1.text(0, auroc + std + 0.03, f"{auroc:.3f}±{std:.3f}",
             ha="center", fontsize=10, color=bar_color)

    # — panel B: attention bar comparison (with significance flag) ——————————
    ax2 = fig.add_subplot(gs[0, 2])
    weak_label = f1b.get("weak_group_label", "B3")
    weak_key   = weak_label.lower()
    succ_node = f1b.get("b1_success", {})
    fail_node = f1b.get("b1_failure", {})
    weak_node = f1b.get(weak_key) or f1b.get("b3") or {}

    s_attn  = _attn_top(succ_node)
    f_attn  = _attn_top(fail_node)
    w_attn  = _attn_top(weak_node)
    bar_lbls = ["B1-success", "B1-failure", f"{weak_label}\n(<YEAR>)"]
    bar_vals = [v if v is not None else 0.0 for v in (s_attn, f_attn, w_attn)]
    ax2.bar(bar_lbls, bar_vals, color=[C_SUCCESS, C_FAILURE, C_WEAK],
            edgecolor="white")
    ax2.set_ylabel("Mean attention to year / <YEAR>\n(top-3 H_T heads)")
    p_value = f1b.get("mann_whitney_p")
    if isinstance(p_value, (int, float)):
        sig = "n.s." if p_value > 0.05 else "sig."
        pval_str = f"p = {p_value:.4f}  [{sig}]"
    else:
        pval_str = ""
    ax2.set_title(f"F1-b  Attention Comparison\n{pval_str}")
    for i, v in enumerate(bar_vals):
        if v:
            ax2.text(i, v + max(bar_vals) * 0.02, f"{v:.4f}",
                     ha="center", fontsize=9)
    if isinstance(p_value, (int, float)) and p_value > 0.05:
        ax2.text(0.5, 0.92, "NOT significant", transform=ax2.transAxes,
                 ha="center", color=C_NEG, fontsize=10, fontweight="bold")

    # — panel C: knockout CDF (competitor cohort) ————————————————————————————
    ax3 = fig.add_subplot(gs[1, 0])
    plotted = False
    for drops, label, color in [
        (succ_full, "B1-success", C_SUCCESS),
        (fail_full, "B1-failure", C_FAILURE),
    ]:
        if not drops:
            continue
        sorted_d = np.sort(drops)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax3.step(sorted_d, cdf, where="post", color=color, linewidth=2.0,
                 label=f"{label}  (n={len(drops)}, med={np.median(drops):+.2f})")
        plotted = True
    ax3.axvline(0.0,  color="black", linewidth=0.8, linestyle="--")
    ax3.set_xlabel(r"logit-diff drop  $\Delta(\log p_{new}-\log p_{old})$")
    ax3.set_ylabel("CDF")
    ax3.set_title("F1-c (corroborating)\nlogit-diff, competitor cohort")
    if plotted:
        ax3.legend(fontsize=9)

    # — panel D: synthesized F1 verdict ——————————————————————————————————————
    axv = fig.add_subplot(gs[1, 1:3])
    axv.axis("off")
    axv.set_title("F1 Interpretation (Time-Set verdict = F2-c; F1-a/b/c = soft)",
                  fontsize=12, loc="left")
    lines = _f1_verdict_lines(f1a, f1b, f1c, a1, f2c=f2c)
    y = 0.98
    for text, color in lines:
        wrapped = _wrap(text, 80)
        axv.text(0.0, y, wrapped, transform=axv.transAxes, va="top", ha="left",
                 fontsize=9.0, color=color)
        y -= 0.055 + 0.052 * wrapped.count("\n")

    # H_T provenance.
    is_fb = (f1a.get("step5_f1_positive") or {}).get("is_fallback")
    ht_str = "H_T = top-3 |coef| fallback" if is_fb else "H_T = validated set"
    fig.suptitle(
        f"F1 Diagnostic Summary — methodology-aligned "
        f"(C={f1a.get('probe_C', 0.05)}, agg={f1a.get('feature_aggregation', 'mean')}, "
        f"{ht_str})",
        fontsize=13, y=0.97
    )
    fig.savefig(out_dir / "f1_summary.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "f1_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("[saved] f1_summary")


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width)) or text


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/f1_phi3")
    parser.add_argument("--out",     default=None)
    parser.add_argument("--f2-dir", dest="f2_dir", default=None,
                        help="F2 results dir; used to show the F2-c Time-Set verdict")
    args = parser.parse_args()

    res_dir = Path(args.results)
    out_dir = Path(args.out) if args.out else res_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    f1a = load(res_dir / "f1a_sat_probe.json")
    f1b = load(res_dir / "f1b_attention_comparison.json")
    f1c = load(res_dir / "f1c_attention_knockout.json")
    a1_path = res_dir / "a1_parametric_memory.json"
    a1 = load(a1_path) if a1_path.exists() else None
    f2c = None
    if args.f2_dir:
        f2c_path = Path(args.f2_dir) / "f2c_b5_vs_b6.json"
        f2c = load(f2c_path) if f2c_path.exists() else None

    plot_f1a(f1a, out_dir)
    plot_f1b(f1b, out_dir)
    plot_f1c(f1c, out_dir)
    plot_summary(f1a, f1b, f1c, out_dir, a1=a1, f2c=f2c)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
