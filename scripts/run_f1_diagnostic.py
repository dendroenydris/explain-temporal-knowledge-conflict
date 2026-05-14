#!/usr/bin/env python3
"""F1 Diagnostic — "Time Not Set"

Standalone script that runs the three F1 sub-experiments from the TATM
methodology:

  F1-a  SAT Probe        — logistic regression on attention-to-year features
  F1-b  Attention Compare — B1-success vs B1-failure vs B3 at temporal heads
  F1-c  Attention Knockout — causal test: block year-token attention, measure
                             probability drop for answer_new.
                             Includes random-token control baseline:
                             the same number of random (non-year) tokens are
                             knocked out to validate that year-token knockout
                             is *specifically* causal, not a generic attention
                             disruption effect.

Usage
-----
    python scripts/run_f1_diagnostic.py \\
        --data  data/processed/wikidata_layer2.jsonl \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --out   results/f1_diagnostic/

The script auto-detects the data format (EvalInstance layer2 or raw JSONL)
and constructs B1 / B3 prompt pairs accordingly.
"""
from __future__ import annotations

import argparse
import json
import os
import random as _random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

# Ensure source/ is on PYTHONPATH
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "source"))

from tatm.model import (
    YEAR_PLACEHOLDER,
    build_prompt,
    check_match,
    find_year_placeholder_positions,
    find_year_positions,
    get_first_answer_token,
    load_model,
)
from tatm.hooks import (
    attention_knockout,
    extract_attention_to_positions,
)
from tatm.sat_probe import (
    analyse_weights,
    collect_features,
    compute_f1_positive_instances,
    fallback_temporal_heads,
    train_probe,
)


# ── Data loading ─────────────────────────────────────────────────────────────

def _is_layer2(record: dict) -> bool:
    return "instance_id" in record and "task_type" in record


def load_instances(
    path: str,
    use_b5: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Load B1/B5 and B3 instance pairs from a JSONL file.

    Args:
        path:    JSONL file path.
        use_b5:  If True, load B5 (multi-span context) instead of B1.
    Returns (b1_instances, b3_instances).
    For layer-2 format, filters by task_type.
    For raw format, constructs B1/B3 pairs from each record.
    """
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {path}")

    if _is_layer2(records[0]):
        return _load_layer2(records, use_b5=use_b5)
    return _load_raw(records)


def _load_layer2(
    records: list[dict],
    use_b5: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Layer-2 EvalInstance format: load strong/weak context pairs.

    Pairs:
      B1 (strong, single-span) + B3 (weak, years stripped from B1)  — default
      B5 (strong, multi-span)  + B6 (weak, years stripped from B5)  — --b5 flag

    Using B5/B6 is the recommended setup for the F1 (temporal attention)
    diagnostic because B5 shows both answer_old and answer_new in context,
    forcing the model to rely on the year token to choose the correct answer.
    B6 provides the year-free baseline for the same dual-span context.
    """
    strong_prefix = "B5" if use_b5 else "B1"
    weak_prefix   = "B6" if use_b5 else "B3"

    b1 = [r for r in records if r["instance_id"].startswith(strong_prefix)]
    b3 = [r for r in records if r["instance_id"].startswith(weak_prefix)]

    # align weak instances to strong by (fact_id, t_old, t_new)
    b3_map = {(r["fact_id"], r["t_old"], r["t_new"]): r for r in b3}
    b3_aligned = [
        b3_map.get((r["fact_id"], r["t_old"], r["t_new"]))
        for r in b1
    ]
    b3_aligned = [r for r in b3_aligned if r is not None]

    return b1, b3_aligned


def _load_raw(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Raw JSONL: each record has question, evidence_new, answer_new, etc.
    Construct B1 and B3 from each."""
    import re

    b1_list, b3_list = [], []
    for r in records:
        evidence = r.get("evidence_new", "")
        question = r.get("question", "")
        t_new = r.get("t_new")

        # parse t_new from ISO date if needed
        if isinstance(t_new, str) and "T" in t_new:
            t_new = int(t_new[:4])

        if not question:
            subj = r.get("subject_label", r.get("subject", ""))
            prop = r.get("property_label", r.get("property", ""))
            question = f"As of {t_new}, what is the {prop} of {subj}?"

        b1 = {**r, "t_new": t_new, "question": question, "evidence_new": evidence,
              "context": evidence}
        b1_list.append(b1)

        # B3: position-preserving <YEAR> placeholder substitution (methodology
        # Step 2(c)).  ``evidence_new`` is kept unchanged for diagnostics;
        # ``context`` carries the stripped passage that the model actually sees.
        weak_evidence = re.sub(r"(?<!\d)(?:19|20)\d{2}(?!\d)", YEAR_PLACEHOLDER, evidence)
        b3 = {**b1, "context": weak_evidence,
              "instance_id": f"B3_{r.get('id', '')}"}
        b3_list.append(b3)

    return b1_list, b3_list


def restrict_instances(
    strong_instances: list[dict],
    weak_instances: list[dict],
    *,
    number: int | None = None,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Randomly sample Layer-1 facts and keep all aligned Layer-2 pairs."""
    if number is None:
        return strong_instances, weak_instances
    if number < 1:
        raise ValueError("--number must be >= 1")

    n_pairs = min(len(strong_instances), len(weak_instances))
    strong_instances = strong_instances[:n_pairs]
    weak_instances = weak_instances[:n_pairs]

    fact_ids = sorted({
        str(inst.get("fact_id") or inst.get("id") or idx)
        for idx, inst in enumerate(strong_instances)
    })
    if number >= len(fact_ids):
        return strong_instances[:n_pairs], weak_instances[:n_pairs]

    rng = _random.Random(seed)
    selected_fact_ids = set(rng.sample(fact_ids, number))
    indices = [
        idx
        for idx, inst in enumerate(strong_instances)
        if str(inst.get("fact_id") or inst.get("id") or idx) in selected_fact_ids
    ]
    return (
        [strong_instances[i] for i in indices],
        [weak_instances[i] for i in indices],
    )


def load_layer3_answers(path: str) -> dict[str, dict]:
    """Load Layer-3 parametric answers keyed by Layer-2 instance_id."""
    layer3_path = Path(path)
    if not layer3_path.exists():
        raise FileNotFoundError(
            f"Layer-3 parametric answer file not found: {layer3_path}. "
            "Build it first with build_wikidata_layer3_1000.sh."
        )

    answers: dict[str, dict] = {}
    with open(layer3_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iid = row.get("instance_id")
            if iid:
                answers[str(iid)] = row
    if not answers:
        raise ValueError(f"No Layer-3 answers found in {layer3_path}")
    return answers


# ── A1: Parametric memory filter ─────────────────────────────────────────────

def run_a1_filter(
    b1_instances: list[dict],
    b3_instances: list[dict],
    out_dir: Path,
    layer3_answers: dict[str, dict],
    strong_label: str = "B1",
    weak_label: str = "B3",
) -> tuple[list[dict], list[dict]]:
    """Use cached Layer-3 answers to keep instances the model cannot answer
    from parametric memory alone.

    Filtering criterion
    -------------------
    L3_KNOWS_NEW  : Layer-3 answer matches answer_new  → exclude
    L3_WRONG      : Layer-3 answer is other             → keep
    """
    print("\n" + "=" * 60)
    print("Layer-3: Year-conditioned Parametric Memory Profiling")
    print("=" * 60)
    print("Source: cached Layer-3 answers, question WITH year, WITHOUT context")
    print("Keeping instances where model does NOT output answer_new")
    print(f"(i.e., evidence passage is genuinely needed for {strong_label} to succeed)\n")

    knows_new_ids: set[str] = set()
    a1_log = []
    missing_ids: list[str] = []

    bar = tqdm(b1_instances, desc="A1  parametric", unit="inst", dynamic_ncols=True)
    for inst in bar:
        question   = inst.get("question", "")
        answer_new = inst.get("answer_new", "")
        answer_old = inst.get("answer_old", "")
        iid        = inst.get("instance_id", "")

        layer3_row = layer3_answers.get(iid)
        if layer3_row is None:
            missing_ids.append(iid)
            continue

        generated = layer3_row.get("extracted_answer") or layer3_row.get("model_output_raw", "")
        already_knows = bool(
            layer3_row.get("matches_answer_new")
            if "matches_answer_new" in layer3_row
            else check_match(generated, answer_new)
        )

        a1_log.append({
            "instance_id": iid,
            "question":    question,
            "generated":   generated,
            "model_output_raw": layer3_row.get("model_output_raw", ""),
            "answer_old":  answer_old,
            "answer_new":  answer_new,
            "knows_new":   already_knows,
            "layer3_id":   layer3_row.get("layer3_id", ""),
        })

        if already_knows:
            knows_new_ids.add(iid)

        bar.set_postfix(
            knows_new=len(knows_new_ids),
            kept=len(b1_instances) - len(knows_new_ids),
            refresh=True,
        )

    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"Layer-3 file is missing {len(missing_ids)} {strong_label} parametric "
            f"answers required by A1 filter. Examples: {preview}. "
            f"Rebuild Layer-3 with LAYERS={strong_label}."
        )

    n_total   = len(b1_instances)
    n_known   = len(knows_new_ids)
    n_kept    = n_total - n_known
    print(f"\n  A1_KNOWS_NEW (excluded): {n_known}/{n_total} ({100*n_known/n_total:.1f}%)")
    print(f"  A1_WRONG     (kept):     {n_kept}/{n_total} ({100*n_kept/n_total:.1f}%)")

    # save A1 log
    with open(out_dir / "a1_parametric_memory.json", "w") as f:
        json.dump({"n_total": n_total, "n_knows_new": n_known, "log": a1_log},
                  f, indent=2, ensure_ascii=False)

    # filter both B1 and B3 to matching subset
    b1_filtered = [r for r in b1_instances if r.get("instance_id") not in knows_new_ids]

    # B3 has different instance_id hashes; match via (fact_id, t_old, t_new)
    excluded_keys = set()
    for r in b1_instances:
        if r.get("instance_id") in knows_new_ids:
            excluded_keys.add((r.get("fact_id"), r.get("t_old"), r.get("t_new")))
    b3_filtered = [
        r for r in b3_instances
        if (r.get("fact_id"), r.get("t_old"), r.get("t_new")) not in excluded_keys
    ]

    print(f"\n→ {len(b1_filtered)} {strong_label} + {len(b3_filtered)} {weak_label} instances kept for F1 diagnosis")
    return b1_filtered, b3_filtered


# ── F1-a: SAT Probe ─────────────────────────────────────────────────────────

def run_f1a(model, b1_instances, template, out_dir, *, probe_c: float = 0.05):
    """SAT Probe: logistic regression on attention-to-year features.

    After the probe (Steps 3–4), Step 5 computes the per-instance
    H_T-attention scalar and the F1-positive verdicts at the {20, 25, 33}
    percentile thresholds.  Under the temporal-head fallback (the default
    here — no DYNAMICQA temporal-head validation has been run), H_T is the
    top-3 heads by ``|probe coefficient|`` (Step 5, final sentence).
    """
    print("\n" + "=" * 60)
    print("F1-a: SAT Probe")
    print("=" * 60)

    X, y, meta = collect_features(
        model, b1_instances, template=template, verbose=True,
    )

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"\nOverride success: {n_pos}/{len(y)} ({100*n_pos/len(y):.1f}%)")
    print(f"Override failure: {n_neg}/{len(y)} ({100*n_neg/len(y):.1f}%)")

    probe_result = train_probe(X, y, C=probe_c)
    print(f"\nSAT Probe AUROC: {probe_result.auroc:.3f} ± {probe_result.auroc_std:.3f}")

    top_heads = analyse_weights(
        probe_result, model.cfg.n_layers, model.cfg.n_heads, top_k=10,
    )
    n_nonzero = sum(1 for _, _, c in top_heads if c != 0.0)
    print(f"\nTop-10 (layer, head) by |coefficient|  [{n_nonzero} non-zero]:")
    for layer, head, coef in top_heads:
        direction = "↑ success" if coef > 0 else ("↓ success" if coef < 0 else "zero")
        print(f"  L{layer:2d}.H{head:2d}  coef={coef:+.4e}  ({direction})")

    if n_nonzero == 0:
        print("\n  WARNING: all coefficients are zero.")
        print("  Possible causes:")
        print(f"    - C={probe_c} too restrictive for this sample size "
              "(try --probe-c 10.0 on pilot splits)")
        print("    - All attention features nearly identical across instances")
        print("    → F1 (attention to year tokens) may genuinely not be the mechanism")

    # ── Step 5: per-instance H_T-attention scalar + percentile sweep ─────────
    ht_heads = fallback_temporal_heads(top_heads, top_k=3)
    f1_pos = compute_f1_positive_instances(
        X, y, ht_heads,
        n_heads=model.cfg.n_heads,
        percentiles=(20, 25, 33),
        primary_percentile=25,
        is_fallback=True,
    )
    primary_mask = f1_pos.f1_positive_by_percentile[f1_pos.primary_percentile]
    print(
        f"\nF1-a Step 5  (temporal-head FALLBACK — top-3 by |coef|): "
        f"{ht_heads}"
    )
    print(
        f"  H_T-attention scalar (mean over heads × constraint tokens): "
        f"primary threshold = {f1_pos.threshold_by_percentile[f1_pos.primary_percentile]:.4e} "
        f"(B1-success p{f1_pos.primary_percentile})"
    )
    for p in (20, 25, 33):
        n_pos_p = sum(primary_mask) if p == f1_pos.primary_percentile else sum(
            f1_pos.f1_positive_by_percentile[p]
        )
        print(
            f"  p{p:>2d} → threshold={f1_pos.threshold_by_percentile[p]:.4e}  "
            f"F1-positive: {n_pos_p}/{len(y)} ({100*n_pos_p/len(y):.1f}%)"
        )

    # save — use scientific notation strings for coef so tiny values are visible
    probe_out = {
        "auroc": probe_result.auroc,
        "auroc_std": probe_result.auroc_std,
        "n_samples": probe_result.n_samples,
        "n_positive": probe_result.n_positive,
        "n_negative": probe_result.n_samples - probe_result.n_positive,
        "probe_C": probe_c,
        "feature_aggregation": "mean",   # methodology F1-a Step 5
        "top_heads": [
            {"layer": l, "head": h, "coef": f"{c:.4e}"} for l, h, c in top_heads
        ],
        "step5_f1_positive": {
            "ht_heads": [{"layer": l, "head": h} for (l, h) in f1_pos.ht_heads],
            "is_fallback": f1_pos.is_fallback,
            "primary_percentile": f1_pos.primary_percentile,
            "threshold_by_percentile": f1_pos.threshold_by_percentile,
            "scalar_per_instance": f1_pos.scalar_per_instance,
            "f1_positive_by_percentile": {
                str(p): mask for p, mask in f1_pos.f1_positive_by_percentile.items()
            },
        },
        "instance_meta": meta,
    }
    with open(out_dir / "f1a_sat_probe.json", "w") as f:
        json.dump(probe_out, f, indent=2, ensure_ascii=False)

    return probe_result, X, y, meta, f1_pos


# ── F1-b: Attention comparison ───────────────────────────────────────────────

def _get_prompt_context(inst: dict) -> str:
    """Return the context string the model actually receives.

    ``context`` carries the post-``strip_years`` (B3 / B6) or
    pre-``strip_years`` (B1 / B5) passage as appropriate.  ``evidence_new``
    is the unstripped Wikipedia evidence used as a diagnostic anchor only —
    using it for B3 would silently treat B3 like B1 and destroy the F1-b
    contrast.
    """
    return inst.get("context") or inst.get("evidence_new") or ""


def _year_or_placeholder_positions(
    inst: dict,
    tokens: torch.Tensor,
    tokenizer,
    *,
    is_year_stripped: bool,
) -> tuple[list[int], str]:
    """Return ((src positions), source-label) for F1-b attention measurement.

    For strong-context groups (B1 / B5): the BPE tokens of the literal
    4-digit year (preferring ``t_new``; fall back to any year if missing).
    For weak-context groups (B3 / B6): the BPE tokens of the ``<YEAR>``
    placeholder inserted by ``strip_years`` — methodology Step 2(d) — so
    the measurement targets the same residual-stream slot the year
    occupied in the strong condition.
    """
    if is_year_stripped:
        positions = find_year_placeholder_positions(tokens[0], tokenizer)
        if positions:
            return positions, "placeholder"
        # Pathological fallback (placeholder somehow missing from tokenisation)
        # — use any remaining year tokens in the prompt (typically from the
        # question), explicitly flagged.
        positions = find_year_positions(tokens[0], tokenizer)
        return positions, "fallback_year"

    t_new = inst.get("t_new")
    positions = find_year_positions(tokens[0], tokenizer, target_year=t_new)
    if positions:
        return positions, "year_target"
    positions = find_year_positions(tokens[0], tokenizer)
    return positions, "year_any"


def run_f1b(
    model,
    b1_instances,
    b3_instances,
    y_labels,
    top_heads,
    template,
    out_dir,
    *,
    weak_label: str = "B3",
):
    """Compare attention to year tokens: B1-success vs B1-failure vs B3/B6.

    Per methodology F1-b:
      - B1-success / B1-failure: attention at H_T to the literal year BPE
        tokens.
      - B3 (or B6): attention at H_T to the ``<YEAR>`` placeholder BPE
        tokens at the *same* residual-stream position.
      Under the temporal-head fallback, H_T = top-3 heads by ``|probe coef|``.
    """
    print("\n" + "=" * 60)
    print(f"F1-b: Attention Comparison (B1-success vs B1-failure vs {weak_label})")
    print("=" * 60)

    if not top_heads:
        print("No top heads from SAT probe. Using all heads.")
        head_set: Optional[list[tuple[int, int]]] = None
    else:
        head_set = fallback_temporal_heads(top_heads, top_k=3)
        print(f"Analysing top-3 heads (temporal-head fallback): {head_set}")

    weak_key = weak_label.lower()
    groups: dict[str, list[np.ndarray]] = {"b1_success": [], "b1_failure": [], weak_key: []}
    src_label_counts: dict[str, Counter] = {
        "b1_success": Counter(), "b1_failure": Counter(), weak_key: Counter()
    }

    for idx, inst in enumerate(tqdm(b1_instances, desc=f"F1-b  B1 attn", unit="inst", dynamic_ncols=True)):
        context = _get_prompt_context(inst)
        question = inst.get("question", "")

        prompt = build_prompt(context, question, template=template)
        tokens = model.to_tokens(prompt, prepend_bos=False)
        src_pos, src_label = _year_or_placeholder_positions(
            inst, tokens, model.tokenizer, is_year_stripped=False,
        )
        if not src_pos:
            continue
        attn = extract_attention_to_positions(model, tokens, src_pos, agg="mean")

        label = y_labels[idx] if idx < len(y_labels) else 0
        key = "b1_success" if label == 1 else "b1_failure"
        groups[key].append(attn.numpy())
        src_label_counts[key][src_label] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for inst in tqdm(b3_instances, desc=f"F1-b  {weak_label} attn", unit="inst", dynamic_ncols=True):
        context = _get_prompt_context(inst)
        question = inst.get("question", "")

        prompt = build_prompt(context, question, template=template)
        tokens = model.to_tokens(prompt, prepend_bos=False)
        src_pos, src_label = _year_or_placeholder_positions(
            inst, tokens, model.tokenizer, is_year_stripped=True,
        )
        if not src_pos:
            continue
        attn = extract_attention_to_positions(model, tokens, src_pos, agg="mean")
        groups[weak_key].append(attn.numpy())
        src_label_counts[weak_key][src_label] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # aggregate per group
    results: dict = {
        "weak_group_label": weak_label,
        "src_position_provenance": {
            g: dict(c) for g, c in src_label_counts.items()
        },
    }
    for group_name, attn_list in groups.items():
        if not attn_list:
            print(f"  {group_name}: no instances")
            continue
        stacked = np.stack(attn_list)  # [N, L, H]
        mean_attn = stacked.mean(axis=0)  # [L, H]
        results[group_name] = {
            "count": len(attn_list),
            "mean_attn_all": mean_attn.tolist(),
        }

        if head_set:
            head_vals = [mean_attn[l, h] for l, h in head_set]
            mean_val = np.mean(head_vals)
            print(f"  {group_name} (n={len(attn_list)}): "
                  f"mean attn at top-3 heads = {mean_val:.4f}")
            results[group_name]["mean_attn_top3"] = float(mean_val)
            results[group_name]["per_head"] = [
                {"layer": l, "head": h, "attn": float(mean_attn[l, h])}
                for l, h in head_set
            ]

    # ``mean_attn_top5`` is kept as an alias of ``mean_attn_top3`` so the
    # downstream plot script (which used top-5 before the methodology
    # alignment) keeps reading a valid key without modification.
    for g in (("b1_success", "b1_failure", weak_key)):
        if g in results and "mean_attn_top3" in results[g]:
            results[g]["mean_attn_top5"] = results[g]["mean_attn_top3"]

    # statistical comparison
    if groups["b1_success"] and groups["b1_failure"] and head_set:
        succ_vals = np.array([
            np.mean([a[l, h] for l, h in head_set])
            for a in groups["b1_success"]
        ])
        fail_vals = np.array([
            np.mean([a[l, h] for l, h in head_set])
            for a in groups["b1_failure"]
        ])
        from scipy.stats import mannwhitneyu
        try:
            stat, pval = mannwhitneyu(succ_vals, fail_vals, alternative="greater")
            results["mann_whitney_U"] = float(stat)
            results["mann_whitney_p"] = float(pval)
            print(f"\n  Mann-Whitney U (success > failure): U={stat:.1f}, p={pval:.4f}")
        except ValueError:
            pass

    # Backwards compat: tools still expecting the literal "b3" key get a
    # pointer to whichever weak group was actually measured.
    if weak_key != "b3" and weak_key in results:
        results["b3"] = results[weak_key]

    with open(out_dir / "f1b_attention_comparison.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ── F1-c helpers ─────────────────────────────────────────────────────────────

def _sample_random_positions(
    seq_len: int,
    exclude: set[int],
    n: int,
    n_samples: int = 3,
    seed: int = 42,
) -> list[list[int]]:
    """Sample *n_samples* sets of *n* random token positions.

    Positions in *exclude* (year positions + last/prediction position) are
    never selected.  If there aren't enough remaining positions to fill a
    full set of *n*, we return as many as possible for that sample.

    Parameters
    ----------
    seq_len   : total number of tokens in the prompt
    exclude   : set of positions to never sample (year tokens + last token)
    n         : number of positions per sample (= number of year tokens)
    n_samples : how many independent random samples to draw
    seed      : for reproducibility

    Returns
    -------
    List of *n_samples* position-lists.  Each inner list has exactly *n*
    elements (or fewer if not enough candidates are available).
    """
    candidates = [p for p in range(seq_len) if p not in exclude]
    rng = _random.Random(seed)
    samples = []
    for _ in range(n_samples):
        chosen = rng.sample(candidates, min(n, len(candidates)))
        samples.append(sorted(chosen))
    return samples


# ── F1-c: Attention knockout ─────────────────────────────────────────────────

def _f1c_run_one(
    model,
    inst: dict,
    template: str,
    ko_layers: list[int],
    n_random_samples: int,
    instance_seed: int,
) -> Optional[dict]:
    """Single-instance year-token knockout + matched random-token control.

    Returns ``None`` if the year/placeholder cannot be located in the prompt.
    """
    context    = _get_prompt_context(inst)
    question   = inst.get("question", "")
    answer_new = inst.get("answer_new", "")
    answer_old = inst.get("answer_old", "")
    t_new      = inst.get("t_new")

    prompt  = build_prompt(context, question, template=template)
    tokens  = model.to_tokens(prompt, prepend_bos=False)
    seq_len = tokens.shape[-1]

    year_pos = find_year_positions(tokens[0], model.tokenizer, target_year=t_new)
    if not year_pos:
        year_pos = find_year_positions(tokens[0], model.tokenizer)
    if not year_pos:
        return None

    new_tid    = get_first_answer_token(model, answer_new)
    old_tid    = get_first_answer_token(model, answer_old)
    track_tids = list({t for t in [new_tid, old_tid] if t >= 0})

    ko = attention_knockout(
        model, tokens, year_pos,
        knockout_layers=ko_layers,
        answer_token_ids=track_tids if track_tids else None,
    )

    entry: dict = {
        "instance_id": inst.get("instance_id", ""),
        "answer_new": answer_new,
        "answer_old": answer_old,
        "n_year_tokens_blocked": len(year_pos),
        "year_positions": year_pos,
        "knockout_layers": f"{ko_layers[0]}-{ko_layers[-1]}",
    }

    if new_tid >= 0 and new_tid in ko["probs_clean"]:
        p_clean = ko["probs_clean"][new_tid]
        p_ko    = ko["probs_ko"][new_tid]
        entry["p_new_clean"]         = p_clean
        entry["p_new_knockout"]      = p_ko
        entry["p_new_drop"]          = p_clean - p_ko
        entry["p_new_drop_relative"] = (p_clean - p_ko) / max(p_clean, 1e-12)

    if old_tid >= 0 and old_tid in ko["probs_clean"]:
        p_clean = ko["probs_clean"][old_tid]
        p_ko    = ko["probs_ko"][old_tid]
        entry["p_old_clean"]   = p_clean
        entry["p_old_knockout"] = p_ko
        entry["p_old_gain"]    = p_ko - p_clean

    # ── Random-token control ────────────────────────────────────────────────
    exclude = set(year_pos) | {seq_len - 1}
    rand_position_sets = _sample_random_positions(
        seq_len, exclude, n=len(year_pos),
        n_samples=n_random_samples,
        seed=instance_seed,
    )

    rand_drops: list[float] = []
    rand_details: list[dict] = []
    for sample_i, rand_pos in enumerate(rand_position_sets):
        if not rand_pos:
            continue
        ko_rand = attention_knockout(
            model, tokens, rand_pos,
            knockout_layers=ko_layers,
            answer_token_ids=track_tids if track_tids else None,
        )
        if new_tid >= 0 and new_tid in ko_rand["probs_clean"]:
            p_c  = ko_rand["probs_clean"][new_tid]
            p_kr = ko_rand["probs_ko"][new_tid]
            rel_drop = (p_c - p_kr) / max(p_c, 1e-12)
            rand_drops.append(rel_drop)
            rand_details.append({
                "sample": sample_i,
                "rand_positions": rand_pos,
                "p_new_drop_relative": rel_drop,
            })

    if rand_drops:
        entry["rand_control"] = {
            "n_samples": len(rand_drops),
            "mean_drop_relative": float(np.mean(rand_drops)),
            "max_drop_relative": float(np.max(rand_drops)),
            "details": rand_details,
        }
        # Specificity ratio: year drop / mean random drop.
        year_drop = entry.get("p_new_drop_relative", 0.0)
        mean_rand = entry["rand_control"]["mean_drop_relative"]
        entry["specificity_ratio"] = year_drop / max(abs(mean_rand), 1e-6)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return entry


def _summarize_drops(label: str, results: list[dict]) -> dict:
    drops = [r["p_new_drop_relative"] for r in results if "p_new_drop_relative" in r]
    if not drops:
        return {}
    mean_drop   = float(np.mean(drops))
    median_drop = float(np.median(drops))
    pct_10      = sum(1 for d in drops if d > 0.10) / len(drops)
    print(f"\n  [{label}]  n={len(drops)}")
    print(f"    Mean   p(answer_new) drop: {mean_drop:+.4f}")
    print(f"    Median p(answer_new) drop: {median_drop:+.4f}")
    print(f"    Fraction >10% drop:        {pct_10:.2%}")

    rand_mean_drops = [
        r["rand_control"]["mean_drop_relative"]
        for r in results if "rand_control" in r
    ]
    rand_mean = rand_p10 = mean_spec = None
    if rand_mean_drops:
        rand_mean = float(np.mean(rand_mean_drops))
        rand_p10  = sum(1 for d in rand_mean_drops if d > 0.10) / len(rand_mean_drops)
        spec_ratios = [r["specificity_ratio"] for r in results if "specificity_ratio" in r]
        mean_spec = float(np.mean(spec_ratios)) if spec_ratios else float("nan")
        print(f"    [control] Mean random-token drop:   {rand_mean:+.4f}")
        print(f"    [control] Fraction random >10% drop:{rand_p10:.2%}")
        print(f"    Specificity ratio (year/random):    {mean_spec:.2f}x")
        if mean_spec is not None and mean_spec > 2.0:
            print(f"    → year-token knockout is SPECIFIC (ratio > 2)")
        elif mean_spec is not None and mean_spec > 1.0:
            print(f"    → year-token knockout is moderately specific (ratio > 1)")
        else:
            print(f"    → year-token effect not clearly specific vs random")
    return {
        "mean_drop": mean_drop,
        "median_drop": median_drop,
        "pct_above_10": pct_10,
        "rand_mean_drop": rand_mean,
        "rand_pct_above_10": rand_p10,
        "mean_specificity_ratio": mean_spec,
    }


def run_f1c(model, b1_instances, y_labels, top_heads, template, out_dir):
    """Methodology F1-c — Attention Knockout with random-token control.

    Per methodology (lines 248–271): block attention from the prediction
    position to year-token positions across **all layers** (full-network
    knockout) and run on **both B1-success and B1-failure** instances; for
    each instance also run ``k = 3`` random-position controls and report
    the specificity ratio.

    The (descriptive) deep-only L24–L31 window is retained as a secondary
    locality check, not as the primary methodology output.
    """
    N_RANDOM_SAMPLES = 3   # methodology: k = 3

    print("\n" + "=" * 60)
    print("F1-c: Attention Knockout  (B1-success + B1-failure + random control)")
    print("=" * 60)

    n_layers = model.cfg.n_layers
    ko_layers_full = list(range(n_layers))                     # methodology: all layers
    ko_layers_deep = list(range(n_layers * 3 // 4, n_layers))  # supplementary

    print(f"Knockout layer window (full):  0–{ko_layers_full[-1]}  (methodology primary)")
    print(f"Knockout layer window (deep):  {ko_layers_deep[0]}–{ko_layers_deep[-1]}  (supplementary)")
    print(f"Random-token control samples per instance: {N_RANDOM_SAMPLES}")

    populations: dict[str, dict] = {}
    for pop_label, target_y in (("b1_success", 1), ("b1_failure", 0)):
        indices = [i for i, y in enumerate(y_labels) if y == target_y]
        if not indices:
            print(f"\n[{pop_label}] no instances, skipping.")
            populations[pop_label] = {
                "n_instances": 0,
                "full_stats": {}, "deep_stats": {},
                "per_instance_full": [], "per_instance_deep": [],
            }
            continue

        print(f"\n[{pop_label}]  {len(indices)} instances")

        # ── Full-network knockout (methodology primary) ─────────────────────
        full_results: list[dict] = []
        bar = tqdm(indices, desc=f"F1-c {pop_label}  full L0–{n_layers-1}",
                   unit="inst", dynamic_ncols=True)
        for idx in bar:
            inst = b1_instances[idx]
            entry = _f1c_run_one(
                model, inst, template,
                ko_layers=ko_layers_full,
                n_random_samples=N_RANDOM_SAMPLES,
                instance_seed=idx,
            )
            if entry is None:
                bar.set_postfix_str("skip (no year toks)", refresh=True)
                continue
            full_results.append(entry)
            drop_pct = entry.get("p_new_drop_relative", float("nan"))
            spec     = entry.get("specificity_ratio", float("nan"))
            bar.set_postfix(
                drop=f"{drop_pct:+.2f}" if drop_pct == drop_pct else "n/a",
                spec=f"{spec:.1f}x"  if spec == spec else "n/a",
                refresh=True,
            )

        # ── Deep-layer knockout (supplementary) ─────────────────────────────
        deep_results: list[dict] = []
        deep_bar = tqdm(indices, desc=f"F1-c {pop_label}  deep L{ko_layers_deep[0]}–{ko_layers_deep[-1]}",
                        unit="inst", dynamic_ncols=True)
        for idx in deep_bar:
            inst = b1_instances[idx]
            entry = _f1c_run_one(
                model, inst, template,
                ko_layers=ko_layers_deep,
                n_random_samples=N_RANDOM_SAMPLES,
                instance_seed=idx,
            )
            if entry is None:
                continue
            deep_results.append(entry)

        full_stats = _summarize_drops(f"{pop_label}  full L0–L{n_layers-1}", full_results)
        deep_stats = _summarize_drops(
            f"{pop_label}  deep L{ko_layers_deep[0]}–L{ko_layers_deep[-1]}",
            deep_results,
        )
        populations[pop_label] = {
            "n_instances": len(full_results),
            "full_stats": full_stats,
            "deep_stats": deep_stats,
            "per_instance_full": full_results,
            "per_instance_deep": deep_results,
        }

    summary: dict = {
        "knockout_layers_full": ko_layers_full,
        "knockout_layers_deep": ko_layers_deep,
        "n_random_control_samples": N_RANDOM_SAMPLES,
        "populations": populations,
    }
    # Backwards compat: many downstream plotting helpers (and the current
    # plot_f1_results.py) read ``per_instance_full`` / ``per_instance_deep``
    # at the top level expecting B1-success.  Expose them as aliases.
    if populations.get("b1_success", {}).get("n_instances"):
        succ = populations["b1_success"]
        summary["per_instance_full"] = succ["per_instance_full"]
        summary["per_instance_deep"] = succ["per_instance_deep"]
        summary["full_stats"] = succ["full_stats"]
        summary["deep_stats"] = succ["deep_stats"]
        summary["n_instances"] = succ["n_instances"]

    with open(out_dir / "f1c_attention_knockout.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="F1 Diagnostic — Time Not Set")
    parser.add_argument("--data", required=True, help="Path to JSONL data file")
    parser.add_argument(
        "--model", default="meta-llama/Llama-2-7b-chat-hf",
        help="HuggingFace model name (must be supported by TransformerLens)",
    )
    parser.add_argument("--template", default="plain", choices=["plain", "llama2", "llama3", "phi3"])
    parser.add_argument("--out", default="results/f1_diagnostic", help="Output directory")
    parser.add_argument(
        "--layer3",
        help=(
            "Layer-3 JSONL with cached parametric answers. Required unless "
            "--no-a1-filter is used."
        ),
    )
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"],
        help="auto: float32 on MPS/CPU, float16 on CUDA",
    )
    parser.add_argument("--max-instances", type=int, default=None, help="Limit instances for quick testing")
    parser.add_argument(
        "-n", "--number", type=int, default=None,
        help="Randomly sample N Layer-1 facts and keep all aligned strong/weak Layer-2 pairs",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=42,
        help="Random seed used with --number (default: 42)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=["f1a", "f1b", "f1c"],
        help="Skip specific sub-experiments",
    )
    parser.add_argument(
        "--no-a1-filter", action="store_true",
        help=(
            "Skip the A1 parametric memory profiling step (NOT recommended). "
            "By default the script runs A1 (year cue, no context) first and "
            "excludes instances where the model already knows answer_new with "
            "just the year cue.  Use this flag only for debugging."
        ),
    )
    parser.add_argument(
        "--probe-c", type=float, default=0.05,
        help=(
            "Inverse L1 regularisation strength for the SAT Probe "
            "(methodology F1-a Step 3 default: 0.05).  On small pilot "
            "datasets (~40 instances) the L1 penalty can zero every "
            "coefficient at C=0.05; increase to C=10.0 in that case."
        ),
    )
    parser.add_argument(
        "--b5", action="store_true",
        help=(
            "Use B5 (multi-span context: evidence_old + evidence_new) instead "
            "of B1 (single evidence_new) as the strong-context group.  B5 "
            "forces the model to read the year cue to disambiguate, making it "
            "the correct testbed for the F1 (Time Not Set) diagnostic.  Requires "
            "the JSONL file to contain B5 instances (build with --layers B1 B3 B5)."
        ),
    )
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    if args.dtype == "auto":
        # MPS/CPU: float32; CUDA: float16
        _dev = args.device
        if _dev == "auto":
            _dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        resolved_dtype = torch.float16 if _dev == "cuda" else torch.float32
    else:
        resolved_dtype = dtype_map[args.dtype]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:    {args.model}")
    print(f"Data:     {args.data}")
    print(f"Layer3:   {args.layer3 or '(not provided)'}")
    print(f"Template: {args.template}")
    print(f"Device:   {args.device}")
    print(f"Output:   {out_dir}")

    # load data
    b1_instances, b3_instances = load_instances(args.data, use_b5=args.b5)
    if args.max_instances:
        b1_instances = b1_instances[:args.max_instances]
        b3_instances = b3_instances[:args.max_instances]
    if args.number:
        b1_instances, b3_instances = restrict_instances(
            b1_instances,
            b3_instances,
            number=args.number,
            seed=args.sample_seed,
        )
    strong_label = "B5" if args.b5 else "B1"
    weak_label   = "B6" if args.b5 else "B3"
    print(f"\nLoaded {len(b1_instances)} {strong_label} instances, {len(b3_instances)} {weak_label} instances")

    layer3_answers: dict[str, dict] = {}
    if not args.no_a1_filter:
        if not args.layer3:
            raise SystemExit(
                "[ERROR] --layer3 is required for the A1 parametric filter. "
                "Build it first with build_wikidata_layer3_1000.sh, or use "
                "--no-a1-filter for debugging."
            )
        layer3_answers = load_layer3_answers(args.layer3)
        print(f"Loaded {len(layer3_answers)} cached Layer-3 parametric answers")

    # load model
    print(f"\nLoading model {args.model}  (this may take 1–3 min on first run)…")
    with tqdm(total=1, desc="Loading weights", unit="model",
              bar_format="{desc}: {elapsed} elapsed {postfix}") as pbar:
        model = load_model(args.model, device=args.device, dtype=resolved_dtype)
        pbar.set_postfix_str("done")
        pbar.update(1)
    print(f"  {model.cfg.n_layers} layers × {model.cfg.n_heads} heads  "
          f"d_model={model.cfg.d_model}")

    # A1 filter: exclude instances where year cue alone is enough (default ON)
    if not args.no_a1_filter:
        b1_instances, b3_instances = run_a1_filter(
            b1_instances, b3_instances, out_dir, layer3_answers,
            strong_label=strong_label, weak_label=weak_label,
        )
        if not b1_instances:
            print("\nNo instances remain after A1 filter. Exiting.")
            return

    # F1-a
    probe_result, X, y, meta, f1_pos = None, None, None, None, None
    if "f1a" not in args.skip:
        probe_result, X, y, meta, f1_pos = run_f1a(
            model, b1_instances, args.template, out_dir,
            probe_c=args.probe_c,
        )

    top_heads_list = probe_result.top_heads if probe_result else []

    # F1-b
    if "f1b" not in args.skip:
        if y is None:
            # need labels — run quick feature collection
            X, y, meta = collect_features(model, b1_instances, template=args.template)
        run_f1b(
            model, b1_instances, b3_instances, y, top_heads_list,
            args.template, out_dir, weak_label=weak_label,
        )

    # F1-c
    if "f1c" not in args.skip:
        if y is None:
            X, y, meta = collect_features(model, b1_instances, template=args.template)
        run_f1c(model, b1_instances, y, top_heads_list, args.template, out_dir)

    print("\n" + "=" * 60)
    print("F1 Diagnostic complete. Results saved to:", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
