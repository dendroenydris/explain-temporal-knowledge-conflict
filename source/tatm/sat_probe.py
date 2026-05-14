"""SAT Probe for F1 diagnosis — "Time Not Set".

Adapts the SAT Probe methodology (Yuksekgonul et al., ICLR 2024) to
temporal knowledge conflicts:

  For every B1 instance, extract the attention weight from each (layer, head)
  to the year constraint tokens at the prediction position.  Train an
  L1-regularised logistic regression to predict override success/failure.
  High-weight heads that overlap with independently identified temporal heads
  confirm that F1 failures (under-attention to year tokens) cause override
  failure.

Pipeline
--------
1. collect_features  — run model, return X [N, L*H] and y [N]
2. train_probe       — fit logistic regression, return metrics
3. analyse_weights   — rank (layer, head) pairs by probe coefficient
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import gc

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformer_lens import HookedTransformer

from tatm.hooks import extract_attention_to_positions
from tatm.model import (
    build_prompt,
    check_match,
    find_year_positions,
    generate_answer,
)


@dataclass
class ProbeResult:
    """Container for SAT Probe training results."""
    auroc: float
    auroc_std: float
    coef: np.ndarray          # [L*H] — probe coefficients
    n_samples: int
    n_positive: int
    top_heads: list[tuple[int, int, float]] = field(default_factory=list)


# ── Feature collection ───────────────────────────────────────────────────────

def collect_features(
    model: HookedTransformer,
    instances: list[dict],
    *,
    template: str = "plain",
    max_new_tokens: int = 32,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run model on B1 instances; return attention features and labels.

    Parameters
    ----------
    model : HookedTransformer
    instances : list of dicts with keys
        {question, evidence_new, answer_new, answer_old, t_new, ...}
    template : prompt template name (see model.build_prompt)
    max_new_tokens : max generation length

    Returns
    -------
    X : ndarray [N, L*H]  — flattened attention features
    y : ndarray [N]        — 1 if override success, 0 if failure
    meta : list of per-instance metadata dicts
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    feat_dim = n_layers * n_heads

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta: list[dict] = []

    bar = tqdm(
        instances,
        desc="F1-a  generate+attn",
        unit="inst",
        dynamic_ncols=True,
    )

    for idx, inst in enumerate(bar):
        context = inst.get("evidence_new", inst.get("context", ""))
        question = inst.get("question", "")
        answer_new = inst.get("answer_new", "")
        t_new = inst.get("t_new")

        prompt = build_prompt(context, question, template=template)
        tokens = model.to_tokens(prompt, prepend_bos=False)
        token_ids_flat = tokens[0]

        bar.set_postfix_str("tokenising…", refresh=False)
        # year_pos: only the target year (t_new) — these are the temporal
        # constraint tokens we want the model to attend to.
        # all_year_pos: every year in the passage, kept for diagnostics only.
        year_pos = find_year_positions(token_ids_flat, model.tokenizer, target_year=t_new)
        all_year_pos = find_year_positions(token_ids_flat, model.tokenizer)

        # generation first — frees its GPU tensors before the attention pass
        bar.set_postfix_str("generating…", refresh=False)
        generated = generate_answer(model, prompt, max_new_tokens=max_new_tokens)
        success = check_match(generated, answer_new)

        # clear any CUDA fragmentation left by the generation loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # attention features: use ONLY t_new positions as the constraint source.
        # Falling back to all_year_pos only when t_new is not found at all.
        # (Using all years would dilute the signal; passages can contain
        #  hundreds of year tokens unrelated to the temporal constraint.)
        src_positions = year_pos if year_pos else all_year_pos
        bar.set_postfix_str("attn hook…", refresh=False)
        attn_vec = extract_attention_to_positions(model, tokens, src_positions)
        features = attn_vec.numpy().flatten()
        assert features.shape[0] == feat_dim

        # clear again after the attention forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        X_list.append(features)
        y_list.append(int(success))
        meta.append({
            "idx": idx,
            "instance_id": inst.get("instance_id", f"inst_{idx}"),
            # Fact-level keys needed for F2's F1-cross-reference: F1 is
            # typically run on B5 instances and F2 on B1, so the
            # instance_id namespaces differ and we have to match on the
            # underlying ``(fact_id, t_old, t_new)`` tuple instead.
            "fact_id": inst.get("fact_id", ""),
            "t_old": inst.get("t_old"),
            "t_new": inst.get("t_new"),
            "generated": generated,
            "answer_new": answer_new,
            "answer_old": inst.get("answer_old", ""),
            "success": success,
            "n_target_year_tokens": len(year_pos),
            "n_all_year_tokens": len(all_year_pos),
            "used_fallback_years": len(year_pos) == 0,
        })

        n_ok = sum(y_list)
        bar.set_postfix(
            success=f"{n_ok}/{idx+1}",
            year_toks=len(all_year_pos),
            refresh=True,
        )

    bar.close()
    return np.stack(X_list), np.array(y_list), meta


# ── Probe training ───────────────────────────────────────────────────────────

def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 0.05,
    n_folds: int = 5,
) -> ProbeResult:
    """Train L1-regularised logistic regression with stratified CV.

    Parameters
    ----------
    X : [N, L*H]
    y : [N] binary labels
    C : inverse regularisation strength.

        The methodology specifies ``C = 0.05`` (matching Yuksekgonul et al.
        2024's SAT Probe, F1-a Step 3), which is the default here.  This is
        appropriate for the planned ``≥ 500``-instance DYNAMICQA-style
        corpus where per-feature gradients are proportionally large.

        For smaller pilot datasets (e.g. the ~38-instance Phi-3 pilot the
        F1-a Step 3 gradient at w=0 is bounded by ~0.45, so ``1/C`` must
        be ``< 0.45`` for any coefficient to survive L1 (i.e. ``C > 2.2``).
        Pass ``C=10.0`` on those splits — the all-zero coefficient
        diagnostic below will warn when ``C`` is too aggressive for the
        current sample size.
    n_folds : number of CV folds

    Returns
    -------
    ProbeResult with AUROC, coefficients, and top-heads ranking.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    min_class = min(n_pos, n_neg)

    # Diagnostic: confirm C value and feature statistics
    feat_std = X.std(axis=0)
    n_constant = int((feat_std < 1e-8).sum())
    print(f"  [probe] C={C}  n={len(y)} ({n_pos}+/{n_neg}-)  "
          f"feat_range=[{X.min():.3e}, {X.max():.3e}]  "
          f"constant_feats={n_constant}/{X.shape[1]}")

    if min_class < 2:
        print(
            f"WARNING: only {n_pos} positive / {n_neg} negative samples. "
            "Cannot train a meaningful probe. Returning dummy result."
        )
        return ProbeResult(
            auroc=0.5, auroc_std=0.0,
            coef=np.zeros(X.shape[1]),
            n_samples=len(y), n_positive=n_pos,
        )

    # Attention weights are tiny (typically 0.001–0.05 for a single token
    # in a ~100-token sequence).  Without scaling, L1 with C=0.05 shrinks
    # every coefficient to exactly 0 because the penalty term dominates.
    # StandardScaler (fit on train fold only, transform both) brings all
    # features to mean=0 / std=1, making the regularisation meaningful.
    def _make_pipe() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            # liblinear: exact coordinate descent, robust on small datasets.
            # saga can fail to converge with n<100; liblinear always converges.
            ("clf", LogisticRegression(
                penalty="l1", C=C, solver="liblinear",
                max_iter=5000, random_state=42,
            )),
        ])

    actual_folds = min(n_folds, min_class)
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    aurocs: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        pipe = _make_pipe()
        pipe.fit(X[train_idx], y[train_idx])
        probs = pipe.predict_proba(X[val_idx])[:, 1]
        try:
            aurocs.append(roc_auc_score(y[val_idx], probs))
        except ValueError:
            pass

    # refit on full data for coefficient analysis
    pipe_full = _make_pipe()
    pipe_full.fit(X, y)
    coef_scaled = pipe_full.named_steps["clf"].coef_[0]

    # Convert scaled coefficients back to original-space units so that
    # the magnitude reflects actual attention-weight contribution.
    scaler: StandardScaler = pipe_full.named_steps["scaler"]
    std = scaler.scale_
    std_safe = np.where(std > 0, std, 1.0)
    coef_original = coef_scaled / std_safe

    return ProbeResult(
        auroc=float(np.mean(aurocs)) if aurocs else 0.5,
        auroc_std=float(np.std(aurocs)) if aurocs else 0.0,
        coef=coef_original,
        n_samples=len(y),
        n_positive=n_pos,
    )


# ── Weight analysis ──────────────────────────────────────────────────────────

def analyse_weights(
    result: ProbeResult,
    n_layers: int,
    n_heads: int,
    *,
    top_k: int = 10,
) -> list[tuple[int, int, float]]:
    """Rank (layer, head) pairs by absolute probe coefficient.

    Returns list of (layer, head, coefficient) sorted by |coef| desc.
    """
    coef = result.coef
    entries: list[tuple[int, int, float]] = []
    for i, c in enumerate(coef):
        layer = i // n_heads
        head = i % n_heads
        entries.append((layer, head, float(c)))

    entries.sort(key=lambda x: abs(x[2]), reverse=True)
    result.top_heads = entries[:top_k]
    return entries[:top_k]


def fallback_temporal_heads(
    top_heads: list[tuple[int, int, float]],
    *,
    top_k: int = 3,
) -> list[tuple[int, int]]:
    """Return the top-``top_k`` (layer, head) pairs by ``|coef|``.

    Used under the **temporal-head fallback** (methodology F1-a Step 5,
    final sentence): when the DYNAMICQA temporal-head validation does not
    return a $\\mathcal{H}_T$, the per-instance F1 scalar instead uses the
    top-3 attention heads by probe-coefficient magnitude.
    """
    return [(l, h) for l, h, _ in top_heads[:top_k]]


# ── F1-positive per-instance detector (methodology F1-a Step 5) ──────────────

@dataclass
class F1PositiveResult:
    """Per-instance F1 verdict from the H_T-attention scalar.

    Methodology F1-a Step 5:
      $\\bar{A}^{H_T}_i = \\tfrac{1}{|H_T|} \\sum_{(l,h) \\in H_T}
        \\tfrac{1}{|C_i|} \\sum_{c \\in C_i} A^{l,h}_{c \\to T}$
    Instance ``i`` is **F1-positive** iff $\\bar{A}^{H_T}_i$ lies below the
    25th-percentile of the B1-success scalar distribution.  We additionally
    report the sweep over ``{20, 25, 33}``-th percentiles (Step 5).
    """
    ht_heads: list[tuple[int, int]]
    is_fallback: bool
    primary_percentile: int                  # which percentile produced the primary verdict
    threshold_by_percentile: dict[int, float]
    scalar_per_instance: list[float]         # length N
    f1_positive_by_percentile: dict[int, list[bool]]


def compute_f1_positive_instances(
    X: np.ndarray,
    y: np.ndarray,
    ht_heads: list[tuple[int, int]],
    *,
    n_heads: int,
    percentiles: tuple[int, ...] = (20, 25, 33),
    primary_percentile: int = 25,
    is_fallback: bool = False,
) -> F1PositiveResult:
    """Compute the per-instance H_T-attention scalar and F1-positive verdicts.

    ``X[i, l*n_heads + h]`` must already encode the mean (over constraint
    tokens ``C_i``) attention from year tokens to the prediction position
    at head ``(l, h)`` — i.e., the features produced by
    :func:`collect_features` with the methodology-default ``agg="mean"``.

    Parameters
    ----------
    X : [N, L*H]  attention features per instance (mean-over-C_i)
    y : [N]       1 if B1 success, 0 if B1 failure
    ht_heads : list of (layer, head) pairs forming H_T (or top-3 fallback)
    n_heads : number of heads per layer (needed to flatten (l, h) → index)
    percentiles : sweep of percentile thresholds, applied to the B1-success
        scalar distribution.  The methodology requires {20, 25, 33}.
    primary_percentile : which entry of *percentiles* the primary verdict uses
    is_fallback : True if ``ht_heads`` was produced by the top-3 fallback
        rather than the DYNAMICQA temporal-head validation (Step 5 says
        the fallback must itself be flagged).

    Returns
    -------
    :class:`F1PositiveResult`.  An instance is F1-positive at percentile
    ``p`` iff its scalar is **strictly below** the ``p``-th percentile of
    the B1-success distribution.
    """
    if primary_percentile not in percentiles:
        raise ValueError(
            f"primary_percentile={primary_percentile} not in percentiles={percentiles}"
        )
    if not ht_heads:
        empty = [False] * len(y)
        return F1PositiveResult(
            ht_heads=[], is_fallback=is_fallback,
            primary_percentile=primary_percentile,
            threshold_by_percentile={p: float("nan") for p in percentiles},
            scalar_per_instance=[float("nan")] * len(y),
            f1_positive_by_percentile={p: empty for p in percentiles},
        )

    flat_idx = np.array([l * n_heads + h for (l, h) in ht_heads], dtype=int)
    scalars = X[:, flat_idx].mean(axis=1)            # mean over H_T heads

    succ_scalars = scalars[y == 1]
    if succ_scalars.size == 0:
        # Fall back to the full distribution if no success instances are
        # available (e.g. probe ran on an all-failure pilot split).
        succ_scalars = scalars

    thresholds: dict[int, float] = {
        p: float(np.percentile(succ_scalars, p)) for p in percentiles
    }
    f1_pos: dict[int, list[bool]] = {
        p: [bool(s < thresholds[p]) for s in scalars] for p in percentiles
    }

    return F1PositiveResult(
        ht_heads=list(ht_heads),
        is_fallback=is_fallback,
        primary_percentile=primary_percentile,
        threshold_by_percentile=thresholds,
        scalar_per_instance=scalars.astype(float).tolist(),
        f1_positive_by_percentile=f1_pos,
    )
