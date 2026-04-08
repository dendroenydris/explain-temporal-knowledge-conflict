"""F2 Diagnosis — "Time Set but Not Routed".

Three complementary experiments:

F2-a  STR Activation Patching — causal confirmation that temporal heads
      mediate the year routing signal.  Runs on instances where the clean
      run (question year = t_new) produces answer_new and the corrupted
      run (question year → t_old) produces answer_old (pure behavioral
      filter, no parametric-memory restriction).

F2-b  RouteScore — continuous metric for temporal signal attenuation
      between temporal-head layers and the final output, derived from
      Logit Lens trajectories cached by ``logit_lens.run_logit_lens``.

F2-c  B5 vs B6 Behavioral Cross-Analysis — cheapest F2 detector.
      Compares model accuracy when both evidence passages are present
      with years intact (B5) vs stripped (B6), cross-referenced with
      single-evidence B1 results.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from tatm.logit_lens import LogitTrajectory, get_first_answer_token, run_logit_lens
from tatm.model import (
    build_prompt,
    check_match,
    find_year_positions,
    generate_answer,
)

# ═════════════════════════════════════════════════════════════════════════════
# F2-a  STR Activation Patching
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class PatchingResult:
    """Outcome of one STR patching trial."""
    instance_id: str
    logit_diff_clean: float       # logit(new) − logit(old), clean run
    logit_diff_corrupted: float   # logit(new) − logit(old), corrupted run
    logit_diff_patched: float     # logit(new) − logit(old), patched run
    recovery_fraction: float      # (patched − corrupted) / (clean − corrupted)
    top1_clean: str
    top1_corrupted: str
    top1_patched: str


def _patch_z_hook(
    z: torch.Tensor,
    hook,
    *,
    clean_z: torch.Tensor,
    head_indices: list[int],
    positions: list[int],
) -> torch.Tensor:
    """Replace z[batch, positions, heads, :] with values from *clean_z*."""
    patched = z.clone()
    for pos in positions:
        for h in head_indices:
            patched[:, pos, h, :] = clean_z[:, pos, h, :]
    return patched


def run_str_patching(
    model: HookedTransformer,
    instances: list[dict],
    temporal_heads: list[tuple[int, int]],
    *,
    template: str = "plain",
    verbose: bool = True,
) -> list[PatchingResult]:
    """F2-a: STR Activation Patching on B5 instances.

    For each instance that passes the behavioral filter (clean → answer_new
    AND corrupted → answer_old), patches the ``hook_z`` activations at the
    year-token positions of every temporal head from the clean run into the
    corrupted run and measures logit-difference recovery.

    Position alignment strategy: rather than heuristically locating the
    question region, we diff the two token sequences and patch exactly the
    positions that changed (= the question-year tokens).  If the two
    sequences differ in length we skip the instance.

    Parameters
    ----------
    model : HookedTransformer
    instances : B5-type EvalInstance dicts (must have ``context``,
        ``question``, ``answer_new``, ``answer_old``, ``t_new``, ``t_old``)
    temporal_heads : list of ``(layer, head)`` tuples from EAP-IG / circuit
        discovery
    template : prompt template name
    verbose : show progress bar
    """
    heads_by_layer: dict[int, list[int]] = {}
    for layer, head in temporal_heads:
        heads_by_layer.setdefault(layer, []).append(head)
    patch_layers = sorted(heads_by_layer.keys())
    cache_names = {f"blocks.{l}.attn.hook_z" for l in patch_layers}

    results: list[PatchingResult] = []
    skipped = {"gen_clean": 0, "gen_corr": 0, "len_mismatch": 0,
               "no_diff": 0, "token_id": 0}

    bar = tqdm(instances, desc="F2-a  STR patching", disable=not verbose)

    for inst in bar:
        iid = inst.get("instance_id", "")
        question = inst.get("question", "")
        context = inst.get("context", "")
        answer_new = inst.get("answer_new", "")
        answer_old = inst.get("answer_old", "")
        t_new = inst.get("t_new")
        t_old = inst.get("t_old")

        if t_new is None or t_old is None:
            continue

        # ── Build clean & corrupted prompts ───────────────────────────
        clean_prompt = build_prompt(context, question, template=template)
        corrupted_q = question.replace(str(t_new), str(t_old))
        corrupted_prompt = build_prompt(context, corrupted_q, template=template)

        clean_tokens = model.to_tokens(clean_prompt, prepend_bos=False)
        corrupted_tokens = model.to_tokens(corrupted_prompt, prepend_bos=False)

        # Sequences must be equal length for position-aligned patching
        if clean_tokens.shape != corrupted_tokens.shape:
            skipped["len_mismatch"] += 1
            continue

        # Diff positions = question-year tokens that changed
        diff_pos = (clean_tokens[0] != corrupted_tokens[0]).nonzero(
            as_tuple=True
        )[0].tolist()
        if not diff_pos:
            skipped["no_diff"] += 1
            continue

        # ── Behavioral filter: generate & verify ──────────────────────
        clean_gen = generate_answer(model, clean_prompt)
        if not check_match(clean_gen, answer_new):
            skipped["gen_clean"] += 1
            continue

        corrupted_gen = generate_answer(model, corrupted_prompt)
        if not check_match(corrupted_gen, answer_old):
            skipped["gen_corr"] += 1
            continue

        # Answer token IDs — must be valid and distinct
        new_tid = get_first_answer_token(model, answer_new)
        old_tid = get_first_answer_token(model, answer_old)
        if new_tid < 0 or old_tid < 0 or new_tid == old_tid:
            skipped["token_id"] += 1
            continue

        # ── Clean run: forward + cache z at temporal-head layers ──────
        with torch.no_grad():
            clean_logits, cache = model.run_with_cache(
                clean_tokens,
                names_filter=lambda n: n in cache_names,
                prepend_bos=False,
            )
        clean_last = clean_logits[0, -1].float().cpu()

        # ── Corrupted run (no patching) ───────────────────────────────
        with torch.no_grad():
            corr_last = model(
                corrupted_tokens, prepend_bos=False
            )[0, -1].float().cpu()

        # ── Patched run: replace z at diff positions from clean ───────
        patch_hooks = []
        for layer in patch_layers:
            cached_z = cache[f"blocks.{layer}.attn.hook_z"]
            patch_hooks.append((
                f"blocks.{layer}.attn.hook_z",
                partial(
                    _patch_z_hook,
                    clean_z=cached_z,
                    head_indices=heads_by_layer[layer],
                    positions=diff_pos,
                ),
            ))

        with torch.no_grad():
            patched_last = model.run_with_hooks(
                corrupted_tokens, fwd_hooks=patch_hooks, prepend_bos=False,
            )[0, -1].float().cpu()

        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ── Metrics ───────────────────────────────────────────────────
        ld_clean = (clean_last[new_tid] - clean_last[old_tid]).item()
        ld_corr = (corr_last[new_tid] - corr_last[old_tid]).item()
        ld_patch = (patched_last[new_tid] - patched_last[old_tid]).item()

        denom = ld_clean - ld_corr
        recovery = (ld_patch - ld_corr) / denom if abs(denom) > 1e-6 else 0.0

        results.append(PatchingResult(
            instance_id=iid,
            logit_diff_clean=ld_clean,
            logit_diff_corrupted=ld_corr,
            logit_diff_patched=ld_patch,
            recovery_fraction=recovery,
            top1_clean=model.tokenizer.decode([int(clean_last.argmax())]),
            top1_corrupted=model.tokenizer.decode([int(corr_last.argmax())]),
            top1_patched=model.tokenizer.decode([int(patched_last.argmax())]),
        ))
        bar.set_postfix(rec=f"{recovery:.2f}", ok=len(results))

    bar.close()
    if verbose and any(skipped.values()):
        parts = [f"{k}={v}" for k, v in skipped.items() if v]
        print(f"  [F2-a] Skipped: {', '.join(parts)}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# F2-b  RouteScore
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class RouteScoreResult:
    """RouteScore and full Logit Lens trajectory for one instance."""
    instance_id: str
    p_new_at_temporal: float   # P^{L_T}(answer_new)
    p_new_at_final: float      # P^{L_final}(answer_new)
    route_score: float         # p_new_at_temporal − p_new_at_final
    trajectory: LogitTrajectory


def compute_route_scores(
    model: HookedTransformer,
    instances: list[dict],
    temporal_head_layers: list[int],
    *,
    template: str = "plain",
    verbose: bool = True,
) -> list[RouteScoreResult]:
    """F2-b: Compute RouteScore for each instance.

    ``RouteScore = P^{L_T}(answer_new) − P^{L_final}(answer_new)``

    where *L_T* is the latest temporal-head layer and *L_final* is the
    last model layer.  The full ``LogitTrajectory`` is retained so it
    can be reused for F3-a analysis.

    Parameters
    ----------
    model : HookedTransformer
    instances : EvalInstance dicts (B1-failure set or any subset)
    temporal_head_layers : layer indices where temporal heads reside
    template : prompt template name
    """
    l_t = max(temporal_head_layers)

    results: list[RouteScoreResult] = []
    bar = tqdm(instances, desc="F2-b  RouteScore", disable=not verbose)

    for inst in bar:
        iid = inst.get("instance_id", "")
        question = inst.get("question", "")
        context = inst.get("context", "")
        answer_new = inst.get("answer_new", "")
        answer_old = inst.get("answer_old", "")

        prompt = build_prompt(context, question, template=template)
        tokens = model.to_tokens(prompt, prepend_bos=False)

        new_tid = get_first_answer_token(model, answer_new)
        old_tid = get_first_answer_token(model, answer_old)
        if new_tid < 0 or old_tid < 0 or new_tid == old_tid:
            continue

        traj = run_logit_lens(model, tokens, new_tid, old_tid)

        p_temporal = float(traj.probs_new[l_t])
        p_final = float(traj.probs_new[-1])
        rs = p_temporal - p_final

        results.append(RouteScoreResult(
            instance_id=iid,
            p_new_at_temporal=p_temporal,
            p_new_at_final=p_final,
            route_score=rs,
            trajectory=traj,
        ))
        bar.set_postfix(rs=f"{rs:+.3f}", n=len(results))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bar.close()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# F2-c  B5 vs B6 Behavioral Cross-Analysis
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class BehavioralCrossResult:
    """Aggregate and per-instance B1 / B5 / B6 comparison."""
    b1_accuracy: float
    b5_accuracy: float
    b6_accuracy: float
    n_b1_fail_b5_success: int
    n_b1_fail_b5_fail: int
    n_b1_fail_b5_fail_rate: float    # among B1-failures, fraction also failing B5
    n_total: int
    details: list[dict] = field(default_factory=list)


def behavioral_cross_analysis(
    model: HookedTransformer,
    b1_instances: list[dict],
    b5_instances: list[dict],
    b6_instances: list[dict],
    *,
    b1_success_map: dict[str, bool] | None = None,
    template: str = "plain",
    verbose: bool = True,
) -> BehavioralCrossResult:
    """F2-c: B5 vs B6 behavioral cross-analysis.

    Instances must share the same ordering so that ``b1[i]``, ``b5[i]``,
    ``b6[i]`` correspond to the same ``(subject, property, t_old, t_new)``
    tuple.  The easiest way is to sort each list by ``fact_id`` before
    calling this function.

    Parameters
    ----------
    b1_success_map : optional {instance_id → bool} from a prior B1 generation
        pass (e.g. REVERTS_OLD filter).  When provided, B1 generation is
        skipped and the cached result is used instead — avoids duplicate
        forward passes.

    Interpretation
    --------------
    - **B5 >> B6**: year in evidence is the critical routing signal.
    - **B1-fail ∩ B5-success**: not a routing problem; single-passage
      persuasion is too weak (excludes F2).
    - **B1-fail ∩ B5-fail**: strong F2 signal — model cannot use the year
      to select the correct evidence even when both are present.
    """
    assert len(b1_instances) == len(b5_instances) == len(b6_instances), (
        f"Instance lists must have equal length: "
        f"B1={len(b1_instances)}, B5={len(b5_instances)}, B6={len(b6_instances)}"
    )

    b1_ok = b5_ok = b6_ok = 0
    b1f_b5s = 0   # B1-fail ∩ B5-success
    b1f_b5f = 0   # B1-fail ∩ B5-fail
    details: list[dict] = []

    desc = "F2-c  B5 vs B6"
    if b1_success_map is not None:
        desc += " (B1 cached)"

    bar = tqdm(
        zip(b1_instances, b5_instances, b6_instances),
        total=len(b1_instances),
        desc=desc,
        disable=not verbose,
    )

    for b1, b5, b6 in bar:
        answer_new = b1.get("answer_new", "")

        def _gen_eval(inst: dict) -> tuple[bool, str]:
            ctx = inst.get("context", "")
            q = inst.get("question", "")
            prompt = build_prompt(ctx, q, template=template)
            gen = generate_answer(model, prompt)
            return check_match(gen, answer_new), gen

        # B1: reuse cached result when available
        iid = b1.get("instance_id", "")
        if b1_success_map is not None and iid in b1_success_map:
            r1 = b1_success_map[iid]
            gen1 = "(cached)"
        else:
            r1, gen1 = _gen_eval(b1)

        r5, gen5 = _gen_eval(b5)
        r6, gen6 = _gen_eval(b6)

        if r1:
            b1_ok += 1
        if r5:
            b5_ok += 1
        if r6:
            b6_ok += 1

        if not r1 and r5:
            b1f_b5s += 1
        if not r1 and not r5:
            b1f_b5f += 1

        details.append({
            "fact_id": b1.get("fact_id", ""),
            "b1_success": r1,
            "b5_success": r5,
            "b6_success": r6,
            "b5_generated": gen5,
            "b6_generated": gen6,
        })
        bar.set_postfix(
            b1=f"{b1_ok}/{len(details)}",
            b5=f"{b5_ok}/{len(details)}",
            b6=f"{b6_ok}/{len(details)}",
        )

    bar.close()

    n = len(b1_instances)
    n_b1_fail = n - b1_ok

    return BehavioralCrossResult(
        b1_accuracy=b1_ok / n if n else 0,
        b5_accuracy=b5_ok / n if n else 0,
        b6_accuracy=b6_ok / n if n else 0,
        n_b1_fail_b5_success=b1f_b5s,
        n_b1_fail_b5_fail=b1f_b5f,
        n_b1_fail_b5_fail_rate=b1f_b5f / n_b1_fail if n_b1_fail else 0.0,
        n_total=n,
        details=details,
    )
