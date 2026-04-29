"""F3 diagnosis — parametric override mechanistic tests."""
from __future__ import annotations

import gc
import json
import math
import random
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from tatm.logit_lens import run_logit_lens
from tatm.model import build_prompt, check_match, get_first_answer_token


@dataclass
class F3PreparedInstance:
    """Layer-2 instance enriched with cached Layer-3 parametric answer."""
    row: dict
    layer3: dict
    a_param: str
    param_class: str
    tok_new: int
    tok_param: int
    b1_success: bool | None = None

    @property
    def instance_id(self) -> str:
        return str(self.row.get("instance_id", ""))


@dataclass
class DLAResult:
    instance_id: str
    param_class: str
    condition: str
    actual_logit_diff: float
    total_contribution: float
    residual_error: float
    component_rows: list[dict] = field(default_factory=list)
    top_negative_late: list[dict] = field(default_factory=list)


def layer2_key(row: dict) -> tuple[str, Any, Any]:
    return (str(row.get("fact_id", "")), row.get("t_old"), row.get("t_new"))


def load_layer3_by_key(path: str) -> tuple[dict[str, dict], dict[tuple[str, Any, Any], dict]]:
    by_id: dict[str, dict] = {}
    by_key: dict[tuple[str, Any, Any], dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iid = str(row.get("instance_id", ""))
            if iid:
                by_id[iid] = row
            key = layer2_key(row)
            if key[0]:
                by_key[key] = row
    return by_id, by_key


def classify_parametric_answer(layer3_row: dict, answer_old: str, answer_new: str) -> str:
    answer = str(layer3_row.get("extracted_answer") or layer3_row.get("model_output_raw") or "")
    if layer3_row.get("matches_answer_new") or check_match(answer, answer_new):
        return "PARAM_NEW"
    if layer3_row.get("matches_answer_old") or check_match(answer, answer_old):
        return "PARAM_OLD"
    return "PARAM_OTHER"


def prepare_f3_instances(
    model: HookedTransformer,
    b1_instances: list[dict],
    layer3_by_id: dict[str, dict],
    layer3_by_key: dict[tuple[str, Any, Any], dict],
) -> list[F3PreparedInstance]:
    """Attach cached Layer-3 answers and target tokens to B1 instances."""
    prepared: list[F3PreparedInstance] = []
    skipped = {"missing_layer3": 0, "empty_param": 0, "token": 0, "same_token": 0}

    for row in b1_instances:
        layer3 = layer3_by_id.get(str(row.get("instance_id", "")))
        if layer3 is None:
            layer3 = layer3_by_key.get(layer2_key(row))
        if layer3 is None:
            skipped["missing_layer3"] += 1
            continue

        a_param = str(layer3.get("extracted_answer") or layer3.get("model_output_raw") or "").strip()
        if not a_param:
            skipped["empty_param"] += 1
            continue

        tok_new = get_first_answer_token(model, str(row.get("answer_new", "")))
        tok_param = get_first_answer_token(model, a_param)
        if tok_new < 0 or tok_param < 0:
            skipped["token"] += 1
            continue
        if tok_new == tok_param:
            skipped["same_token"] += 1
            continue

        prepared.append(F3PreparedInstance(
            row=row,
            layer3=layer3,
            a_param=a_param,
            param_class=classify_parametric_answer(
                layer3,
                str(row.get("answer_old", "")),
                str(row.get("answer_new", "")),
            ),
            tok_new=tok_new,
            tok_param=tok_param,
        ))

    if any(skipped.values()):
        parts = ", ".join(f"{k}={v}" for k, v in skipped.items() if v)
        print(f"[F3] Skipped during preparation: {parts}")
    return prepared


def _ensure_attn_result(model: HookedTransformer) -> None:
    model.cfg.use_attn_result = True


def _tokens(model: HookedTransformer, prompt: str) -> torch.Tensor:
    return model.to_tokens(prompt, prepend_bos=False)


def _logit_diff(logits: torch.Tensor, tok_new: int, tok_param: int) -> torch.Tensor:
    return logits[..., tok_new] - logits[..., tok_param]


def _component_score(component: torch.Tensor, direction: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (component.float() @ direction.float()) / scale.float().clamp_min(1e-6)


def _decode_token(model: HookedTransformer, token_id: int) -> str:
    return model.tokenizer.decode([int(token_id)]).strip()


def _cache_b1(
    model: HookedTransformer,
    inst: F3PreparedInstance,
    *,
    template: str,
    names: set[str],
) -> tuple[str, torch.Tensor, torch.Tensor, Any]:
    prompt = build_prompt(str(inst.row.get("context", "")), str(inst.row.get("question", "")), template=template)
    tokens = _tokens(model, prompt)
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in names,
            prepend_bos=False,
        )
    return prompt, tokens, logits, cache


def run_f3a_dla(
    model: HookedTransformer,
    instances: list[F3PreparedInstance],
    *,
    template: str,
    top_k_negative: int = 8,
    late_fraction: float = 2 / 3,
) -> tuple[list[DLAResult], list[dict]]:
    """F3-a: Direct Logit Attribution over attention heads and MLPs."""
    _ensure_attn_result(model)
    n_layers = model.cfg.n_layers
    names = {
        "hook_embed",
        *{f"blocks.{l}.hook_resid_post" for l in range(n_layers)},
        *{f"blocks.{l}.attn.hook_result" for l in range(n_layers)},
        *{f"blocks.{l}.hook_mlp_out" for l in range(n_layers)},
    }
    late_start = int(math.floor(n_layers * late_fraction))
    results: list[DLAResult] = []
    long_rows: list[dict] = []

    for inst in tqdm(instances, desc="F3-a DLA", unit="inst", dynamic_ncols=True):
        _, _, logits, cache = _cache_b1(model, inst, template=template, names=names)
        final_resid = cache[f"blocks.{n_layers - 1}.hook_resid_post"][:, -1, :]
        scale = final_resid.norm(dim=-1) / math.sqrt(model.cfg.d_model)
        direction = model.W_U[:, inst.tok_new] - model.W_U[:, inst.tok_param]
        actual = float(_logit_diff(logits[:, -1, :].float(), inst.tok_new, inst.tok_param)[0].item())

        rows: list[dict] = []
        total = 0.0

        if "hook_embed" in cache:
            value = float(_component_score(cache["hook_embed"][:, -1, :], direction, scale)[0].item())
            rows.append(_f3_row(inst, "F3a_DLA", "logit_diff_contribution", None, "embed", value, "B1"))
            total += value

        for layer in range(n_layers):
            head_result = cache[f"blocks.{layer}.attn.hook_result"][:, -1, :, :]
            for head in range(head_result.shape[1]):
                value = float(_component_score(head_result[:, head, :], direction, scale)[0].item())
                rows.append(_f3_row(inst, "F3a_DLA", "logit_diff_contribution", layer, f"attn_head_{head}", value, "B1"))
                total += value

            mlp = cache[f"blocks.{layer}.hook_mlp_out"][:, -1, :]
            value = float(_component_score(mlp, direction, scale)[0].item())
            rows.append(_f3_row(inst, "F3a_DLA", "logit_diff_contribution", layer, "mlp", value, "B1"))
            total += value

        denom = max(abs(actual), 1e-6)
        residual_error = abs(total - actual) / denom
        threshold = 0.05 * abs(actual)
        top_negative_late = sorted(
            [
                row for row in rows
                if row["layer"] is not None
                and int(row["layer"]) >= late_start
                and row["value"] < -threshold
                and (str(row["component"]).startswith("attn_head_") or row["component"] == "mlp")
            ],
            key=lambda row: row["value"],
        )[:top_k_negative]

        result = DLAResult(
            instance_id=inst.instance_id,
            param_class=inst.param_class,
            condition="B1",
            actual_logit_diff=actual,
            total_contribution=total,
            residual_error=residual_error,
            component_rows=rows,
            top_negative_late=top_negative_late,
        )
        results.append(result)
        long_rows.extend(rows)
        long_rows.append(_f3_row(inst, "F3a_DLA", "actual_logit_diff", None, "final_logits", actual, "B1"))
        long_rows.append(_f3_row(inst, "F3a_DLA", "residual_error", None, "decomposition", residual_error, "B1"))

        del cache, logits
        _cleanup()

    return results, long_rows


def run_f3b_ffn_lens(
    model: HookedTransformer,
    instances: list[F3PreparedInstance],
    *,
    template: str,
    condition: str = "B1",
) -> list[dict]:
    """F3-b: project each layer's MLP output through the unembedding."""
    n_layers = model.cfg.n_layers
    names = {f"blocks.{l}.hook_mlp_out" for l in range(n_layers)}
    rows: list[dict] = []

    for inst in tqdm(instances, desc=f"F3-b FFN lens ({condition})", unit="inst", dynamic_ncols=True):
        if condition == "A1":
            prompt = build_prompt("", str(inst.row.get("question", "")), template=template)
            tokens = _tokens(model, prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names, prepend_bos=False)
        else:
            _, _, _, cache = _cache_b1(model, inst, template=template, names=names)

        for layer in range(n_layers):
            h_mlp = cache[f"blocks.{layer}.hook_mlp_out"][:, -1, :].float()
            scale = (h_mlp.norm(dim=-1) / math.sqrt(model.cfg.d_model)).clamp_min(1e-6)
            pseudo_logits = (h_mlp / scale.unsqueeze(-1)) @ model.W_U.float()
            param_logit = pseudo_logits[:, inst.tok_param]
            new_logit = pseudo_logits[:, inst.tok_new]
            rank_param = int((pseudo_logits > param_logit.unsqueeze(-1)).sum(dim=-1).item() + 1)
            rank_new = int((pseudo_logits > new_logit.unsqueeze(-1)).sum(dim=-1).item() + 1)
            top_values, top_indices = pseudo_logits.topk(10, dim=-1)
            top_probs = torch.softmax(top_values[0], dim=-1)
            rows.append({
                "instance_id": inst.instance_id,
                "param_class": inst.param_class,
                "condition": condition,
                "layer": layer,
                "rank_param": rank_param,
                "rank_new": rank_new,
                "a_param": inst.a_param,
                "answer_new": inst.row.get("answer_new", ""),
                "logit_param": float(param_logit.item()),
                "logit_new": float(new_logit.item()),
                "top10_tokens": [_decode_token(model, int(t)) for t in top_indices[0].tolist()],
                "top10_token_ids": [int(t) for t in top_indices[0].tolist()],
                "top10_probs": [float(v) for v in top_probs.tolist()],
                "is_a_param_writer": rank_param <= 10 and rank_new > 50,
            })

        del cache
        _cleanup()

    return rows


def run_f3c_dual_trajectory(
    model: HookedTransformer,
    instances: list[F3PreparedInstance],
    *,
    template: str,
    epsilon: float = 0.0,
) -> list[dict]:
    """F3-c: compare A1 and B1 logit-diff trajectories."""
    rows: list[dict] = []
    n_layers = model.cfg.n_layers

    for inst in tqdm(instances, desc="F3-c A1/B1 trajectory", unit="inst", dynamic_ncols=True):
        a1_prompt = build_prompt("", str(inst.row.get("question", "")), template=template)
        b1_prompt = build_prompt(str(inst.row.get("context", "")), str(inst.row.get("question", "")), template=template)
        a1_traj = run_logit_lens(model, _tokens(model, a1_prompt), inst.tok_new, inst.tok_param)
        b1_traj = run_logit_lens(model, _tokens(model, b1_prompt), inst.tok_new, inst.tok_param)
        ld_a1 = a1_traj.logits_new - a1_traj.logits_old
        ld_b1 = b1_traj.logits_new - b1_traj.logits_old
        d_traj = ld_b1 - ld_a1
        mid = d_traj[n_layers // 3 : max(n_layers // 3 + 1, 2 * n_layers // 3)]
        d_mid_max = float(np.max(mid)) if len(mid) else float(np.max(d_traj))
        d_late_final = float(d_traj[-1])
        d_drop = d_mid_max - d_late_final
        if d_mid_max < epsilon:
            subtype = "F3a"
        elif d_mid_max > epsilon and d_late_final < epsilon:
            subtype = "F3b"
        elif d_late_final > epsilon and inst.b1_success is False:
            subtype = "F3-paradox"
        else:
            subtype = "ambiguous"
        rows.append({
            "instance_id": inst.instance_id,
            "param_class": inst.param_class,
            "a_param": inst.a_param,
            "answer_new": inst.row.get("answer_new", ""),
            "ld_a1": [float(x) for x in ld_a1.tolist()],
            "ld_b1": [float(x) for x in ld_b1.tolist()],
            "D_trajectory": [float(x) for x in d_traj.tolist()],
            "D_mid_max": d_mid_max,
            "D_late_final": d_late_final,
            "D_drop": float(d_drop),
            "F3_subtype": subtype,
        })
        _cleanup()

    return rows


def run_f3d_targeted_patch(
    model: HookedTransformer,
    instances: list[F3PreparedInstance],
    donor_pool: list[F3PreparedInstance],
    dla_results: list[DLAResult],
    *,
    template: str,
    max_components: int = 6,
    random_seed: int = 1234,
) -> list[dict]:
    """F3-d: patch top negative late components from PARAM_NEW donors."""
    _ensure_attn_result(model)
    rng = random.Random(random_seed)
    by_iid = {result.instance_id: result for result in dla_results}
    rows: list[dict] = []

    for inst in tqdm(instances, desc="F3-d patch", unit="inst", dynamic_ncols=True):
        dla = by_iid.get(inst.instance_id)
        if not dla or len(dla.top_negative_late) < 1:
            continue
        components = dla.top_negative_late[:max_components]
        donor = _select_donor(inst, donor_pool, rng)
        if donor is None:
            continue
        names = _component_cache_names(components)
        _, _, _, donor_cache = _cache_b1(model, donor, template=template, names=names)
        prompt = build_prompt(str(inst.row.get("context", "")), str(inst.row.get("question", "")), template=template)
        tokens = _tokens(model, prompt)
        with torch.no_grad():
            clean_logits = model(tokens, prepend_bos=False)
            patched_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=_make_patch_hooks(components, donor_cache, mode="donor"),
                prepend_bos=False,
            )
            zero_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=_make_patch_hooks(components, donor_cache, mode="zero"),
                prepend_bos=False,
            )
        ld_clean = float(_logit_diff(clean_logits[:, -1, :].float(), inst.tok_new, inst.tok_param)[0].item())
        ld_patched = float(_logit_diff(patched_logits[:, -1, :].float(), inst.tok_new, inst.tok_param)[0].item())
        ld_zero = float(_logit_diff(zero_logits[:, -1, :].float(), inst.tok_new, inst.tok_param)[0].item())
        rows.append({
            "instance_id": inst.instance_id,
            "param_class": inst.param_class,
            "patch_config": "targeted_patch",
            "donor_id": donor.instance_id,
            "n_patched_components": len(components),
            "ld_clean": ld_clean,
            "ld_patched": ld_patched,
            "recovery": ld_patched - ld_clean,
            "components": components,
        })
        rows.append({
            "instance_id": inst.instance_id,
            "param_class": inst.param_class,
            "patch_config": "zero_knockout",
            "donor_id": "",
            "n_patched_components": len(components),
            "ld_clean": ld_clean,
            "ld_patched": ld_zero,
            "recovery": ld_zero - ld_clean,
            "components": components,
        })
        del donor_cache, clean_logits, patched_logits, zero_logits
        _cleanup()

    return rows


def run_f3e_causal_trace(
    model: HookedTransformer,
    instances: list[F3PreparedInstance],
    *,
    template: str,
    replacement_subject: str = "Germany",
    max_instances: int = 20,
) -> list[dict]:
    """F3-e: lightweight causal tracing for parametric answer storage.

    This patches ``hook_resid_pre`` from a clean A1 run into a subject-corrupted
    A1 run at the last position and detected subject-token positions.
    """
    n_layers = model.cfg.n_layers
    names = {f"blocks.{l}.hook_resid_pre" for l in range(n_layers)}
    rows: list[dict] = []

    for inst in tqdm(instances[:max_instances], desc="F3-e causal trace", unit="inst", dynamic_ncols=True):
        question = str(inst.row.get("question", ""))
        subject = str(inst.row.get("subject_label", ""))
        clean_prompt = build_prompt("", question, template=template)
        corrupted_question = corrupt_subject_in_question(question, subject, replacement=replacement_subject)
        corrupted_prompt = build_prompt("", corrupted_question, template=template)
        clean_tokens = _tokens(model, clean_prompt)
        corrupted_tokens = _tokens(model, corrupted_prompt)
        subject_positions = _find_subject_positions(model, clean_tokens, subject)
        patch_positions = [("last", -1), *[(f"subject_{pos}", pos) for pos in subject_positions]]

        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(
                clean_tokens,
                names_filter=lambda name: name in names,
                prepend_bos=False,
            )
            corrupted_logits = model(corrupted_tokens, prepend_bos=False)

        p_clean = float(torch.softmax(clean_logits[:, -1, :].float(), dim=-1)[0, inst.tok_param].item())
        p_corrupted = float(torch.softmax(corrupted_logits[:, -1, :].float(), dim=-1)[0, inst.tok_param].item())
        threshold = 0.5 * p_clean

        for pos_label, pos in patch_positions:
            for layer in range(n_layers):
                name = f"blocks.{layer}.hook_resid_pre"
                clean_value = clean_cache[name][:, pos, :].detach()

                def _patch_resid(activation: torch.Tensor, hook, *, patch_pos=pos, value=clean_value):
                    patched = activation.clone()
                    patched[:, patch_pos, :] = value.to(patched.device)
                    return patched

                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        corrupted_tokens,
                        fwd_hooks=[(name, _patch_resid)],
                        prepend_bos=False,
                    )
                p_patched = float(torch.softmax(patched_logits[:, -1, :].float(), dim=-1)[0, inst.tok_param].item())
                rows.append({
                    "instance_id": inst.instance_id,
                    "param_class": inst.param_class,
                    "position": pos_label,
                    "layer": layer,
                    "a_param": inst.a_param,
                    "p_clean_param": p_clean,
                    "p_corrupted_param": p_corrupted,
                    "p_patched_param": p_patched,
                    "recovery": p_patched - p_corrupted,
                    "is_storage_candidate": p_patched > threshold,
                })

        del clean_cache, clean_logits, corrupted_logits
        _cleanup()

    return rows


def _f3_row(
    inst: F3PreparedInstance,
    experiment: str,
    metric_name: str,
    layer: int | None,
    component: str,
    value: float,
    condition: str,
) -> dict:
    return {
        "instance_id": inst.instance_id,
        "param_class": inst.param_class,
        "experiment": experiment,
        "metric_name": metric_name,
        "layer": layer,
        "component": component,
        "value": value,
        "condition": condition,
        "a_param": inst.a_param,
        "answer_new": inst.row.get("answer_new", ""),
    }


def _component_cache_names(components: list[dict]) -> set[str]:
    names: set[str] = set()
    for comp in components:
        layer = int(comp["layer"])
        component = str(comp["component"])
        if component.startswith("attn_head_"):
            names.add(f"blocks.{layer}.attn.hook_result")
        elif component == "mlp":
            names.add(f"blocks.{layer}.hook_mlp_out")
    return names


def _make_patch_hooks(components: list[dict], donor_cache, *, mode: str) -> list[tuple[str, Any]]:
    hooks: list[tuple[str, Any]] = []
    for comp in components:
        layer = int(comp["layer"])
        component = str(comp["component"])
        if component.startswith("attn_head_"):
            head = int(component.rsplit("_", 1)[-1])
            name = f"blocks.{layer}.attn.hook_result"
            donor_value = donor_cache[name][:, -1, head, :].detach()
            hooks.append((name, partial(_patch_attn_result, head=head, donor_value=donor_value, mode=mode)))
        elif component == "mlp":
            name = f"blocks.{layer}.hook_mlp_out"
            donor_value = donor_cache[name][:, -1, :].detach()
            hooks.append((name, partial(_patch_mlp_out, donor_value=donor_value, mode=mode)))
    return hooks


def _patch_attn_result(
    activation: torch.Tensor,
    hook,
    *,
    head: int,
    donor_value: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    patched = activation.clone()
    patched[:, -1, head, :] = 0 if mode == "zero" else donor_value.to(patched.device)
    return patched


def _patch_mlp_out(
    activation: torch.Tensor,
    hook,
    *,
    donor_value: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    patched = activation.clone()
    patched[:, -1, :] = 0 if mode == "zero" else donor_value.to(patched.device)
    return patched


def _select_donor(
    inst: F3PreparedInstance,
    donor_pool: list[F3PreparedInstance],
    rng: random.Random,
) -> F3PreparedInstance | None:
    same_relation = [
        donor for donor in donor_pool
        if donor.instance_id != inst.instance_id
        and donor.row.get("property_pid") == inst.row.get("property_pid")
        and donor.row.get("subject_qid") != inst.row.get("subject_qid")
    ]
    if same_relation:
        return rng.choice(same_relation)
    different = [donor for donor in donor_pool if donor.instance_id != inst.instance_id]
    return rng.choice(different) if different else None


def _cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def corrupt_subject_in_question(question: str, subject: str, replacement: str = "Germany") -> str:
    if subject and subject in question:
        return question.replace(subject, replacement, 1)
    return re.sub(r"of ([^?]+)\?", f"of {replacement}?", question, count=1)


def _find_subject_positions(
    model: HookedTransformer,
    prompt_tokens: torch.Tensor,
    subject: str,
) -> list[int]:
    if not subject:
        return []
    subject_ids = model.to_tokens(subject, prepend_bos=False)[0].tolist()
    token_ids = prompt_tokens[0].tolist()
    if not subject_ids:
        return []
    for start in range(0, len(token_ids) - len(subject_ids) + 1):
        if token_ids[start : start + len(subject_ids)] == subject_ids:
            return list(range(start, start + len(subject_ids)))
    # Some tokenizers encode leading spaces differently.
    subject_ids = model.to_tokens(" " + subject, prepend_bos=False)[0].tolist()
    for start in range(0, len(token_ids) - len(subject_ids) + 1):
        if token_ids[start : start + len(subject_ids)] == subject_ids:
            return list(range(start, start + len(subject_ids)))
    return []
