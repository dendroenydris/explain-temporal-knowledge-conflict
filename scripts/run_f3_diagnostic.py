#!/usr/bin/env python3
"""Run F3 parametric override diagnostics.

Inputs:
  - Layer-2 JSONL with B1 instances.
  - Layer-3 JSONL with cached parametric answers.

Outputs:
  - f3_manifest.json
  - f3a_dla_long.jsonl
  - f3a_dla_summary.json
  - f3b_ffn_lens.jsonl
  - f3c_dual_trajectory.json
  - f3d_patch.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "source"))

from tatm.f3_diagnosis import (  # noqa: E402
    F3PreparedInstance,
    build_f3_pair_prompts,
    load_layer3_by_key,
    prepare_f3_instances,
    run_f3a_dla,
    run_f3b_ffn_lens,
    run_f3c_dual_trajectory,
    run_f3d_targeted_patch,
    run_f3e_causal_trace,
)
from tatm.model import build_prompt, generate_answer, load_model  # noqa: E402


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs used by F3 for reproducible diagnostics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_b1(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if str(row.get("instance_id", "")).startswith("B1"):
                    records.append(row)
    if not records:
        raise ValueError(f"No B1 instances found in {path}")
    return records


def restrict_by_fact(
    instances: list[dict],
    *,
    number: int | None,
    seed: int,
) -> list[dict]:
    if number is None:
        return instances
    if number < 1:
        raise ValueError("--number must be >= 1")
    fact_ids = sorted({str(row.get("fact_id") or row.get("instance_id")) for row in instances})
    if number >= len(fact_ids):
        return instances
    selected = set(random.Random(seed).sample(fact_ids, number))
    return [row for row in instances if str(row.get("fact_id") or row.get("instance_id")) in selected]


def _rouge_tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def rouge_l_f1(reference: str, prediction: str) -> float:
    """Compute word-level RougeL F1 without adding an external dependency."""
    ref = _rouge_tokens(reference)
    pred = _rouge_tokens(prediction)
    if not ref or not pred:
        return 0.0

    # Longest common subsequence length.
    prev = [0] * (len(pred) + 1)
    for ref_tok in ref:
        curr = [0]
        for j, pred_tok in enumerate(pred, start=1):
            if ref_tok == pred_tok:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def classify_b1_behavior(
    model,
    instances: list[F3PreparedInstance],
    *,
    template: str,
    out_dir: Path,
    rouge_threshold: float = 0.3,
    rouge_margin: float = 0.1,
    skip_generation: bool = False,
) -> None:
    """Set B1 success labels used for F3 target sets."""
    log: list[dict] = []
    if skip_generation:
        for inst in instances:
            inst.b1_success = None
        return

    for inst in tqdm(instances, desc="F3 B1 behavior", unit="inst", dynamic_ncols=True):
        row = inst.row
        prompt = build_prompt(str(row.get("context", "")), str(row.get("question", "")), template=template)
        generated = generate_answer(model, prompt)
        rouge_new = rouge_l_f1(str(row.get("answer_new", "")), generated)
        rouge_param = rouge_l_f1(inst.a_param, generated)
        both_pass = rouge_new > rouge_threshold and rouge_param > rouge_threshold
        ambiguous = both_pass and abs(rouge_new - rouge_param) < rouge_margin
        is_new = (
            rouge_new > rouge_threshold
            and rouge_new > rouge_param + rouge_margin
            and not ambiguous
        )
        outputs_param = (
            rouge_param > rouge_threshold
            and rouge_param >= rouge_new
        )
        inst.b1_success = None if ambiguous else is_new
        inst.b1_outputs_param = outputs_param
        inst.b1_ambiguous = ambiguous
        inst.b1_rouge_new = rouge_new
        inst.b1_rouge_param = rouge_param
        log.append({
            "instance_id": inst.instance_id,
            "param_class": inst.param_class,
            "generated": generated,
            "answer_new": row.get("answer_new", ""),
            "a_param": inst.a_param,
            "rougeL_answer_new": rouge_new,
            "rougeL_a_param": rouge_param,
            "rougeL_threshold": rouge_threshold,
            "rougeL_margin": rouge_margin,
            "b1_success": is_new,
            "b1_outputs_param": outputs_param,
            "b1_ambiguous": ambiguous,
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(out_dir / "f3_b1_behavior.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "n": len(log),
                "metric": "rougeL_f1",
                "threshold": rouge_threshold,
                "margin": rouge_margin,
                "log": log,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )


def split_sets(
    instances: list[F3PreparedInstance],
) -> tuple[
    list[F3PreparedInstance],
    list[F3PreparedInstance],
    list[F3PreparedInstance],
    list[F3PreparedInstance],
]:
    conflict = [inst for inst in instances if inst.param_class in {"PARAM_OLD", "PARAM_OTHER"}]
    ambiguous = [inst for inst in conflict if inst.b1_ambiguous]
    failure = [inst for inst in conflict if inst.b1_success is False and not inst.b1_ambiguous]
    success = [inst for inst in conflict if inst.b1_success is True]
    control = [inst for inst in instances if inst.param_class == "PARAM_NEW"]
    return failure, success, control, ambiguous


def choose_run_set(
    failure: list[F3PreparedInstance],
    success: list[F3PreparedInstance],
    control: list[F3PreparedInstance],
    *,
    include_success: bool,
    include_control: bool = True,
) -> list[F3PreparedInstance]:
    run_set = list(failure)
    if include_success:
        run_set.extend(success)
    if include_control:
        run_set.extend(control)
    return run_set


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_f3a(dla_results) -> dict:
    if not dla_results:
        return {"n": 0}
    negative_counts = {
        r.instance_id: len(r.top_negative_late)
        for r in dla_results
    }
    by_class: dict[str, list[int]] = {}
    for result in dla_results:
        by_class.setdefault(result.param_class, []).append(negative_counts[result.instance_id])
    control_counts = by_class.get("PARAM_NEW", [])
    if control_counts:
        sorted_control = sorted(control_counts)
        cutoff_index = min(len(sorted_control) - 1, int(np.ceil(0.95 * len(sorted_control))) - 1)
        k_star = int(sorted_control[cutoff_index] + 1)
        calibration_note = (
            "k_star is the smallest integer above the empirical 95th percentile "
            "of PARAM_NEW negative-late component counts; fewer than about 5% "
            "of controls should satisfy count >= k_star."
        )
    else:
        k_star = None
        calibration_note = (
            "No PARAM_NEW control baseline in this run; specificity claims and "
            "control-calibrated k_star are unavailable."
        )

    class_summary = {
        param_class: {
            "n": len(counts),
            "mean_negative_late_count": float(np.mean(counts)) if counts else 0.0,
            "median_negative_late_count": float(np.median(counts)) if counts else 0.0,
            "p95_negative_late_count": float(np.percentile(counts, 95)) if counts else 0.0,
            "pct_ge_k_star": (
                float(sum(1 for count in counts if count >= k_star) / len(counts))
                if counts and k_star is not None else None
            ),
        }
        for param_class, counts in sorted(by_class.items())
    }
    return {
        "n": len(dla_results),
        "mean_actual_logit_diff": float(np.mean([r.actual_logit_diff for r in dla_results])),
        "mean_residual_error": float(np.mean([r.residual_error for r in dla_results])),
        "median_residual_error": float(np.median([r.residual_error for r in dla_results])),
        "pct_residual_error_lt_0_05": float(
            sum(1 for r in dla_results if r.residual_error < 0.05) / len(dla_results)
        ),
        "negative_late_count_calibration": {
            "k_star": k_star,
            "note": calibration_note,
            "by_param_class": class_summary,
        },
        "per_instance": [
            {
                "instance_id": r.instance_id,
                "param_class": r.param_class,
                "actual_logit_diff": r.actual_logit_diff,
                "total_contribution": r.total_contribution,
                "residual_error": r.residual_error,
                "negative_late_count": negative_counts[r.instance_id],
                "f3_positive_by_k_star": (
                    negative_counts[r.instance_id] >= k_star
                    if k_star is not None else None
                ),
                "top_negative_late": r.top_negative_late,
            }
            for r in dla_results
        ],
    }


def validate_f3c_prompts(
    instances: list[F3PreparedInstance],
    *,
    template: str,
    out_dir: Path,
    n_preview: int = 3,
) -> None:
    """Print and save A1/B1 prompt pairs before F3-c runs."""
    samples: list[dict] = []
    for inst in instances[:n_preview]:
        context = str(inst.row.get("context", ""))
        question = str(inst.row.get("question", ""))
        a1_prompt, b1_prompt = build_f3_pair_prompts(
            context=context,
            question=question,
            template=template,
        )

        if context and context in a1_prompt:
            raise ValueError(f"F3-c A1 prompt unexpectedly contains context for {inst.instance_id}")
        if context and context not in b1_prompt:
            raise ValueError(f"F3-c B1 prompt is missing context for {inst.instance_id}")
        if question not in a1_prompt or question not in b1_prompt:
            raise ValueError(f"F3-c paired prompts are missing the question for {inst.instance_id}")

        samples.append({
            "instance_id": inst.instance_id,
            "template": template,
            "question": question,
            "context_preview": context[:300],
            "a1_prompt": a1_prompt,
            "b1_prompt": b1_prompt,
        })

    with open(out_dir / "f3c_prompt_sanity.json", "w", encoding="utf-8") as fh:
        json.dump({"n": len(samples), "samples": samples}, fh, indent=2, ensure_ascii=False)

    print("\nF3-c prompt sanity check")
    print("A1 and B1 use the same F3 paired prompt builder; B1 adds only Context text.")
    for sample in samples:
        print("\n---", sample["instance_id"], "---")
        print("[A1 prompt]")
        print(sample["a1_prompt"])
        print("[B1 prompt]")
        print(sample["b1_prompt"])


def main() -> None:
    parser = argparse.ArgumentParser(description="F3 Diagnostic — Parametric Override")
    parser.add_argument("--data", required=True, help="Layer-2 JSONL with B1 instances")
    parser.add_argument("--layer3", required=True, help="Layer-3 JSONL with cached parametric answers")
    parser.add_argument("--model", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--template", default="phi3", choices=["plain", "llama2", "llama3", "phi3", "qwen"])
    parser.add_argument("--out", default="results/f3_diagnostic")
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    parser.add_argument("--dtype", default="float32", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("-n", "--number", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=["f3a", "f3b", "f3c", "f3d", "f3e"],
        help="Skip F3 sub-experiments",
    )
    parser.add_argument("--include-success", action="store_true", help="Also run F3-a/b/c on conflict B1-success")
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Do not include PARAM_NEW controls. Specificity/k* calibration will be unavailable.",
    )
    parser.add_argument("--no-b1-behavior", action="store_true", help="Skip B1 generation; run on all conflict instances")
    parser.add_argument(
        "--rouge-threshold",
        type=float,
        default=0.3,
        help="RougeL-F1 threshold for B1 success and a_param output flags (default: 0.3)",
    )
    parser.add_argument(
        "--rouge-margin",
        type=float,
        default=0.1,
        help="Minimum RougeL-F1 margin required to call B1 success over a_param (default: 0.1)",
    )
    parser.add_argument("--f3d-max-components", type=int, default=6)
    parser.add_argument("--run-f3e", action="store_true", help="Run optional F3-e causal tracing")
    parser.add_argument("--f3e-max-instances", type=int, default=20)
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    if args.dtype == "auto":
        dev = args.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        resolved_dtype = torch.float16 if dev == "cuda" else torch.float32
    else:
        resolved_dtype = dtype_map[args.dtype]
    if resolved_dtype is not torch.float32:
        print(
            "[WARNING] F3 mechanistic analysis is designed for float32. "
            "Use non-float32 only for memory-constrained smoke tests."
        )
    set_global_seed(args.sample_seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model : {args.model}")
    print(f"Data  : {args.data}")
    print(f"Layer3: {args.layer3}")
    print(f"Dtype : {resolved_dtype}")
    print(f"Output: {out_dir}")

    b1_instances = load_b1(args.data)
    if args.max_instances:
        b1_instances = b1_instances[:args.max_instances]
    b1_instances = restrict_by_fact(b1_instances, number=args.number, seed=args.sample_seed)

    print(f"\nLoaded B1 instances: {len(b1_instances)}")

    print(f"\nLoading model {args.model} ...")
    model = load_model(args.model, device=args.device, dtype=resolved_dtype)
    model.cfg.use_attn_result = True
    print(f"  {model.cfg.n_layers} layers x {model.cfg.n_heads} heads, d_model={model.cfg.d_model}")

    layer3_by_id, layer3_by_key = load_layer3_by_key(args.layer3)
    prepared = prepare_f3_instances(model, b1_instances, layer3_by_id, layer3_by_key)
    if not prepared:
        raise SystemExit("[ERROR] No F3-ready instances after Layer3/token filtering.")

    classify_b1_behavior(
        model,
        prepared,
        template=args.template,
        out_dir=out_dir,
        rouge_threshold=args.rouge_threshold,
        rouge_margin=args.rouge_margin,
        skip_generation=args.no_b1_behavior,
    )

    failure, success, control, ambiguous = split_sets(prepared)
    if args.no_b1_behavior:
        failure = [inst for inst in prepared if inst.param_class in {"PARAM_OLD", "PARAM_OTHER"}]
        if "f3d" not in args.skip:
            print("[WARNING] F3-d skipped: requires B1 behavioral labels.")
            args.skip = list(args.skip) + ["f3d"]

    run_set = choose_run_set(
        failure, success, control,
        include_success=args.include_success,
        include_control=not args.no_control,
    )
    if not run_set:
        raise SystemExit("[ERROR] No F3 run-set instances. Try --no-b1-behavior or include controls.")

    manifest = {
        "model": args.model,
        "template": args.template,
        "n_b1_loaded": len(b1_instances),
        "n_prepared": len(prepared),
        "n_failure": len(failure),
        "n_success": len(success),
        "n_control": len(control),
        "n_b1_ambiguous": len(ambiguous),
        "n_run_set": len(run_set),
        "include_control": not args.no_control,
        "b1_success_metric": "rougeL_f1",
        "b1_success_threshold": args.rouge_threshold,
        "b1_success_margin": args.rouge_margin,
        "sample_seed": args.sample_seed,
        "deterministic": True,
        "control_warning": (
            None if not args.no_control else
            "No control baseline; F3 specificity claims and k* calibration are unavailable."
        ),
        "skip": args.skip,
    }
    if args.no_control:
        print("[WARNING] Running without PARAM_NEW controls; specificity claims unavailable.")
    with open(out_dir / "f3_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print("\nF3 set sizes:", manifest)

    dla_results = []
    if "f3a" not in args.skip or "f3d" not in args.skip:
        dla_results, f3a_long = run_f3a_dla(model, run_set, template=args.template)
        write_jsonl(out_dir / "f3a_dla_long.jsonl", f3a_long)
        with open(out_dir / "f3a_dla_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summarize_f3a(dla_results), fh, indent=2, ensure_ascii=False)

    if "f3b" not in args.skip:
        rows = run_f3b_ffn_lens(model, run_set, template=args.template, condition="B1")
        write_jsonl(out_dir / "f3b_ffn_lens.jsonl", rows)

    if "f3c" not in args.skip:
        validate_f3c_prompts(run_set, template=args.template, out_dir=out_dir)
        rows = run_f3c_dual_trajectory(model, run_set, template=args.template)
        with open(out_dir / "f3c_dual_trajectory.json", "w", encoding="utf-8") as fh:
            json.dump({"n": len(rows), "per_instance": rows}, fh, indent=2, ensure_ascii=False)

    if "f3d" not in args.skip:
        donor_pool = control or [inst for inst in prepared if inst.param_class == "PARAM_NEW"]
        rows = run_f3d_targeted_patch(
            model,
            failure,
            donor_pool,
            dla_results,
            template=args.template,
            max_components=args.f3d_max_components,
            random_seed=args.sample_seed,
        )
        with open(out_dir / "f3d_patch.json", "w", encoding="utf-8") as fh:
            json.dump({"n": len(rows), "per_instance": rows}, fh, indent=2, ensure_ascii=False)

    if args.run_f3e and "f3e" not in args.skip:
        rows = run_f3e_causal_trace(
            model,
            failure,
            template=args.template,
            max_instances=args.f3e_max_instances,
        )
        write_jsonl(out_dir / "f3e_causal_trace.jsonl", rows)

    print("\nF3 Diagnostic complete. Results saved to:", out_dir)


if __name__ == "__main__":
    main()
