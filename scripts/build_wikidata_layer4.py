#!/usr/bin/env python3
"""Build Layer-4 behavior cache from Layer-2 EvalInstances.

Layer 4 stores model-dependent, context-conditioned generated answers and
behavior labels.  It is intended to cache expensive B1/B5/B6 generation so F3
can read behavior labels from data instead of generating during diagnosis.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "source"))

from tatm.model import build_prompt, check_match, generate_answer, load_model  # noqa: E402

DEFAULT_LAYER2 = REPO_ROOT / "data/processed/wikidata_layer2.jsonl"
DEFAULT_LAYER3 = REPO_ROOT / "data/processed/wikidata_layer3.jsonl"
DEFAULT_OUT = REPO_ROOT / "data/processed/wikidata_layer4.jsonl"


def layer_key(row: dict) -> tuple[str, Any, Any]:
    return (str(row.get("fact_id", "")), row.get("t_old"), row.get("t_new"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_layer3(path: Path) -> tuple[dict[str, dict], dict[tuple[str, Any, Any], dict]]:
    by_id: dict[str, dict] = {}
    by_key: dict[tuple[str, Any, Any], dict] = {}
    for row in load_jsonl(path):
        iid = str(row.get("instance_id", ""))
        if iid:
            by_id[iid] = row
        key = layer_key(row)
        if key[0]:
            by_key[key] = row
    return by_id, by_key


def select_instances(
    records: list[dict],
    *,
    layers: list[str],
    max_instances: int | None,
    number: int | None,
    seed: int,
) -> list[dict]:
    if layers != ["all"]:
        keep_layers = set(layers)
        records = [
            record for record in records
            if str(record.get("instance_id", ""))[:2] in keep_layers
        ]
    if max_instances is not None:
        records = records[:max_instances]
    if number is None:
        return records
    if number < 1:
        raise ValueError("--number must be >= 1")

    fact_ids = sorted({
        str(record.get("fact_id") or record.get("instance_id"))
        for record in records
    })
    if number >= len(fact_ids):
        return records
    selected = set(random.Random(seed).sample(fact_ids, number))
    return [
        record for record in records
        if str(record.get("fact_id") or record.get("instance_id")) in selected
    ]


def _rouge_tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref = _rouge_tokens(reference)
    pred = _rouge_tokens(prediction)
    if not ref or not pred:
        return 0.0
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


def classify_behavior(
    *,
    generated: str,
    answer_new: str,
    a_param: str,
    threshold: float,
    margin: float,
) -> dict:
    rouge_new = rouge_l_f1(answer_new, generated)
    rouge_param = rouge_l_f1(a_param, generated)
    # ``success`` (did the model output the new, correct-for-year answer?) and
    # ``outputs_param`` (did it revert to its parametric answer?) are ORTHOGONAL
    # axes.  When a_param == answer_new (PARAM_NEW / KNOWS_NEW) the new-vs-param
    # comparison is degenerate (rouge_new == rouge_param), so the old clause
    # ``rouge_new > rouge_param + margin`` wrongly scored a *correct* answer as a
    # failure.  Gate the new-vs-param disambiguation on a_param != answer_new.
    param_eq_new = check_match(a_param, answer_new)
    if param_eq_new:
        success = rouge_new > threshold
        outputs_param = False
        ambiguous = False
    else:
        both_pass = rouge_new > threshold and rouge_param > threshold
        ambiguous = both_pass and abs(rouge_new - rouge_param) < margin
        success = (
            rouge_new > threshold
            and rouge_new > rouge_param + margin
            and not ambiguous
        )
        outputs_param = rouge_param > threshold and rouge_param >= rouge_new
    return {
        "rougeL_answer_new": rouge_new,
        "rougeL_a_param": rouge_param,
        "rougeL_threshold": threshold,
        "rougeL_margin": margin,
        "b_success": success,
        "b_outputs_param": outputs_param,
        "b_ambiguous": ambiguous,
    }


def classify_parametric_answer(layer3_row: dict, answer_old: str, answer_new: str) -> str:
    answer = str(layer3_row.get("extracted_answer") or layer3_row.get("model_output_raw") or "")
    if layer3_row.get("matches_answer_new") or check_match(answer, answer_new):
        return "PARAM_NEW"
    if layer3_row.get("matches_answer_old") or check_match(answer, answer_old):
        return "PARAM_OLD"
    return "PARAM_OTHER"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer2", default=str(DEFAULT_LAYER2))
    parser.add_argument("--layer3", default=str(DEFAULT_LAYER3))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--model", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--template", default="phi3", choices=["plain", "llama2", "llama3", "mistral", "phi3", "qwen"])
    parser.add_argument("--layers", nargs="+", default=["B1"], help="Layer-2 prefixes to generate, e.g. B1 B5 B6")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("-n", "--number", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--rouge-threshold", type=float, default=0.3)
    parser.add_argument("--rouge-margin", type=float, default=0.1)
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    args = parser.parse_args()

    layer2_path = Path(args.layer2)
    layer3_path = Path(args.layer3)
    out_path = Path(args.out)
    if not layer2_path.exists():
        raise SystemExit(f"[ERROR] Layer-2 not found: {layer2_path}")
    if not layer3_path.exists():
        raise SystemExit(f"[ERROR] Layer-3 not found: {layer3_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    if args.dtype == "auto":
        dev = args.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        resolved_dtype = torch.float16 if dev == "cuda" else torch.float32
    else:
        resolved_dtype = dtype_map[args.dtype]

    layer3_by_id, layer3_by_key = load_layer3(layer3_path)
    records = select_instances(
        load_jsonl(layer2_path),
        layers=args.layers,
        max_instances=args.max_instances,
        number=args.number,
        seed=args.sample_seed,
    )

    print(f"Layer-2 input : {layer2_path}")
    print(f"Layer-3 input : {layer3_path}")
    print(f"Layer-4 output: {out_path}")
    print(f"Layers        : {args.layers}")
    print(f"Selected rows : {len(records)}")

    print(f"\nLoading model {args.model} ...")
    model = load_model(args.model, device=args.device, dtype=resolved_dtype)

    missing_layer3 = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in tqdm(records, desc="Generating Layer-4", unit="inst", dynamic_ncols=True):
            layer3 = layer3_by_id.get(str(record.get("instance_id", "")))
            if layer3 is None:
                layer3 = layer3_by_key.get(layer_key(record))
            if layer3 is None:
                missing_layer3 += 1
                continue

            a_param = str(layer3.get("extracted_answer") or layer3.get("model_output_raw") or "")
            prompt = build_prompt(
                str(record.get("context", "")),
                str(record.get("question", "")),
                template=args.template,
            )
            generated = generate_answer(model, prompt)
            behavior = classify_behavior(
                generated=generated,
                answer_new=str(record.get("answer_new", "")),
                a_param=a_param,
                threshold=args.rouge_threshold,
                margin=args.rouge_margin,
            )
            output = {
                "layer4_id": f"L4_{record.get('instance_id', '')}",
                "instance_id": record.get("instance_id", ""),
                "fact_id": record.get("fact_id", ""),
                "layer2_type": str(record.get("instance_id", ""))[:2],
                "model": args.model,
                "template": args.template,
                "uses_context": bool(str(record.get("context", "")).strip()),
                "question": record.get("question", ""),
                "context": record.get("context", ""),
                "prompt": prompt,
                "model_output_raw": generated,
                "generated": generated,
                "answer_old": record.get("answer_old", ""),
                "answer_new": record.get("answer_new", ""),
                "a_param": a_param,
                "param_class": classify_parametric_answer(
                    layer3,
                    str(record.get("answer_old", "")),
                    str(record.get("answer_new", "")),
                ),
                **behavior,
            }
            fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            fh.flush()

    if missing_layer3:
        print(f"[WARNING] skipped {missing_layer3} rows missing matching Layer-3 entries")
    print(f"\n[OK] Layer-4 written to {out_path}")


if __name__ == "__main__":
    main()
