#!/usr/bin/env python3
"""Build Layer-3 parametric answer cache from Layer-2 EvalInstances.

Layer 3 stores model-dependent answers for selected Layer-2 questions.  Unlike
Layer 2, this file is tied to a specific model, template, and decoding setup.

Default usage builds parametric answers from B1 questions without using their
Layer-2 context.  Pass --use-context only when you intentionally want to cache
context-conditioned answers.

    python scripts/build_wikidata_layer3.py \
        --layer2 data/processed/wikidata_layer2_1000.jsonl \
        --model microsoft/phi-3-mini-4k-instruct \
        --template phi3 \
        --out data/processed/wikidata_layer3_phi3_1000.jsonl
"""
from __future__ import annotations

import argparse
import json
import random as _random
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "source"))

from tatm.model import build_prompt, check_match, load_model  # noqa: E402

DEFAULT_LAYER2 = REPO_ROOT / "data/processed/wikidata_layer2.jsonl"
DEFAULT_OUT = REPO_ROOT / "data/processed/wikidata_layer3.jsonl"


def load_layer2(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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

    selected_fact_ids = set(_random.Random(seed).sample(fact_ids, number))
    return [
        record for record in records
        if str(record.get("fact_id") or record.get("instance_id")) in selected_fact_ids
    ]


def _collect_eos_ids(tokenizer) -> list[int]:
    eos: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos.append(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for marker in ("<|end|>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>"):
        if marker in vocab and vocab[marker] not in eos:
            eos.append(vocab[marker])
    return eos


def extract_answer(text: str) -> str:
    """Extract the first short answer phrase from raw generated text."""
    text = text.strip()
    match = re.search(r"(?i)Answer\s*:\s*([^\n\r]+)", text)
    if match:
        text = match.group(1)
    else:
        text = text.split("\n")[0]
    for stop in ("<|", "[", "Instruction"):
        text = text.split(stop)[0]
    text = text.split(".")[0]
    return text.strip()


def generate_raw_answer(model, prompt: str, max_new_tokens: int) -> str:
    tok = model.tokenizer
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    current_ids = enc["input_ids"].to(model.cfg.device)
    eos_ids = set(_collect_eos_ids(tok))
    generated: list[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(current_ids, prepend_bos=False)
            next_id = int(logits[0, -1, :].argmax())
            del logits
            if next_id in eos_ids:
                break
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=current_ids.device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)
            del next_tensor

    return tok.decode(generated, skip_special_tokens=True).strip()


def model_tag(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1].replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--layer2", default=str(DEFAULT_LAYER2), help="Layer-2 JSONL input")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Layer-3 JSONL output")
    parser.add_argument(
        "--model",
        default="microsoft/phi-3-mini-4k-instruct",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--template",
        default="phi3",
        choices=["plain", "llama2", "llama3", "phi3", "qwen"],
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["B1"],
        help="Layer-2 instance prefixes to answer, e.g. B1 A1, or 'all' (default: B1)",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "-n", "--number", type=int, default=None,
        help="Randomly sample N Layer-1 facts and keep all selected Layer-2 instances",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="Use each Layer-2 instance context. Default is question-only parametric answering.",
    )
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "float32", "bfloat16"],
    )
    args = parser.parse_args()

    layer2_path = Path(args.layer2)
    out_path = Path(args.out)
    if not layer2_path.exists():
        raise SystemExit(f"[ERROR] Layer-2 not found: {layer2_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if args.dtype == "auto":
        dev = args.device
        if dev == "auto":
            dev = (
                "cuda" if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        resolved_dtype = torch.float16 if dev == "cuda" else torch.float32
    else:
        resolved_dtype = dtype_map[args.dtype]

    records = select_instances(
        load_layer2(layer2_path),
        layers=args.layers,
        max_instances=args.max_instances,
        number=args.number,
        seed=args.sample_seed,
    )
    print(f"Layer-2 input : {layer2_path}")
    print(f"Layer-3 output: {out_path}")
    print(f"Model         : {args.model}")
    print(f"Template      : {args.template}")
    print(f"Use context   : {args.use_context}")
    print(f"Selected rows : {len(records)}")

    print(f"\nLoading model {args.model} ...")
    model = load_model(args.model, device=args.device, dtype=resolved_dtype)

    tag = model_tag(args.model)
    with open(out_path, "w", encoding="utf-8") as fh:
        for record in tqdm(records, desc="Generating Layer-3", unit="inst", dynamic_ncols=True):
            prompt_context = record.get("context", "") if args.use_context else ""
            prompt = build_prompt(
                context=prompt_context,
                question=record.get("question", ""),
                template=args.template,
            )
            raw = generate_raw_answer(model, prompt, max_new_tokens=args.max_new_tokens)
            extracted = extract_answer(raw)
            output = {
                "layer3_id": f"L3_{tag}_{record.get('instance_id', '')}",
                "instance_id": record.get("instance_id", ""),
                "fact_id": record.get("fact_id", ""),
                "layer2_type": str(record.get("instance_id", ""))[:2],
                "model": args.model,
                "template": args.template,
                "question": record.get("question", ""),
                "uses_context": args.use_context,
                "context": prompt_context,
                "prompt": prompt,
                "model_output_raw": raw,
                "extracted_answer": extracted,
                "answer_old": record.get("answer_old", ""),
                "answer_new": record.get("answer_new", ""),
                "matches_answer_old": check_match(extracted, record.get("answer_old", "")),
                "matches_answer_new": check_match(extracted, record.get("answer_new", "")),
                "subject_qid": record.get("subject_qid", ""),
                "subject_label": record.get("subject_label", ""),
                "property_pid": record.get("property_pid", ""),
                "property_label": record.get("property_label", ""),
                "t_old": record.get("t_old"),
                "t_new": record.get("t_new"),
                "decoding": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": 0.0,
                    "method": "greedy_argmax",
                },
            }
            fh.write(json.dumps(output, ensure_ascii=False) + "\n")
            fh.flush()

    print(f"\n[OK] Layer-3 written to {out_path}")


if __name__ == "__main__":
    main()
