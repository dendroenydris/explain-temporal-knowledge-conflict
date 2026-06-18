#!/usr/bin/env python3
"""Re-extract Layer-3 answers from existing `model_output_raw`.

This script does NOT run model inference. It only re-parses `model_output_raw`
using the latest extraction logic in `scripts/build_wikidata_layer3.py`, then
updates:
  - extracted_answer
  - matches_answer_old
  - matches_answer_new

By default it rewrites the input file in-place with a backup.

Example:
    python scripts/reextract_layer3_answers.py \
      --input data/processed/wikidata_layer3_mistral_1000.jsonl
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_extract_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("build_wikidata_layer3", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rewrite(
    input_path: Path,
    output_path: Path,
    *,
    extract_answer,
    check_match,
) -> dict:
    n_total = 0
    n_changed = 0
    n_empty = 0
    n_invalid = 0
    n_match_old = 0
    n_match_new = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_total += 1
            try:
                rec = json.loads(line)
            except Exception:
                n_invalid += 1
                continue

            old_extracted = rec.get("extracted_answer", "")
            raw = rec.get("model_output_raw", "") or ""
            candidates = []
            if rec.get("answer_old"):
                candidates.append(str(rec["answer_old"]))
            if rec.get("answer_new"):
                candidates.append(str(rec["answer_new"]))

            new_extracted = extract_answer(raw, candidates=candidates)
            rec["extracted_answer"] = new_extracted
            rec["matches_answer_old"] = check_match(new_extracted, rec.get("answer_old", ""))
            rec["matches_answer_new"] = check_match(new_extracted, rec.get("answer_new", ""))

            if new_extracted != old_extracted:
                n_changed += 1
            if not new_extracted:
                n_empty += 1
            if rec["matches_answer_old"]:
                n_match_old += 1
            if rec["matches_answer_new"]:
                n_match_new += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "n_total": n_total,
        "n_changed": n_changed,
        "n_empty": n_empty,
        "n_invalid": n_invalid,
        "n_match_old": n_match_old,
        "n_match_new": n_match_new,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Input Layer-3 JSONL path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default: in-place rewrite of --input.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak_reextract",
        help="Backup suffix used only for in-place rewrite.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "build_wikidata_layer3.py"
    layer3_mod = _load_extract_module(script_path)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = repo_root / input_path
    if not input_path.exists():
        raise SystemExit(f"[ERROR] Input not found: {input_path}")

    in_place = args.output is None
    if in_place:
        tmp_path = input_path.with_suffix(input_path.suffix + ".tmp_reextract")
        backup_path = input_path.with_suffix(input_path.suffix + args.backup_suffix)
        stats = _rewrite(
            input_path,
            tmp_path,
            extract_answer=layer3_mod.extract_answer,
            check_match=layer3_mod.check_match,
        )
        if backup_path.exists():
            backup_path.unlink()
        input_path.replace(backup_path)
        tmp_path.replace(input_path)
        out_path = input_path
        print(f"[OK] backup: {backup_path}")
    else:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        stats = _rewrite(
            input_path,
            out_path,
            extract_answer=layer3_mod.extract_answer,
            check_match=layer3_mod.check_match,
        )

    print(f"[OK] output: {out_path}")
    print(
        "[STAT] total={n_total} changed={n_changed} empty={n_empty} invalid={n_invalid} "
        "match_old={n_match_old} match_new={n_match_new}".format(**stats)
    )


if __name__ == "__main__":
    main()
