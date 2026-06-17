#!/usr/bin/env python3
"""Headless EAP-IG temporal-head discovery (cluster-runnable).

Self-contained port of the TemporalHead notebooks
(``1.circuit_construction.ipynb`` + ``2.circuit_analysis.ipynb``) so temporal
heads can be discovered for any model via ``sbatch``. The EAP-IG attribution
package lives in ``code/source/eap`` and the circuit datasets in
``code/data/temporal_circuits/{Temporal,Invariant}``.

Pipeline (per model):
  1. Load the model with the EAP hook flags.
  2. For each (temporal category, target year) and each invariant category,
     build a clean/corrupted fact-retrieval dataset, run EAP-IG attribution,
     threshold-simplify the circuit, and dump the surviving node set plus a
     continuous per-head importance score.
  3. Rank attention heads by temporal relevance =
       mean(normalized score over temporal circuits)
       - mean(normalized score over invariant circuits)
     and write the top-K to ``<out>/<model_name>/discovered_temporal_heads.json``
     (same schema as ``data/external/temporal_heads/paper_temporal_heads.json``).

Run from the ``code/`` directory (the cluster working dir).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from random import Random

import pandas as pd

# code/source on PYTHONPATH so ``import eap`` resolves (mirrors run_f1_diagnostic)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "source"))

import torch  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

from eap.metrics import logit_diff  # noqa: E402
from eap.graph import Graph  # noqa: E402
from eap.dataset import EAPDataset  # noqa: E402
from eap.attribute import attribute  # noqa: E402

HEAD_RE = re.compile(r"^a(\d+)\.h(\d+)$")
YEAR_RE = re.compile(r"(\d{4})")


def _assert_eap_attribute_bf16_safe(dtype: str) -> None:
    """Fail before model loading if the cluster imported an old EAP checkout."""
    src_path = inspect.getsourcefile(attribute)
    print(f"[preflight] eap.attribute source: {src_path}")
    if dtype != "bfloat16":
        return
    try:
        src = inspect.getsource(attribute)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot inspect eap.attribute at {src_path}; refusing bf16 EAP run"
        ) from exc
    if ".float().cpu().numpy()" not in src:
        raise RuntimeError(
            "The imported eap.attribute is not bf16-safe. It still appears to "
            "convert scores with scores.cpu().numpy(), which fails for "
            "bfloat16 tensors. Sync source/eap/attribute.py so attribute() uses "
            "scores.float().cpu().numpy(), or rerun with DTYPE=float16/float32."
        )


# ──────────────────────────── dataset construction ──────────────────────────
def _find_category_file(directory: Path, category: str) -> Path | None:
    if not directory.is_dir():
        return None
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(".json") and category in fn:
            return directory / fn
    return None


def _same_len(model, a: str, b: str) -> bool:
    return len(model.to_str_tokens(a)) == len(model.to_str_tokens(b))


def build_temporal_rows(model, cat_file: Path, year: str, obj_idx: int,
                        max_pairs: int, rng: Random) -> list[dict]:
    data = json.load(open(cat_file))
    tmpl = data["prompt_templates"][0]
    samples = data["samples"]
    rows: list[dict] = []
    for s in samples:
        if s["time"] != year:
            continue
        subject, obj_clean = s["subject"], s["object"]
        cands = [c for c in samples
                 if c["subject"] == subject and c["time"] != year
                 and c["object"] != obj_clean]
        rng.shuffle(cands)
        for c in cands:
            clean = tmpl.format(time=year, subject=subject)
            corrupted = tmpl.format(time=c["time"], subject=subject)
            if not _same_len(model, clean, corrupted):
                continue
            ci = model.tokenizer(obj_clean, add_special_tokens=False).input_ids
            cci = model.tokenizer(c["object"], add_special_tokens=False).input_ids
            if len(ci) <= obj_idx or len(cci) <= obj_idx:
                continue
            rows.append({"clean": clean, "corrupted": corrupted,
                         "country_idx": ci[obj_idx],
                         "corrupted_country_idx": cci[obj_idx]})
            if len(rows) >= max_pairs:
                return rows
    return rows


def build_invariant_rows(model, cat_file: Path, obj_idx: int,
                         max_pairs: int, rng: Random) -> list[dict]:
    data = json.load(open(cat_file))
    tmpl = data["prompt_templates"][0]
    samples = data["samples"]
    rows: list[dict] = []
    for s in samples:
        subject, obj_clean = s["subject"], s["object"]
        cands = [c for c in samples if c["object"] != obj_clean]
        rng.shuffle(cands)
        for c in cands:
            clean = tmpl.format(subject=subject)
            corrupted = tmpl.format(subject=c["subject"])
            if not _same_len(model, clean, corrupted):
                continue
            ci = model.tokenizer(obj_clean, add_special_tokens=False).input_ids
            cci = model.tokenizer(c["object"], add_special_tokens=False).input_ids
            if len(ci) <= obj_idx or len(cci) <= obj_idx:
                continue
            rows.append({"clean": clean, "corrupted": corrupted,
                         "country_idx": ci[obj_idx],
                         "corrupted_country_idx": cci[obj_idx]})
            if len(rows) >= max_pairs:
                return rows
    return rows


# ──────────────────────────── circuit per dataset ───────────────────────────
def run_one_circuit(model, rows: list[dict], tag: str, *, out_dir: Path,
                    tmp_dir: Path, ig_steps: int, threshold: float,
                    max_seq_len: int | None = None) -> int:
    """Attribute one dataset, threshold-simplify, dump surviving nodes + per-head
    scores. Returns the number of surviving nodes (0 ⇒ skipped)."""
    if len(rows) < 2:
        print(f"  [skip] {tag}: only {len(rows)} pairs")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / f"{tag}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    dataset = EAPDataset(filename=str(csv_path), task="fact-retrieval")
    dataloader = dataset.to_dataloader(batch_size=1)

    g = Graph.from_model(model)
    attribute(model, g, dataloader,
              partial(logit_diff, loss=True, mean=True),
              method="EAP-IG", ig_steps=ig_steps, max_seq_len=max_seq_len)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # threshold-simplified node set (notebook cell 10 logic): a node survives if
    # it is incident to an edge whose |score| clears the threshold.
    g.apply_threshold(threshold=threshold, absolute=False)
    important: set[str] = set()
    for edge in g.edges.values():
        if edge.in_graph:
            important.add(edge.parent.name)
            important.add(edge.child.name)

    # Continuous per-head importance (NOT just binary membership): sum of |score|
    # over every edge whose PARENT is an attention head = that head's total
    # downstream causal contribution in this circuit. Lets us rank heads later.
    head_scores: dict[str, float] = defaultdict(float)
    for edge in g.edges.values():
        name = edge.parent.name
        if HEAD_RE.match(name) and edge.score is not None:
            head_scores[name] += abs(float(edge.score))

    out_path = out_dir / f"simplified_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump({"nodes": {n: {} for n in sorted(important)},
                   "head_scores": dict(head_scores)}, fh, indent=1)
    print(f"  [ok] {tag}: {len(rows)} pairs -> {len(important)} nodes, "
          f"{len(head_scores)} scored heads")
    return len(important)


# ──────────────────────────── head aggregation ──────────────────────────────
def _category_of(filename: str) -> str:
    m = YEAR_RE.search(filename)
    return f"year{m.group(1)}" if m else "time_invariant"


def _strict_temporal_only(graph_dir: Path, threshold_ratio: float) -> set[str]:
    """Notebook-2 logic: nodes 'major' (>= ratio of files) in the temporal
    year-categories but not in the invariant category. Strict binary set."""
    cat_files: dict[str, list[set]] = defaultdict(list)
    for fn in os.listdir(graph_dir):
        if not (fn.startswith("simplified_") and fn.endswith(".json")):
            continue
        nodes = set(json.load(open(graph_dir / fn)).get("nodes", {}).keys())
        cat_files[_category_of(fn)].append(nodes)

    def major(file_sets: list[set]) -> set:
        cnt = Counter()
        for ns in file_sets:
            cnt.update(ns)
        need = int(threshold_ratio * len(file_sets) + 0.9999999)
        return {n for n, c in cnt.items() if c >= need}

    temporal_major, invariant_major = set(), set()
    for cat, file_sets in cat_files.items():
        (temporal_major if cat.startswith("year") else invariant_major).update(
            major(file_sets))
    return temporal_major - invariant_major


def rank_temporal_heads(graph_dir: Path, top_k: int,
                        threshold_ratio: float) -> list[dict]:
    """Rank attention heads by temporal relevance and return the top-K.

    Per circuit, head scores are normalized to [0, 1] (divide by the circuit's
    max head score) so circuits with different score scales contribute equally.
    A head's relevance = mean(normalized score over temporal circuits) -
    mean(over invariant circuits). Graded list (>= top_k heads), unlike the
    strict binary temporal-only set."""
    temporal: list[dict] = []
    invariant: list[dict] = []
    for fn in os.listdir(graph_dir):
        if not (fn.startswith("simplified_") and fn.endswith(".json")):
            continue
        hs = json.load(open(graph_dir / fn)).get("head_scores", {})
        mx = max(hs.values()) if hs else 0.0
        norm = {h: (v / mx if mx > 0 else 0.0) for h, v in hs.items()}
        (temporal if _category_of(fn).startswith("year") else invariant).append(norm)

    def mean_score(files: list[dict], head: str) -> float:
        return sum(f.get(head, 0.0) for f in files) / len(files) if files else 0.0

    def freq(files: list[dict], head: str) -> float:
        return (sum(1 for f in files if f.get(head, 0.0) > 0) / len(files)
                if files else 0.0)

    strict = _strict_temporal_only(graph_dir, threshold_ratio)
    heads = {h for f in temporal for h in f} | {h for f in invariant for h in f}
    ranked = []
    for h in heads:
        m = HEAD_RE.match(h)
        if not m:
            continue
        t, inv = mean_score(temporal, h), mean_score(invariant, h)
        ranked.append({
            "layer": int(m.group(1)), "head": int(m.group(2)), "name": h,
            "relevance": round(t - inv, 5),
            "temporal_score": round(t, 5),
            "invariant_score": round(inv, 5),
            "temporal_freq": round(freq(temporal, h), 3),
            "strict_temporal_only": h in strict,
        })
    ranked.sort(key=lambda d: (d["relevance"], d["temporal_score"]), reverse=True)
    return ranked[:top_k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF path for HookedTransformer")
    ap.add_argument("--model-name", required=True, help="tag for the output dir")
    ap.add_argument("--obj-token-idx", type=int, default=0,
                    help="object first-token index (Phi tokenizers need 1)")
    ap.add_argument("--data-dir", default=str(_ROOT / "data" / "temporal_circuits"),
                    help="dir containing Temporal/ and Invariant/ subdirs")
    ap.add_argument("--out-root", default=str(_ROOT / "results" / "eap_circuits"))
    ap.add_argument("--years", nargs="+", default=["1999", "2004", "2009"])
    ap.add_argument("--temporal-categories", nargs="+",
                    default=["time_sports", "time_presidents", "time_ceo",
                             "time_defense", "time_movies", "time_gdp",
                             "time_inflations"])
    ap.add_argument("--invariant-categories", nargs="+",
                    default=["fruit_inside_color", "object_superclass",
                             "geometric_shape", "roman_numerals"])
    ap.add_argument("--ig-steps", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--max-pairs", type=int, default=20)
    ap.add_argument("--max-seq-len", type=int, default=256,
                    help="truncate tokenized prompts to this length (saves VRAM)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"],
                    help="model weight/activation dtype (bfloat16 recommended for 7B)")
    ap.add_argument("--top-k", type=int, default=8,
                    help="number of top temporal heads to report (>=5 recommended)")
    ap.add_argument("--threshold-ratio", type=float, default=0.9)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-split-qkv", action="store_true",
                    help="disable use_split_qkv_input (try if a GQA model errors)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    _assert_eap_attribute_bf16_safe(args.dtype)

    rng = Random(args.seed)
    data_dir = Path(args.data_dir)
    temporal_dir = data_dir / "Temporal"
    invariant_dir = data_dir / "Invariant"
    out_dir = Path(args.out_root) / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_csv"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model}  dtype={args.dtype}  max_seq_len={args.max_seq_len}")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model = HookedTransformer.from_pretrained(
        args.model, device=args.device,
        dtype=dtype_map[args.dtype],
        fold_ln=False, center_writing_weights=False, center_unembed=False)
    model.cfg.use_split_qkv_input = not args.no_split_qkv
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.default_prepend_bos = False
    n_kv = getattr(model.cfg, "n_key_value_heads", None)
    if n_kv is not None and n_kv != model.cfg.n_heads:
        print(f"[info] GQA model (n_kv={n_kv}, n_heads={model.cfg.n_heads}); "
              f"use_split_qkv_input={model.cfg.use_split_qkv_input}. "
              f"If attribution errors, rerun with --no-split-qkv.")

    t0 = time.time()
    for cat in args.temporal_categories:
        cf = _find_category_file(temporal_dir, cat)
        if cf is None:
            print(f"[warn] temporal category file not found: {cat}")
            continue
        for year in args.years:
            tag = f"{cat}_{year}"
            if args.skip_existing and (out_dir / f"simplified_{tag}.json").exists():
                print(f"  [have] {tag}")
                continue
            rows = build_temporal_rows(model, cf, year, args.obj_token_idx,
                                       args.max_pairs, rng)
            run_one_circuit(model, rows, tag, out_dir=out_dir, tmp_dir=tmp_dir,
                            ig_steps=args.ig_steps, threshold=args.threshold,
                            max_seq_len=args.max_seq_len)

    # invariant circuits (control) — tag must NOT contain a 4-digit year
    for cat in args.invariant_categories:
        cf = _find_category_file(invariant_dir, cat)
        if cf is None:
            print(f"[warn] invariant category file not found: {cat}")
            continue
        tag = f"inv_{cat}"
        if args.skip_existing and (out_dir / f"simplified_{tag}.json").exists():
            print(f"  [have] {tag}")
            continue
        rows = build_invariant_rows(model, cf, args.obj_token_idx,
                                    args.max_pairs, rng)
        run_one_circuit(model, rows, tag, out_dir=out_dir, tmp_dir=tmp_dir,
                        ig_steps=args.ig_steps, threshold=args.threshold,
                        max_seq_len=args.max_seq_len)

    # aggregate -> RANKED top-K temporal heads (graded, not a binary set)
    ranked = rank_temporal_heads(out_dir, args.top_k, args.threshold_ratio)
    top_heads = [{"layer": h["layer"], "head": h["head"], "coef": h["relevance"],
                  "temporal_score": h["temporal_score"],
                  "invariant_score": h["invariant_score"],
                  "temporal_freq": h["temporal_freq"],
                  "strict_temporal_only": h["strict_temporal_only"]}
                 for h in ranked]
    result = {
        "source": "EAP-IG self-discovered (scripts/run_eap_circuit.py)",
        "method": ("EAP-IG circuit; heads ranked by temporal relevance = "
                   "mean(normalized score over temporal circuits) - "
                   "mean(over invariant circuits)"),
        "config": {"years": args.years, "ig_steps": args.ig_steps,
                   "threshold": args.threshold,
                   "threshold_ratio": args.threshold_ratio,
                   "max_pairs": args.max_pairs, "top_k": args.top_k},
        "models": {args.model_name: {"model": args.model, "top_heads": top_heads}},
    }
    out_json = out_dir / "discovered_temporal_heads.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as fh:
        json.dump(result, fh, indent=2)
    print("\n[done] {:.0f}s".format(time.time() - t0))
    print("[heads] top-{} temporal heads:".format(args.top_k))
    for h in ranked:
        print("  a{}.h{}  rel={:+.4f}  t={:.4f}  inv={:.4f}  freq={:.2f}  {}"
              .format(h["layer"], h["head"], h["relevance"], h["temporal_score"],
                      h["invariant_score"], h["temporal_freq"],
                      "TEMPORAL-ONLY" if h["strict_temporal_only"] else ""))
    print("[saved] {}".format(out_json))


if __name__ == "__main__":
    main()
