#!/usr/bin/env python3
"""Run F3 diagnosis — methodology section *F3 Diagnosis* (L364–L728).

Phases executed in order:

1.  Load Layer-2 B1 + Layer-3 (cached A1 parametric answers), prepare
    instances, label B1 behaviour (from Layer-4 if available, otherwise
    by RougeL match on a generated answer).
2.  **F3-a** — Logit Lens trajectory + sublayer Δ decomposition.  Emits
    per-instance trajectories and population-level positive-control
    summary.
3.  Partition B1-success pool into A / B / C (frozen after F3-a labels).
4.  **F3-0.5** — Attribution patching on Partition A (Stage 1) +
    B1-failure F3-trajectory PARAM_OLD/PARAM_OTHER panels (Stage 2).
    Selects routing set ``R`` plus panel-asymmetry verdict.
5.  **F3-b** — Dual (M)/(Z) ablation of ``R`` / ``H_T`` / random
    baselines.  Records specificity ratios with bootstrap CIs and the
    decision matrix verdict.
6.  **F3-c Step 1** — Identify ``L^*_σ`` on the selection split (S) of
    B1-failure F3-trajectory PARAM_OLD ∪ PARAM_OTHER per σ ∈ {attn, mlp}.
7.  **F3-c Steps 2–3** — Run the 2×2 Override + Chain on the test split T,
    per σ and per panel.
8.  **F3-c Step 4** — Raw-``W_U`` projection of sublayer updates with
    closed-book / random-late / random-mid baselines, per σ and per panel.
9.  Combined F3 verdict (title-resolution policy at methodology L367–376).

Outputs (under ``--out``):

```
f3_manifest.json                    # configuration + partition sizes
f3_b1_behavior.json                 # B1 generation log (when not from Layer-4)
f3a_trajectory.json                 # per-instance F3-a + summary
f3a_partition.json                  # Partition A/B/C assignments
f3_half_attribution.json            # F3-0.5 routing-set selection
f3b_ablation.json                   # F3-b dual (M)/(Z) outcomes
f3c_step1_l_star.json               # L^*_σ per arm + selector / robustness
f3c_step2_3_<sigma>_<panel>.json    # F3-c 2x2 outcomes per (σ, panel)
f3c_step4_<sigma>_<panel>.json      # F3-c Step-4 Content outcomes per (σ, panel)
f3_verdict.json                     # title-resolved final verdict
```

Cross-experiment inputs (optional):

  --temporal-heads <list>  (l,h) pairs from F1-b for $\\mathcal{H}_T$
  --f1b-results            F1-b JSON with population-level p-values
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "source"))

from tatm.f3_diagnosis import (  # noqa: E402
    F3PreparedInstance,
    assign_f3_verdict,
    build_f3_pair_prompts,
    load_layer3_by_key,
    partition_b1_success_pool,
    prepare_f3_instances,
    run_f3a_trajectory,
    run_f3_half_bridge,
    run_f3b_ablation,
    run_f3c_step1_l_star,
    run_f3c_step2_3,
    run_f3c_step4_content,
    summarize_f3a_population,
    to_jsonable,
)
from tatm.model import build_prompt, generate_answer, load_model  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═════════════════════════════════════════════════════════════════════════════


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═════════════════════════════════════════════════════════════════════════════
# Layer-2 / Layer-4 IO
# ═════════════════════════════════════════════════════════════════════════════


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


def restrict_by_fact(instances: list[dict], *, number: int | None, seed: int) -> list[dict]:
    if number is None:
        return instances
    if number < 1:
        raise ValueError("--number must be >= 1")
    fact_ids = sorted({str(row.get("fact_id") or row.get("instance_id")) for row in instances})
    if number >= len(fact_ids):
        return instances
    selected = set(random.Random(seed).sample(fact_ids, number))
    return [row for row in instances if str(row.get("fact_id") or row.get("instance_id")) in selected]


def _layer_key(row: dict) -> tuple[str, object, object]:
    return (str(row.get("fact_id", "")), row.get("t_old"), row.get("t_new"))


def load_layer4(path: str) -> tuple[dict[str, dict], dict[tuple, dict]]:
    by_id: dict[str, dict] = {}
    by_key: dict[tuple, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("layer2_type", "")) != "B1":
                continue
            iid = str(row.get("instance_id", ""))
            if iid:
                by_id[iid] = row
            key = _layer_key(row)
            if key[0]:
                by_key[key] = row
    return by_id, by_key


def apply_layer4_behavior(
    instances: list[F3PreparedInstance],
    *,
    layer4_by_id: dict[str, dict],
    layer4_by_key: dict[tuple, dict],
    out_dir: Path,
) -> None:
    log: list[dict] = []
    missing: list[str] = []
    for inst in instances:
        row = layer4_by_id.get(inst.instance_id) or layer4_by_key.get(_layer_key(inst.row))
        if row is None:
            missing.append(inst.instance_id)
            continue
        inst.b1_success = row.get("b_success")
        inst.b1_outputs_param = row.get("b_outputs_param")
        inst.b1_ambiguous = bool(row.get("b_ambiguous", False))
        inst.b1_rouge_new = row.get("rougeL_answer_new")
        inst.b1_rouge_param = row.get("rougeL_a_param")
        log.append({
            "instance_id": inst.instance_id,
            "param_class": inst.param_class,
            "b1_success": inst.b1_success,
            "b1_outputs_param": inst.b1_outputs_param,
            "rougeL_answer_new": inst.b1_rouge_new,
            "rougeL_a_param":   inst.b1_rouge_param,
        })

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Layer-4 is missing {len(missing)} B1 behavior rows required by F3. "
            f"Examples: {preview}. Rebuild Layer-4 with LAYERS=B1."
        )

    with open(out_dir / "f3_b1_behavior.json", "w", encoding="utf-8") as fh:
        json.dump({
            "n": len(log), "source": "layer4", "metric": "rougeL_f1", "log": log,
        }, fh, indent=2, ensure_ascii=False)


# ── Local B1 generation fallback (when --layer4 not provided) ───────────────


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
    log: list[dict] = []
    if skip_generation:
        for inst in instances:
            inst.b1_success = None
        return
    for inst in tqdm(instances, desc="F3 B1 behavior", unit="inst", dynamic_ncols=True):
        row = inst.row
        prompt = build_prompt(str(row.get("context", "")),
                              str(row.get("question", "")),
                              template=template)
        generated = generate_answer(model, prompt)
        rouge_new = rouge_l_f1(str(row.get("answer_new", "")), generated)
        rouge_param = rouge_l_f1(inst.a_param, generated)
        both_pass = rouge_new > rouge_threshold and rouge_param > rouge_threshold
        ambiguous = both_pass and abs(rouge_new - rouge_param) < rouge_margin
        is_new = (rouge_new > rouge_threshold
                  and rouge_new > rouge_param + rouge_margin
                  and not ambiguous)
        outputs_param = (rouge_param > rouge_threshold
                         and rouge_param >= rouge_new)
        inst.b1_success = None if ambiguous else is_new
        inst.b1_outputs_param = outputs_param
        inst.b1_ambiguous = ambiguous
        inst.b1_rouge_new = rouge_new
        inst.b1_rouge_param = rouge_param
        log.append({
            "instance_id": inst.instance_id, "param_class": inst.param_class,
            "generated": generated, "answer_new": row.get("answer_new", ""),
            "a_param": inst.a_param, "rougeL_answer_new": rouge_new,
            "rougeL_a_param": rouge_param, "b1_success": is_new,
            "b1_outputs_param": outputs_param, "b1_ambiguous": ambiguous,
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(out_dir / "f3_b1_behavior.json", "w", encoding="utf-8") as fh:
        json.dump({
            "n": len(log), "metric": "rougeL_f1",
            "threshold": rouge_threshold, "margin": rouge_margin, "log": log,
        }, fh, indent=2, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# Cross-experiment inputs
# ═════════════════════════════════════════════════════════════════════════════


def parse_head_list(arg: str | None) -> list[tuple[int, int]]:
    """Parse ``"l1:h1,l2:h2,..."`` or JSON ``[[l,h], ...]`` head specs."""
    if not arg:
        return []
    arg = arg.strip()
    if arg.startswith("["):
        return [tuple(map(int, pair)) for pair in json.loads(arg)]
    out: list[tuple[int, int]] = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        l, h = token.split(":")
        out.append((int(l), int(h)))
    return out


def load_temporal_heads_from_f1b(path: str) -> list[tuple[int, int]]:
    """Read ``H_T`` from an F1-b results JSON.

    Looks for fields ``"H_T_heads"`` (list of [l,h] pairs) or
    ``"temporal_heads"``; absent → empty list (temporal-head fallback).
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for key in ("H_T_heads", "temporal_heads"):
        if key in data:
            return [tuple(pair) for pair in data[key]]
    return []


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(payload), fh, indent=2, ensure_ascii=False)


def _stratify_failure_traj(
    instances: list[F3PreparedInstance],
) -> tuple[list[F3PreparedInstance], list[F3PreparedInstance]]:
    """Return (PARAM_OLD failure F3-trajectory, PARAM_OTHER failure F3-trajectory)."""
    old = [i for i in instances if i.param_class == "PARAM_OLD"
           and i.b1_success is False and i.is_f3_trajectory]
    other = [i for i in instances if i.param_class == "PARAM_OTHER"
             and i.b1_success is False and i.is_f3_trajectory]
    return old, other


def _split_S_T(
    pool: list[F3PreparedInstance], *, seed: int,
) -> tuple[list[F3PreparedInstance], list[F3PreparedInstance]]:
    """50/50 stratified split by ``param_class`` for F3-c Step 1 ↔ Step 2–3."""
    rng = random.Random(seed)
    by_class: dict[str, list[F3PreparedInstance]] = {}
    for inst in pool:
        by_class.setdefault(inst.param_class, []).append(inst)
    S: list[F3PreparedInstance] = []
    T: list[F3PreparedInstance] = []
    for cls, members in by_class.items():
        rng.shuffle(members)
        mid = len(members) // 2
        S.extend(members[:mid])
        T.extend(members[mid:])
    return S, T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Layer-2 JSONL with B1 instances")
    parser.add_argument("--layer3", required=True, help="Layer-3 JSONL with A1 parametric answers")
    parser.add_argument("--layer4", help="Layer-4 JSONL with cached B1 behavior labels")
    parser.add_argument("--model", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--template", default="phi3",
                        choices=["plain", "llama2", "llama3", "phi3", "qwen"])
    parser.add_argument("--out", default="results/f3_diagnostic")
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    parser.add_argument("--dtype", default="float32",
                        choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Permit running on CPU (F3 is hours-slow on Phi-3 CPU)")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("-n", "--number", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--no-b1-behavior", action="store_true",
                        help="Skip B1 generation; treat all conflict instances as failure")
    parser.add_argument("--rouge-threshold", type=float, default=0.3)
    parser.add_argument("--rouge-margin", type=float, default=0.1)

    # F3 sub-experiment toggles.
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["f3a", "f3half", "f3b", "f3c"],
                        help="Skip F3 sub-experiments")
    parser.add_argument("--lens-kind", default="tuned", choices=["raw", "tuned"],
                        help="Lens for F3-a / F3-b / F3-c residual measurements "
                             "(methodology line 376; tuned falls back to raw "
                             "with a warning until per-layer probes are trained).")

    # Cross-experiment inputs.
    parser.add_argument("--temporal-heads", default=None,
                        help='H_T as "l:h,l:h,..." or JSON [[l,h],...]')
    parser.add_argument("--f1b-results", default=None,
                        help="F1-b results JSON (auto-extracts H_T_heads)")
    parser.add_argument("--ell-HT", type=int, default=None,
                        help="Override median temporal-head layer (else median(H_T))")
    parser.add_argument("--tau", type=float, default=0.10,
                        help="F3-a default τ (methodology line 374); sweep auto-includes 0.05/0.10/0.15")

    # F3-0.5 / F3-b knobs.
    parser.add_argument("--partition-A-size", type=int, default=100,
                        help="Methodology Partition A size (line 419)")
    parser.add_argument("--f3b-random-samples", type=int, default=20,
                        help="Methodology F3-b random-baseline samples (line 497)")

    # F3-c knobs.
    parser.add_argument("--f3c-arms", nargs="+", default=["attn", "mlp"],
                        choices=["attn", "mlp"],
                        help="Substrate arms for F3-c Steps 1–4 (methodology line 583 "
                             "pre-registers ATTN as primary; MLP retained unless demoted)")
    parser.add_argument("--f3c-late-protocol", default="Z", choices=["M", "Z"],
                        help="Tie-break for F3-c Late-KO; runs the chosen protocol. "
                             "Methodology requires both (M) and (Z); pass twice for full coverage.")
    args = parser.parse_args()

    # Resolve dtype + seed.
    dtype_map = {"float16": torch.float16, "float32": torch.float32,
                 "bfloat16": torch.bfloat16}
    if args.dtype == "auto":
        dev = args.device
        if dev == "auto":
            dev = ("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu"))
        resolved_dtype = torch.float16 if dev == "cuda" else torch.float32
    else:
        resolved_dtype = dtype_map[args.dtype]
    if resolved_dtype is not torch.float32:
        print("[WARNING] F3 mechanistic analysis is designed for float32. "
              "Use non-float32 only for memory-constrained smoke tests.")
    set_global_seed(args.sample_seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model : {args.model}\nData  : {args.data}\nLayer3: {args.layer3}\n"
          f"Dtype : {resolved_dtype}\nOutput: {out_dir}")

    # ── Load data ────────────────────────────────────────────────────
    b1_rows = load_b1(args.data)
    if args.max_instances:
        b1_rows = b1_rows[:args.max_instances]
    b1_rows = restrict_by_fact(b1_rows, number=args.number, seed=args.sample_seed)
    print(f"\nLoaded B1 instances: {len(b1_rows)}")

    print(f"\nLoading model {args.model} ...")
    model = load_model(args.model, device=args.device, dtype=resolved_dtype)
    model_device = str(model.cfg.device)
    if model_device == "cpu" and not args.allow_cpu:
        raise SystemExit(
            "[ERROR] F3 loaded the model on CPU. Run on a GPU node or pass "
            "--allow-cpu for tiny debugging runs.")
    # Do not enable per-head attention result materialization.  F3 only hooks
    # hook_z / hook_attn_out / hook_mlp_out; use_attn_result=True makes
    # TransformerLens build a huge [pos, head, d_head, d_model] intermediate
    # in attention and OOMs on 24GB GPUs during F3-a.
    model.cfg.use_attn_result = False
    print(f"  device={model_device}  {model.cfg.n_layers} layers x {model.cfg.n_heads} heads")

    layer3_by_id, layer3_by_key = load_layer3_by_key(args.layer3)
    prepared = prepare_f3_instances(model, b1_rows, layer3_by_id, layer3_by_key)
    if not prepared:
        raise SystemExit("[ERROR] No F3-ready instances after Layer3 join.")

    # B1 behavior labels.
    if args.layer4:
        layer4_by_id, layer4_by_key = load_layer4(args.layer4)
        apply_layer4_behavior(prepared, layer4_by_id=layer4_by_id,
                              layer4_by_key=layer4_by_key, out_dir=out_dir)
    else:
        classify_b1_behavior(model, prepared, template=args.template,
                             out_dir=out_dir,
                             rouge_threshold=args.rouge_threshold,
                             rouge_margin=args.rouge_margin,
                             skip_generation=args.no_b1_behavior)

    # Cross-experiment H_T.
    H_T = parse_head_list(args.temporal_heads)
    if not H_T and args.f1b_results:
        H_T = load_temporal_heads_from_f1b(args.f1b_results)
    ell_HT = args.ell_HT
    if ell_HT is None and H_T:
        ell_HT = int(np.median([l for l, _ in H_T]))

    manifest: dict[str, Any] = {
        "model": args.model, "template": args.template,
        "n_b1_loaded": len(b1_rows), "n_prepared": len(prepared),
        "lens_kind": args.lens_kind, "tau": args.tau,
        "temporal_heads": H_T, "ell_HT": ell_HT,
        "seed": args.sample_seed, "skip": args.skip,
        "partition_A_size": args.partition_A_size,
        "f3c_arms": args.f3c_arms, "f3c_late_protocol": args.f3c_late_protocol,
    }
    _write_json(out_dir / "f3_manifest.json", manifest)

    # ── F3-a ─────────────────────────────────────────────────────────
    f3a_results = []
    f3a_summary = None
    ell_R: int | None = None
    if "f3a" not in args.skip:
        f3a_results = run_f3a_trajectory(
            model, prepared, template=args.template,
            lens_kind=args.lens_kind, tau=args.tau,
        )
        # F3-0.5 might pivot ell_R; if absent (F3-0.5 skipped) we summarise with None.
        f3a_summary = summarize_f3a_population(
            f3a_results, ell_HT=ell_HT, ell_R=None, tau=args.tau,
        )
        _write_json(out_dir / "f3a_trajectory.json", {
            "tau": args.tau,
            "ell_HT": ell_HT,
            "lens_kind": args.lens_kind,
            "summary": f3a_summary,
            "per_instance": f3a_results,
        })
        print("\n[F3-a] Summary:")
        print(f"  reading: {f3a_summary.reading}")
        print(f"  traj rates: {f3a_summary.f3_traj_rate}")
        print(f"  align HT: {f3a_summary.align_HT_rate}")

    # Partitions on B1-success.
    partitions = partition_b1_success_pool(
        prepared, n_partition_A=args.partition_A_size, seed=args.sample_seed,
    )
    _write_json(out_dir / "f3a_partition.json", {
        "n_A": len(partitions["A"]), "n_B": len(partitions["B"]),
        "n_C": len(partitions["C"]),
        "A": [i.instance_id for i in partitions["A"]],
        "B": [i.instance_id for i in partitions["B"]],
        "C": [i.instance_id for i in partitions["C"]],
    })
    print(f"\nPartitions: A={len(partitions['A'])}  B={len(partitions['B'])}  "
          f"C={len(partitions['C'])}")

    # ── F3-0.5 ───────────────────────────────────────────────────────
    f3_half = None
    R_pooled: list[tuple[int, int]] = []
    R_param_old: list[tuple[int, int]] = []
    R_param_other: list[tuple[int, int]] = []
    if "f3half" not in args.skip:
        fail_old, fail_other = _stratify_failure_traj(prepared)
        fail_traj_pool = fail_old + fail_other
        print(
            "\n[F3 cohort counts] "
            f"B1-success={sum(i.b1_success is True for i in prepared)}  "
            f"B1-failure={sum(i.b1_success is False for i in prepared)}  "
            f"failure_F3traj_PARAM_OLD={len(fail_old)}  "
            f"failure_F3traj_PARAM_OTHER={len(fail_other)}"
        )
        f3_half = run_f3_half_bridge(
            model, partitions["A"], fail_old, fail_other,
            template=args.template, lens_kind=args.lens_kind, H_T=H_T,
        )
        ell_R = (int(np.median([l for l, _ in f3_half.R_pooled]))
                 if f3_half.R_pooled else None)
        # Re-summarise F3-a with ell_R now known.
        if f3a_results:
            f3a_summary = summarize_f3a_population(
                f3a_results, ell_HT=ell_HT, ell_R=ell_R, tau=args.tau,
            )
            _write_json(out_dir / "f3a_trajectory.json", {
                "tau": args.tau,
                "ell_HT": ell_HT, "ell_R": ell_R,
                "lens_kind": args.lens_kind,
                "summary": f3a_summary,
                "per_instance": f3a_results,
            })
        _write_json(out_dir / "f3_half_attribution.json", f3_half)
        R_pooled = f3_half.R_pooled
        R_param_old = f3_half.R_param_old
        R_param_other = f3_half.R_param_other
        print(f"\n[F3-0.5] rule={f3_half.selection_rule} |R|={len(R_pooled)} "
              f"panel_asymmetric={f3_half.panel_asymmetric} "
              f"ω={f3_half.omega_pooled:.2f}")
        if not fail_traj_pool:
            raise SystemExit(
                "[ERROR] F3 cannot continue: no B1-failure F3-trajectory "
                "instances in PARAM_OLD/PARAM_OTHER after F3-a. "
                "Increase MAX_INSTANCES, run the full 1000-item dataset, or "
                "lower --tau for an exploratory diagnostic."
            )
        if not R_pooled:
            raise SystemExit(
                "[ERROR] F3 cannot continue: F3-0.5 produced an empty routing "
                "set R. Increase MAX_INSTANCES or inspect f3a_trajectory.json "
                "and f3_half_attribution.json."
            )

    # ── F3-b ─────────────────────────────────────────────────────────
    f3b_res = None
    if "f3b" not in args.skip and R_pooled:
        fail_traj_pool = [i for i in prepared
                          if i.b1_success is False and i.is_f3_trajectory]
        f3b_res = run_f3b_ablation(
            model, fail_traj_pool, R_pooled,
            H_T=H_T,
            partition_B_donors=partitions["B"],
            template=args.template, lens_kind=args.lens_kind,
            n_random_samples=args.f3b_random_samples,
            rng_seed=args.sample_seed,
        )
        _write_json(out_dir / "f3b_ablation.json", f3b_res)
        print(f"\n[F3-b] {f3b_res.routed_verdict}; ρ_HT={f3b_res.rho_HT}; "
              f"primary={f3b_res.primary_protocol}")

    # ── F3-c ─────────────────────────────────────────────────────────
    f3c_by_arm_panel: dict[tuple[str, str], Any] = {}
    f3c_content_by_arm_panel: dict[tuple[str, str], Any] = {}
    if "f3c" not in args.skip and R_pooled:
        fail_traj_old   = [i for i in prepared if i.param_class == "PARAM_OLD"
                           and i.b1_success is False and i.is_f3_trajectory]
        fail_traj_other = [i for i in prepared if i.param_class == "PARAM_OTHER"
                           and i.b1_success is False and i.is_f3_trajectory]

        S_old, T_old = _split_S_T(fail_traj_old, seed=args.sample_seed)
        S_other, T_other = _split_S_T(fail_traj_other, seed=args.sample_seed + 1)
        S = S_old + S_other
        T_pool = {"PARAM_OLD": T_old, "PARAM_OTHER": T_other}

        # Step 1: L^*_σ on S, committal control on Partition C.
        l_star = run_f3c_step1_l_star(
            model, S, partitions["C"],
            template=args.template, lens_kind=args.lens_kind,
        )
        _write_json(out_dir / "f3c_step1_l_star.json", l_star)
        print(f"\n[F3-c Step 1] L*_attn={l_star['attn'].layers}  "
              f"L*_mlp={l_star['mlp'].layers}")

        R_protocol = f3b_res.primary_protocol if f3b_res else "Z"
        if R_protocol == "disagree":
            print("  [WARNING] F3-b's (M) and (Z) disagreed directionally; "
                  "F3-c Chain interaction will be reported as descriptive only.")
            R_protocol = "Z"  # fall back so the pipeline can still emit metrics

        for sigma in args.f3c_arms:
            ls = l_star[sigma]
            for panel, panel_T in T_pool.items():
                if not panel_T:
                    continue
                R_for_panel = (R_param_old if panel == "PARAM_OLD"
                               else R_param_other) or R_pooled
                res = run_f3c_step2_3(
                    model, panel_T, sigma=sigma, L_star=ls.layers,
                    R=R_for_panel, R_protocol=R_protocol,
                    late_ko_protocol=args.f3c_late_protocol,
                    partition_B_donors=partitions["B"],
                    partition_C_success=partitions["C"],
                    template=args.template, lens_kind=args.lens_kind,
                    rng_seed=args.sample_seed,
                    population_label=f"{panel} ∩ T",
                )
                f3c_by_arm_panel[(sigma, panel)] = res
                _write_json(out_dir / f"f3c_step2_3_{sigma}_{panel}.json", res)
                print(f"  [F3-c 2x2 {sigma}/{panel}] "
                      f"{res.verdict_override}; {res.verdict_chain}")

                content = run_f3c_step4_content(
                    model, panel_T, sigma=sigma, L_star=ls.layers,
                    template=args.template,
                    rng_seed=args.sample_seed,
                )
                f3c_content_by_arm_panel[(sigma, panel)] = content
                _write_json(out_dir / f"f3c_step4_{sigma}_{panel}.json", content)
                print(f"  [F3-c Step4 {sigma}/{panel}] {content.verdict}")

    # ── Combined verdict ─────────────────────────────────────────────
    if f3a_summary is not None:
        verdict = assign_f3_verdict(
            f3a_summary, f3_half, f3b_res,
            f3c_by_arm_panel, f3c_content_by_arm_panel,
        )
        _write_json(out_dir / "f3_verdict.json", verdict)
        print(f"\n[F3 verdict] title = {verdict.title!r}")
        print(f"  routed     = {verdict.routed}")
        print(f"  overridden = {verdict.overridden}")
        print(f"  chain      = {verdict.chain}")
        print(f"  content    = {verdict.content}")

    print("\nF3 Diagnostic complete. Results saved to:", out_dir)


if __name__ == "__main__":
    main()
