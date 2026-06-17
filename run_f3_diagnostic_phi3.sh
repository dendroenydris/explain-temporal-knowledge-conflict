#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_f3_phi3
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --exclude=ga03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=logs/tatm_f3_phi3_%j.out
#SBATCH --error=logs/tatm_f3_phi3_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# F3 Diagnostic — full pipeline (F3-a → F3-0.5 → F3-b → F3-c Z + M protocols)
#
# Resource rationale (vs Stage-1/2 which use 48G / 24h):
#   • Phi-3-mini float16: ~7.5 GB VRAM, loaded once for the entire job
#   • F3 disables TransformerLens per-head hook_result materialization; otherwise
#     F3-a attention creates >25 GB intermediates and OOMs on 24 GB GPUs.
#   • F3-a stores scalar trajectories only; CPU RAM is reserved for JSON buffers,
#     donor means, partition metadata, and plotting.
#   • F3-b: 20 random-baseline samples × dual (M)/(Z) protocol = 40 extra
#     forward passes per failure instance; runs after F3-a cache is released
#   • F3-c Step 1 (L*_σ on S) + Step 2-3 (2×2 on T) + Step 4 (W_U projection):
#     each instance stored temporarily; peak is 2-arm × 2-panel simultaneous
#   • Total wall-clock: 24-48h for 1000 items depending on context length.
#   • 160G RAM gives headroom for full-result JSONs and plotting.
#   • Disk: F3-c per-(σ,panel) JSONs + F3-a per-instance trajectories ≈ 2-5 GB;
#     SLURM_TMPDIR is used for Tuned Lens checkpoint cache to avoid NFS pressure
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results

# Reduce CUDA fragmentation on long multi-phase jobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ── Configurable defaults (override via env on sbatch call) ──────────────────

LAYER2_JSONL="${LAYER2_JSONL:-data/processed/wikidata_layer2_1000.jsonl}"
LAYER3_JSONL="${LAYER3_JSONL:-data/processed/wikidata_layer3_phi3_1000.jsonl}"
LAYER4_JSONL="${LAYER4_JSONL:-data/processed/wikidata_layer4_phi3_1000.jsonl}"

MODEL="${MODEL:-microsoft/phi-3-mini-4k-instruct}"
MODEL_TAG="${MODEL_TAG:-phi3}"
TEMPLATE="${TEMPLATE:-phi3}"

# Park (2025) a10.h13 for Phi-3-mini; override with F1-b output if available
TEMPORAL_HEADS="${TEMPORAL_HEADS:-10:13}"

OUT_DIR="${OUT_DIR:-results/f3_diagnostic_1000_${MODEL_TAG}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"

# F3 knobs
TAU="${TAU:-0.10}"
# Default lens = raw (the Tuned Lens mandate was dropped; ~0.2pp on
# high-crystallization models). The spine verdict never depends on the lens.
LENS_KIND="${LENS_KIND:-raw}"
# One-directional timeline-confirmed temporal-stale lower bound (Change 0).
TIMELINE_JSONL="${TIMELINE_JSONL:-data/processed/wikidata_layer1_1000.jsonl}"
# Spine = DLA-head-ablation causal verdict (default). HARDENING=1 additionally
# runs the appendix lattice (span KO when not lens_na, F3-0.5/F3-b/F3-c).
HARDENING="${HARDENING:-0}"
HEAD_TOPK="${HEAD_TOPK:-4}"
PARTITION_A_SIZE="${PARTITION_A_SIZE:-100}"
F3B_RANDOM_SAMPLES="${F3B_RANDOM_SAMPLES:-20}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
DTYPE="${DTYPE:-float16}"
# M protocol only affects the hardening F3-c late-KO; gate it on HARDENING.
RUN_M_PROTOCOL="${RUN_M_PROTOCOL:-0}"
F3_FAILURE_COHORT="${F3_FAILURE_COHORT:-auto}"

# Maximum number of instances to process from the dataset.
# Leave empty (default) to use all instances in LAYER2_JSONL.
# Set to a smaller number (e.g. MAX_INSTANCES=100) for a quick smoke-test run.
MAX_INSTANCES="${MAX_INSTANCES:-}"

# ── Tuned Lens cache on fast local scratch (avoids NFS write pressure) ───────
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export TUNED_LENS_CACHE="${SLURM_TMPDIR}/tuned_lens_cache"
    mkdir -p "${TUNED_LENS_CACHE}"
    echo "Tuned Lens cache: ${TUNED_LENS_CACHE}"
fi

# ── Disk space check (need ≥ 10 GB free on the output filesystem) ────────────
_avail_kb=$(df --output=avail "$(dirname "${OUT_DIR}")" 2>/dev/null | tail -1 || echo 0)
if [ "${_avail_kb}" -lt 10485760 ] 2>/dev/null; then
    echo "[WARNING] Less than 10 GB free on output filesystem (${_avail_kb} kB available)."
    echo "  F3-a per-instance trajectories + F3-c outputs may require 3-6 GB."
    echo "  Continuing, but watch disk usage."
fi

# ── Conda ────────────────────────────────────────────────────────────────────
command -v conda >/dev/null || {
    echo "[ERROR] conda not found. Run setup-conda3 on the cluster, then create ${CONDA_ENV_NAME} from environment.yml." >&2
    exit 1
}
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
[ -f "${LAYER2_JSONL}" ] || {
    echo "[ERROR] Missing Layer-2 file: ${LAYER2_JSONL}" >&2
    echo "  Build with: python scripts/build_wikidata_layer2.py --layer1 data/processed/wikidata_layer1_1000.jsonl --out ${LAYER2_JSONL} --layers B1 B5" >&2
    exit 1
}
[ -f "${LAYER3_JSONL}" ] || {
    echo "[ERROR] Missing Layer-3 (A1 parametric answers): ${LAYER3_JSONL}" >&2
    echo "  Build with: LAYERS=A1 sbatch build_wikidata_layer3_1000.sh" >&2
    exit 1
}
python - <<PY
import json
from pathlib import Path

layer2_path = Path("${LAYER2_JSONL}")
layer3_path = Path("${LAYER3_JSONL}")
max_instances = "${MAX_INSTANCES}"
max_instances = int(max_instances) if max_instances else None

def key(row):
    return (str(row.get("fact_id", "")), row.get("t_old"), row.get("t_new"))

b1_keys = []
with layer2_path.open(encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line)
        if not str(row.get("instance_id", "")).startswith("B1"):
            continue
        b1_keys.append(key(row))
        if max_instances is not None and len(b1_keys) >= max_instances:
            break

a1_keys = set()
a1_rows = 0
with layer3_path.open(encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("layer2_type", "")) != "A1":
            continue
        a1_rows += 1
        a1_keys.add(key(row))

overlap = sum(1 for k in b1_keys if k in a1_keys)
if a1_rows == 0:
    raise SystemExit(
        "[ERROR] Layer-3 has 0 A1 rows. Rebuild it with:\n"
        "  LAYERS=A1 OUT_JSONL=${LAYER3_JSONL} sbatch build_wikidata_layer3_1000.sh"
    )
if overlap == 0:
    raise SystemExit(
        "[ERROR] Layer-3 A1 rows do not overlap the selected B1 rows.\n"
        f"  selected_B1={len(b1_keys)} A1_rows={a1_rows} overlap=0\n"
        "  Rebuild Layer-3 from the same Layer-2 file and the same sample:\n"
        "  LAYERS=A1 LAYER2_JSONL=${LAYER2_JSONL} OUT_JSONL=${LAYER3_JSONL} sbatch build_wikidata_layer3_1000.sh"
    )
print(f"[OK] Layer-3 A1 preflight: selected_B1={len(b1_keys)} A1_rows={a1_rows} overlap={overlap}")
PY
if [ -f "${LAYER4_JSONL}" ]; then
    LAYER4_ARG="--layer4 ${LAYER4_JSONL}"
else
    echo "[WARNING] Layer-4 behavior labels not found: ${LAYER4_JSONL}" >&2
    echo "  F3 will fall back to online B1 generation (slower; adds ~2h)." >&2
    echo "  Build with: LAYERS=B1 sbatch build_wikidata_layer4_1000.sh" >&2
    LAYER4_ARG=""
fi

echo "──────────────────────────────────────────────────────────────"
echo " F3 Diagnostic — Phi-3-mini"
echo " MODEL        : ${MODEL}"
echo " MODEL_TAG    : ${MODEL_TAG}"
echo " TEMPLATE     : ${TEMPLATE}"
echo " LAYER2       : ${LAYER2_JSONL}"
echo " LAYER3       : ${LAYER3_JSONL}"
echo " LAYER4       : ${LAYER4_JSONL:-<not provided, online fallback>}"
echo " TEMP. HEADS  : ${TEMPORAL_HEADS}"
echo " OUT_DIR      : ${OUT_DIR}"
echo " TAU          : ${TAU}"
echo " LENS         : ${LENS_KIND}"
echo " TIMELINE     : ${TIMELINE_JSONL}"
echo " HARDENING    : ${HARDENING}"
echo " HEAD_TOPK    : ${HEAD_TOPK}"
echo " DTYPE        : ${DTYPE}"
echo " RUN_M_PROTOCOL: ${RUN_M_PROTOCOL}"
echo " F3_FAILURE_COHORT: ${F3_FAILURE_COHORT}"
echo " PARTITION A  : ${PARTITION_A_SIZE}"
echo " F3B RANDOMS  : ${F3B_RANDOM_SAMPLES}"
echo " SEED         : ${SAMPLE_SEED}"
echo " MAX_INSTANCES: ${MAX_INSTANCES:-<all>}"
echo "──────────────────────────────────────────────────────────────"

# ── Optional extra CLI passthrough ───────────────────────────────────────────
ARGS=()
if [ -n "${MAX_INSTANCES}" ]; then ARGS+=(--max-instances "${MAX_INSTANCES}"); fi
if [ -n "${NUMBER:-}" ];      then ARGS+=(--number "${NUMBER}"); fi
# Pass F1-b results file if available (overrides --temporal-heads with H_T_heads key)
if [ -n "${F1B_RESULTS:-}" ] && [ -f "${F1B_RESULTS}" ]; then
    ARGS+=(--f1b-results "${F1B_RESULTS}")
fi
# Override ell_HT if pre-computed
if [ -n "${ELL_HT:-}" ]; then ARGS+=(--ell-HT "${ELL_HT}"); fi
# Timeline for the confirmed-stale lower bound (Change 0).
if [ -n "${TIMELINE_JSONL}" ]; then ARGS+=(--timeline-jsonl "${TIMELINE_JSONL}"); fi
# Spine head-ablation top-k.
ARGS+=(--head-topk "${HEAD_TOPK}")
# Appendix lattice only when HARDENING=1.
if [ "${HARDENING}" = "1" ]; then ARGS+=(--hardening); fi

mkdir -p "${OUT_DIR}"

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — F3-a + F3-0.5 + F3-b + F3-c (Z protocol)
# Methodology §F3: primary Late-KO protocol = Z for first run
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "[$(date '+%H:%M:%S')] Starting F3 — Protocol Z (primary)"

# shellcheck disable=SC2086
python scripts/run_f3_diagnostic.py \
    --data            "${LAYER2_JSONL}" \
    --layer3          "${LAYER3_JSONL}" \
    ${LAYER4_ARG:-} \
    --model           "${MODEL}" \
    --template        "${TEMPLATE}" \
    --temporal-heads  "${TEMPORAL_HEADS}" \
    --out             "${OUT_DIR}" \
    --tau             "${TAU}" \
    --lens-kind       "${LENS_KIND}" \
    --partition-A-size "${PARTITION_A_SIZE}" \
    --f3b-random-samples "${F3B_RANDOM_SAMPLES}" \
    --f3c-arms        attn mlp \
    --f3c-late-protocol Z \
    --sample-seed     "${SAMPLE_SEED}" \
    --dtype           "${DTYPE}" \
    --f3-failure-cohort "${F3_FAILURE_COHORT}" \
    "${ARGS[@]}" \
    "$@"

echo "[$(date '+%H:%M:%S')] Phase 1 (Z protocol) complete."

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Full F3 with M protocol — methodology requires both (M) and (Z).
# This intentionally re-runs F3-a / F3-0.5 / F3-b inside OUT_DIR_M so the
# M-protocol pass has its own in-process routing set R and final verdict.
# Results go into a sibling directory so Phase 1 outputs are not overwritten.
# ═════════════════════════════════════════════════════════════════════════════

OUT_DIR_M="${OUT_DIR}_M"
if [ "${HARDENING}" = "1" ] && [ "${RUN_M_PROTOCOL}" = "1" ]; then
mkdir -p "${OUT_DIR_M}"

# Symlink Phase-1 upstream outputs so Phase-2 can resolve partition/routing data.
for f in f3_manifest.json f3_b1_behavior.json f3a_trajectory.json \
          f3a_partition.json f3_half_attribution.json f3b_ablation.json \
          f3c_step1_l_star.json; do
    src="${OUT_DIR}/${f}"
    dst="${OUT_DIR_M}/${f}"
    [ -f "${src}" ] && [ ! -e "${dst}" ] && ln -s "$(realpath "${src}")" "${dst}"
done

echo ""
echo "[$(date '+%H:%M:%S')] Starting full F3 — Protocol M (methodology completeness)"

# shellcheck disable=SC2086
python scripts/run_f3_diagnostic.py \
    --data            "${LAYER2_JSONL}" \
    --layer3          "${LAYER3_JSONL}" \
    ${LAYER4_ARG:-} \
    --model           "${MODEL}" \
    --template        "${TEMPLATE}" \
    --temporal-heads  "${TEMPORAL_HEADS}" \
    --out             "${OUT_DIR_M}" \
    --tau             "${TAU}" \
    --lens-kind       "${LENS_KIND}" \
    --partition-A-size "${PARTITION_A_SIZE}" \
    --f3b-random-samples "${F3B_RANDOM_SAMPLES}" \
    --f3c-arms        attn mlp \
    --f3c-late-protocol M \
    --sample-seed     "${SAMPLE_SEED}" \
    --dtype           "${DTYPE}" \
    --f3-failure-cohort "${F3_FAILURE_COHORT}" \
    "${ARGS[@]}"

echo "[$(date '+%H:%M:%S')] Phase 2 (full M protocol) complete."
else
echo ""
echo "[$(date '+%H:%M:%S')] Skipping Phase 2 (M protocol) — spine-only run."
echo "  (Set HARDENING=1 RUN_M_PROTOCOL=1 to run the appendix F3-c M protocol.)"
fi

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Plotting
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo "[$(date '+%H:%M:%S')] Generating plots"

PLOT_ARGS=(--f3-dir "${OUT_DIR}" --out "${OUT_DIR}/plots" --model-tag "${MODEL_TAG}" --tau "${TAU}")
if [ "${RUN_M_PROTOCOL}" = "1" ] && [ -d "${OUT_DIR_M}" ]; then
    PLOT_ARGS+=(--f3-dir-m "${OUT_DIR_M}")
fi
python scripts/plot_f3_results.py "${PLOT_ARGS[@]}" || {
        echo "[WARNING] plot_f3_results.py failed; skipping plots (results JSONs are still valid)."
    }

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Output verification
# ═════════════════════════════════════════════════════════════════════════════

python - <<PY
import json, sys
from pathlib import Path

out_z = Path("${OUT_DIR}")
out_m = Path("${OUT_DIR_M}")
hardening = "${HARDENING}" == "1"
run_m = hardening and "${RUN_M_PROTOCOL}" == "1"

# Spine artifacts are always required; the lattice artifacts only under HARDENING.
required_z = [
    "f3_manifest.json",
    "f3a_trajectory.json",
    "f3a_failure_modes.json",
    "f3a_partition.json",
    "f3_head_ablation.json",
    "f3_verdict.json",
]
if hardening:
    required_z += [
        "f3_half_attribution.json",
        "f3b_ablation.json",
        "f3c_step1_l_star.json",
    ]
required_m = ["f3_verdict.json"] if run_m else []

missing = []
for f in required_z:
    p = out_z / f
    if not p.exists():
        missing.append(str(p))
for f in required_m:
    p = out_m / f
    if not p.exists():
        missing.append(str(p))

if missing:
    print("[ERROR] F3 finished but outputs missing:")
    for m in missing:
        print(f"  {m}")
    sys.exit(1)

# Print final spine verdict (head-ablation Δ vs B1-success population null).
verdict_path = out_z / "f3_verdict.json"
with open(verdict_path) as fh:
    v = json.load(fh)
print("")
print("══ F3 Spine Verdict (DLA-head ablation) ════════════════════")
print(f"  Verdict    : {v.get('verdict', '?')}")
print(f"  Title      : {v.get('title', '?')}")
print(f"  Delta      : {v.get('delta', '?')}  CI={v.get('delta_CI', '?')}")
print(f"  n_clean_f3 : {v.get('n_clean_f3', '?')}")
print(f"  lens_na    : {v.get('lens_na', '?')}  "
      f"(decodable={v.get('lens_decodable_fraction', '?')})")
print(f"  confirmed_stale (lower bound): {v.get('confirmed_stale_lower_bound', '?')}")
print(f"  top-k heads: {v.get('top_k_heads', '?')}")
print("════════════════════════════════════════════════════════════")
print("")
print("[OK] All required outputs present.")
print(f"  Z results : {out_z}")
if run_m:
    print(f"  M results : {out_m}")
PY
