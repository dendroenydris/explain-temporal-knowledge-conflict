#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_stage2_phi3
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --exclude=ga03
#SBATCH --time=24:00:00
#SBATCH --output=logs/tatm_stage2_phi3_%j.out
#SBATCH --error=logs/tatm_stage2_phi3_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results

DATA_JSONL="${DATA_JSONL:-data/processed/wikidata_layer2_1000.jsonl}"
MODEL="${MODEL:-microsoft/phi-3-mini-4k-instruct}"
MODEL_TAG="${MODEL_TAG:-phi3}"
TEMPLATE="${TEMPLATE:-phi3}"
OUT_DIR="${OUT_DIR:-results/f2_diagnostic_1000_${MODEL_TAG}}"
TEMPORAL_HEADS_FILE="${TEMPORAL_HEADS_FILE:-data/external/temporal_heads/paper_temporal_heads.json}"
TEMPORAL_HEADS_MANUAL="${TEMPORAL_HEADS_MANUAL:-10,13}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"
DTYPE="${DTYPE:-auto}"
F2B_POPULATION="${F2B_POPULATION:-reverts_old}"
# F1 cross-reference (optional): if these exist, F2 verdicts are ruled-out
# against F1-a Step-5 (per instance) and F1-b is recorded as a SOFT annotation
# (no longer invalidates the verdicts).  Default to the results/ F1 dir for
# this model tag; skipped automatically if the files are absent.
F1_RESULTS="${F1_RESULTS:-results/f1_diagnostic_1000_${MODEL_TAG}/f1a_sat_probe.json}"
F1B_RESULTS="${F1B_RESULTS:-results/f1_diagnostic_1000_${MODEL_TAG}/f1b_attention_comparison.json}"

command -v conda >/dev/null || {
  echo "[ERROR] conda not found. Run setup-conda3 on the cluster, then create ${CONDA_ENV_NAME} from environment.yml." >&2
  exit 1
}

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

# ── GPU pre-flight ───────────────────────────────────────────────────────────
# Reproducible cudaErrorDevicesUnavailable usually means the assigned GPU is
# wedged/occupied.  Log the GPU state and fail fast (before the model download)
# so the node can be identified and excluded.
echo "── GPU pre-flight ────────────────────────────────────────────"
echo "  NODE              : $(hostname)"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>&1 || \
    echo "  [WARN] nvidia-smi failed"
python - <<'PY' || { echo "[ERROR] CUDA not usable on $(hostname); resubmit (exclude this node if it recurs)."; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("  [ERROR] torch.cuda.is_available() == False"); sys.exit(1)
try:
    torch.zeros(8, device="cuda").sum().item()
    print(f"  [OK] CUDA usable: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"  [ERROR] CUDA alloc failed: {exc}"); sys.exit(1)
PY
echo "──────────────────────────────────────────────────────────────"

[ -f "${DATA_JSONL}" ] || {
  echo "[ERROR] Missing data file: ${DATA_JSONL}" >&2
  exit 1
}

ARGS=()
if [ -n "${MAX_INSTANCES:-}" ]; then
  ARGS+=(--max-instances "${MAX_INSTANCES}")
fi
if [ -n "${NUMBER:-}" ]; then
  ARGS+=(--number "${NUMBER}")
fi
if [ -n "${SAMPLE_SEED:-}" ]; then
  ARGS+=(--sample-seed "${SAMPLE_SEED}")
fi
if [ -n "${SKIP:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=(--skip ${SKIP})
fi

if [ -f "${TEMPORAL_HEADS_FILE}" ]; then
  HEAD_ARGS=(--temporal-heads "${TEMPORAL_HEADS_FILE}")
else
  # shellcheck disable=SC2206
  HEAD_ARGS=(--temporal-heads-manual ${TEMPORAL_HEADS_MANUAL})
fi

# F1 cross-reference (optional; activates the F2 verdict ruling-out + soft f1b).
if [ -f "${F1_RESULTS}" ];  then ARGS+=(--f1-results  "${F1_RESULTS}");  fi
if [ -f "${F1B_RESULTS}" ]; then ARGS+=(--f1b-results "${F1B_RESULTS}"); fi

echo "MODEL=${MODEL}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "TEMPLATE=${TEMPLATE}"
echo "TEMPORAL_HEADS_FILE=${TEMPORAL_HEADS_FILE}"
echo "TEMPORAL_HEADS_MANUAL=${TEMPORAL_HEADS_MANUAL}"
echo "DTYPE=${DTYPE}"
echo "F2B_POPULATION=${F2B_POPULATION}"
echo "F1_RESULTS=${F1_RESULTS} (exists=$([ -f "${F1_RESULTS}" ] && echo yes || echo no))"
echo "F1B_RESULTS=${F1B_RESULTS} (exists=$([ -f "${F1B_RESULTS}" ] && echo yes || echo no))"
echo "MAX_INSTANCES=${MAX_INSTANCES:-<all>}"
echo "SKIP=${SKIP:-<none>}"
echo "OUT_DIR=${OUT_DIR}"

python scripts/run_f2_diagnostic.py \
  --data "${DATA_JSONL}" \
  --model "${MODEL}" \
  --template "${TEMPLATE}" \
  "${HEAD_ARGS[@]}" \
  --out "${OUT_DIR}" \
  --dtype "${DTYPE}" \
  --f2b-population "${F2B_POPULATION}" \
  "${ARGS[@]}" \
  "$@"

python - <<PY
import json
from pathlib import Path

out_dir = Path("${OUT_DIR}")
outputs = sorted(out_dir.glob("*.json"))
if not outputs:
    raise SystemExit(f"[ERROR] Stage 2 finished but no JSON outputs found in {out_dir}")

verdict_path = out_dir / "f2_verdicts.json"
if verdict_path.exists():
    verdict = json.load(open(verdict_path))
    schema = verdict.get("schema_version")
    if schema != "f2_v2_dla_clean_verdict":
        raise SystemExit(
            "[ERROR] Stage 2 wrote an old/incompatible f2_verdicts.json schema "
            f"({schema!r}). Make sure the cluster copy includes the latest "
            "scripts/run_f2_diagnostic.py and source/tatm/f2_diagnosis.py."
        )
    print(f"[OK] F2 schema: {schema}")
    print(f"[OK] F2 diagnosis source: {verdict.get('tatm_f2_diagnosis_path')}")

print("[OK] Stage 2 outputs are ready:")
for path in outputs:
    print(f"  {path}")
PY
