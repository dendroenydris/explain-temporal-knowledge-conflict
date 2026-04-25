#!/bin/bash
#SBATCH --job-name=tatm_stage2_phi3
#SBATCH --account=yuxin.xue
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tatm_stage2_phi3_%j.out
#SBATCH --error=logs/tatm_stage2_phi3_%j.err

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs results

CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"
DATA_JSONL="${DATA_JSONL:-data/processed/wikidata_layer2_1000.jsonl}"
MODEL="${MODEL:-microsoft/phi-3-mini-4k-instruct}"
MODEL_TAG="${MODEL_TAG:-phi3}"
TEMPLATE="${TEMPLATE:-phi3}"
F1_OUT_DIR="${F1_OUT_DIR:-results/f1_diagnostic_1000_${MODEL_TAG}}"
OUT_DIR="${OUT_DIR:-results/f2_diagnostic_1000_${MODEL_TAG}}"
TEMPORAL_HEADS="${TEMPORAL_HEADS:-${F1_OUT_DIR}/f1a_sat_probe.json}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

command -v conda >/dev/null || {
  echo "[ERROR] conda not found. Load conda before submitting this job." >&2
  exit 1
}

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  conda env create -f environment.yml
fi
conda activate "${CONDA_ENV_NAME}"

[ -f "${DATA_JSONL}" ] || {
  echo "[ERROR] Missing data file: ${DATA_JSONL}" >&2
  exit 1
}

[ -f "${TEMPORAL_HEADS}" ] || {
  echo "[ERROR] Missing temporal heads file: ${TEMPORAL_HEADS}" >&2
  echo "Run stage 1 first: sbatch run_diagnostic_stage1_${MODEL_TAG}.sh" >&2
  exit 1
}

ARGS=()
if [ -n "${MAX_INSTANCES:-}" ]; then
  ARGS+=(--max-instances "${MAX_INSTANCES}")
fi

echo "MODEL=${MODEL}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "TEMPLATE=${TEMPLATE}"
echo "TEMPORAL_HEADS=${TEMPORAL_HEADS}"
echo "OUT_DIR=${OUT_DIR}"

python scripts/run_f2_diagnostic.py \
  --data "${DATA_JSONL}" \
  --model "${MODEL}" \
  --template "${TEMPLATE}" \
  --temporal-heads "${TEMPORAL_HEADS}" \
  --out "${OUT_DIR}" \
  "${ARGS[@]}"

python - <<PY
from pathlib import Path

out_dir = Path("${OUT_DIR}")
outputs = sorted(out_dir.glob("*.json"))
if not outputs:
    raise SystemExit(f"[ERROR] Stage 2 finished but no JSON outputs found in {out_dir}")

print("[OK] Stage 2 outputs are ready:")
for path in outputs:
    print(f"  {path}")
PY
