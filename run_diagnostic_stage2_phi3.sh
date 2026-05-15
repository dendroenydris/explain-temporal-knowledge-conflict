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

command -v conda >/dev/null || {
  echo "[ERROR] conda not found. Run setup-conda3 on the cluster, then create ${CONDA_ENV_NAME} from environment.yml." >&2
  exit 1
}

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

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

echo "MODEL=${MODEL}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "TEMPLATE=${TEMPLATE}"
echo "TEMPORAL_HEADS_FILE=${TEMPORAL_HEADS_FILE}"
echo "TEMPORAL_HEADS_MANUAL=${TEMPORAL_HEADS_MANUAL}"
echo "DTYPE=${DTYPE}"
echo "F2B_POPULATION=${F2B_POPULATION}"
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
from pathlib import Path

out_dir = Path("${OUT_DIR}")
outputs = sorted(out_dir.glob("*.json"))
if not outputs:
    raise SystemExit(f"[ERROR] Stage 2 finished but no JSON outputs found in {out_dir}")

print("[OK] Stage 2 outputs are ready:")
for path in outputs:
    print(f"  {path}")
PY
