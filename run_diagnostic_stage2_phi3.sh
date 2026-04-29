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
TEMPORAL_HEADS="${TEMPORAL_HEADS:-data/external/temporal_heads/paper_temporal_heads.json}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"

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

[ -f "${TEMPORAL_HEADS}" ] || {
  echo "[ERROR] Missing temporal heads file: ${TEMPORAL_HEADS}" >&2
  echo "Expected the paper-derived temporal heads file at data/external/temporal_heads/paper_temporal_heads.json" >&2
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
