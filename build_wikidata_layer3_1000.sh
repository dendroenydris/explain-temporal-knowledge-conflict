#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=wd_layer3_phi3
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --exclude=ga03
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wd_layer3_phi3_%j.out
#SBATCH --error=logs/wd_layer3_phi3_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data/processed

LAYER2_JSONL="${LAYER2_JSONL:-data/processed/wikidata_layer2_1000.jsonl}"
MODEL="${MODEL:-microsoft/phi-3-mini-4k-instruct}"
MODEL_TAG="${MODEL_TAG:-phi3}"
TEMPLATE="${TEMPLATE:-phi3}"
OUT_JSONL="${OUT_JSONL:-data/processed/wikidata_layer3_${MODEL_TAG}_1000.jsonl}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"

if [ -f "${HOME}/conda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/conda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
else
  echo "[ERROR] conda not found. Run setup-conda3 on the cluster, then create ${CONDA_ENV_NAME} from environment.yml." >&2
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"

[ -f "${LAYER2_JSONL}" ] || {
  echo "[ERROR] Missing Layer-2 file: ${LAYER2_JSONL}" >&2
  echo "Build it first with build_wikidata_layer2_1000.sh." >&2
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
if [ -n "${LAYERS:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=(--layers ${LAYERS})
fi
if [ -n "${MAX_NEW_TOKENS:-}" ]; then
  ARGS+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [ "${USE_CONTEXT:-0}" = "1" ]; then
  ARGS+=(--use-context)
fi

echo "LAYER2_JSONL=${LAYER2_JSONL}"
echo "MODEL=${MODEL}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "TEMPLATE=${TEMPLATE}"
echo "OUT_JSONL=${OUT_JSONL}"
echo "USE_CONTEXT=${USE_CONTEXT:-0}"

python scripts/build_wikidata_layer3.py \
  --layer2 "${LAYER2_JSONL}" \
  --model "${MODEL}" \
  --template "${TEMPLATE}" \
  --out "${OUT_JSONL}" \
  "${ARGS[@]}" \
  "$@"

python - <<PY
from pathlib import Path

out_path = Path("${OUT_JSONL}")
if not out_path.exists() or out_path.stat().st_size == 0:
    raise SystemExit(f"[ERROR] Layer-3 build finished but output is missing or empty: {out_path}")

print(f"[OK] Layer-3 output is ready: {out_path}")
PY
