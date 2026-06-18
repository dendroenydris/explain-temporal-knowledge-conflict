#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=reextract_l3
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/reextract_l3_%j.out
#SBATCH --error=logs/reextract_l3_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"
INPUT_JSONL="${INPUT_JSONL:-data/processed/wikidata_layer3_mistral_1000.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-}"
BACKUP_SUFFIX="${BACKUP_SUFFIX:-.bak_reextract}"

if [ -f "${HOME}/conda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/conda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
else
  echo "[ERROR] conda not found. Run setup-conda3 and create ${CONDA_ENV_NAME}." >&2
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"

[ -f "${INPUT_JSONL}" ] || {
  echo "[ERROR] Missing input file: ${INPUT_JSONL}" >&2
  exit 1
}

echo "INPUT_JSONL=${INPUT_JSONL}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL:-<in-place>}"
echo "BACKUP_SUFFIX=${BACKUP_SUFFIX}"
echo "CONDA_ENV_NAME=${CONDA_ENV_NAME}"

ARGS=(--input "${INPUT_JSONL}" --backup-suffix "${BACKUP_SUFFIX}")
if [ -n "${OUTPUT_JSONL}" ]; then
  ARGS+=(--output "${OUTPUT_JSONL}")
fi

python scripts/reextract_layer3_answers.py "${ARGS[@]}"

echo "[OK] Re-extraction done."
