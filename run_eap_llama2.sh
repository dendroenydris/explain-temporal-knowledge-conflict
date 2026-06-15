#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=eap_llama2
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=ga03
#SBATCH --time=24:00:00
#SBATCH --output=logs/eap_llama2_%j.out
#SBATCH --error=logs/eap_llama2_%j.err

# EAP-IG temporal-head discovery for Llama-2-7b-chat.
# Llama SentencePiece keeps the object's first content token at index 0.
# Sanity check: this should recover the paper's a15.h0 / a18.h3.
# NOTE: Llama-2 is gated on HF — export HF_TOKEN before submitting (unless cached).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results2

CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
MODEL_NAME="${MODEL_NAME:-Llama-2-7b-chat-hf}"
OBJ_TOKEN_IDX="${OBJ_TOKEN_IDX:-0}"
YEARS="${YEARS:-1999 2004 2009}"
IG_STEPS="${IG_STEPS:-100}"
MAX_PAIRS="${MAX_PAIRS:-20}"
TOP_K="${TOP_K:-8}"

command -v conda >/dev/null || { echo "[ERROR] conda not found" >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
else
  echo "[WARNING] HF_TOKEN not set — Llama-2 is gated; download fails unless cached." >&2
fi

echo "MODEL=${MODEL}  MODEL_NAME=${MODEL_NAME}  OBJ_TOKEN_IDX=${OBJ_TOKEN_IDX}  TOP_K=${TOP_K}"
python scripts/run_eap_circuit.py \
  --model "${MODEL}" \
  --model-name "${MODEL_NAME}" \
  --obj-token-idx "${OBJ_TOKEN_IDX}" \
  --years ${YEARS} \
  --ig-steps "${IG_STEPS}" \
  --max-pairs "${MAX_PAIRS}" \
  --top-k "${TOP_K}" \
  --skip-existing \
  "$@"

echo "[OK] heads -> results2/eap_circuits/${MODEL_NAME}/discovered_temporal_heads.json"
