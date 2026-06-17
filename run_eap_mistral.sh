#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=eap_mistral
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --exclude=ga03
#SBATCH --time=24:00:00
#SBATCH --output=logs/eap_mistral_%j.out
#SBATCH --error=logs/eap_mistral_%j.err

# EAP-IG temporal-head discovery for Mistral-7B-Instruct.
# No published temporal heads exist for Mistral — discovering them is itself a
# contribution (first SWA/GQA model in this taxonomy).
#  * TransformerLens 3.0 ships configs for v0.1 (NOT v0.2); use v0.1.
#  * Mistral is GQA (8 KV heads). use_split_qkv_input stays ON by default; if the
#    attribution errors on the q/k/v input hooks, resubmit with NO_SPLIT_QKV=1.
#  * SentencePiece tokenizer -> OBJ_TOKEN_IDX=0 (same as Llama).
#  * Mistral is gated on HF — export HF_TOKEN before submitting (unless cached).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results

CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.1}"
MODEL_NAME="${MODEL_NAME:-Mistral-7B-Instruct-v0.1}"
OBJ_TOKEN_IDX="${OBJ_TOKEN_IDX:-0}"
YEARS="${YEARS:-1999 2004 2009}"
IG_STEPS="${IG_STEPS:-50}"
MAX_PAIRS="${MAX_PAIRS:-20}"
TOP_K="${TOP_K:-8}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
NO_SPLIT_QKV="${NO_SPLIT_QKV:-0}"

command -v conda >/dev/null || { echo "[ERROR] conda not found" >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

python - <<'PY'
from pathlib import Path

path = Path("source/eap/attribute.py")
text = path.read_text()
if "scores.float().cpu().numpy()" not in text:
    raise SystemExit(
        "[ERROR] source/eap/attribute.py is not bf16-safe. "
        "Sync the latest file; attribute() must use "
        "scores.float().cpu().numpy()."
    )
print(f"[OK] bf16-safe EAP attribute: {path.resolve()}")
PY

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
else
  echo "[WARNING] HF_TOKEN not set — Mistral is gated; download fails unless cached." >&2
fi

ARGS=()
[ "${NO_SPLIT_QKV}" = "1" ] && ARGS+=(--no-split-qkv)

echo "MODEL=${MODEL}  MODEL_NAME=${MODEL_NAME}  OBJ_TOKEN_IDX=${OBJ_TOKEN_IDX}  TOP_K=${TOP_K}  DTYPE=${DTYPE}  MAX_SEQ_LEN=${MAX_SEQ_LEN}  IG_STEPS=${IG_STEPS}  NO_SPLIT_QKV=${NO_SPLIT_QKV}"
python scripts/run_eap_circuit.py \
  --model "${MODEL}" \
  --model-name "${MODEL_NAME}" \
  --obj-token-idx "${OBJ_TOKEN_IDX}" \
  --years ${YEARS} \
  --ig-steps "${IG_STEPS}" \
  --max-pairs "${MAX_PAIRS}" \
  --top-k "${TOP_K}" \
  --dtype "${DTYPE}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --skip-existing \
  "${ARGS[@]}" \
  "$@"

echo "[OK] heads -> results/eap_circuits/${MODEL_NAME}/discovered_temporal_heads.json"
