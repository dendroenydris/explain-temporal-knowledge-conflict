#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_stage1_mistral
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=ga03
#SBATCH --time=24:00:00
#SBATCH --output=logs/tatm_stage1_mistral_%j.out
#SBATCH --error=logs/tatm_stage1_mistral_%j.err

# F1 (stage 1) for Mistral-7B-Instruct-v0.1 (low-crystallization, ~27%).
# PIN v0.1 everywhere (TransformerLens config + EAP head indices must match).
# Requires the `mistral` chat template (added to tatm/model.py) and HF_TOKEN.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.1}"
export MODEL_TAG="${MODEL_TAG:-mistral}"
export TEMPLATE="${TEMPLATE:-mistral}"
export OUT_DIR="${OUT_DIR:-results/f1_diagnostic_1000_${MODEL_TAG}}"
export TEMPORAL_HEADS_FILE="${TEMPORAL_HEADS_FILE:-results/eap_circuits/Mistral-7B-Instruct-v0.1/discovered_temporal_heads.json}"

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
else
  echo "[WARNING] HF_TOKEN not set — Mistral is gated; download fails unless cached." >&2
fi

exec bash run_diagnostic_stage1_phi3.sh "$@"
