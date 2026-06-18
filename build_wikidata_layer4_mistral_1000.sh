#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=wd_layer4_mistral
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --exclude=ga03
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wd_layer4_mistral_%j.out
#SBATCH --error=logs/wd_layer4_mistral_%j.err

# Layer-4 (B1 behavioral labels) for Mistral-7B-Instruct-v0.1.
# Thin wrapper over the shared build_wikidata_layer4_1000.sh driver.
# Prerequisite: Layer-3 A1 file must already exist (build_wikidata_layer3_mistral_1000.sh).
# Required for run_f3_diagnostic_mistral.sh (avoids slower online B1 generation).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.1}"
export MODEL_TAG="${MODEL_TAG:-mistral}"
export TEMPLATE="${TEMPLATE:-mistral}"
export LAYERS="${LAYERS:-B1}"
export LAYER3_JSONL="${LAYER3_JSONL:-data/processed/wikidata_layer3_${MODEL_TAG}_1000.jsonl}"
export OUT_JSONL="${OUT_JSONL:-data/processed/wikidata_layer4_${MODEL_TAG}_1000.jsonl}"

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
else
  echo "[WARNING] HF_TOKEN not set — Mistral is gated; download fails unless cached." >&2
fi

exec bash build_wikidata_layer4_1000.sh "$@"
