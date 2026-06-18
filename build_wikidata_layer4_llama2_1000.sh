#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=wd_layer4_llama2
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --exclude=ga03
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wd_layer4_llama2_%j.out
#SBATCH --error=logs/wd_layer4_llama2_%j.err

# Layer-4 (B1 behavioral labels) for Llama-2-7b-chat-hf.
# Thin wrapper over the shared build_wikidata_layer4_1000.sh driver.
# Prerequisite: Layer-3 A1 file must already exist (build_wikidata_layer3_llama2_1000.sh).
# Required for run_f3_diagnostic_llama2.sh (avoids slower online B1 generation).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
export MODEL_TAG="${MODEL_TAG:-llama2}"
export TEMPLATE="${TEMPLATE:-llama2}"
export LAYERS="${LAYERS:-B1}"
export LAYER3_JSONL="${LAYER3_JSONL:-data/processed/wikidata_layer3_${MODEL_TAG}_1000.jsonl}"
export OUT_JSONL="${OUT_JSONL:-data/processed/wikidata_layer4_${MODEL_TAG}_1000.jsonl}"

if [ -n "${HF_TOKEN:-}" ]; then export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; fi

exec bash build_wikidata_layer4_1000.sh "$@"
