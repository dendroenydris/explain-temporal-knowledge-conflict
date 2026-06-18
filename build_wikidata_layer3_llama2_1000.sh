#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=wd_layer3_llama2
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --exclude=ga03
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wd_layer3_llama2_%j.out
#SBATCH --error=logs/wd_layer3_llama2_%j.err

# Layer-3 (A1 parametric answers) for Llama-2-7b-chat-hf.
# Thin wrapper over the shared build_wikidata_layer3_1000.sh driver.
# Must be run before run_diagnostic_stage1_llama2.sh (needs A1 for B5 filter)
# and before build_wikidata_layer4_llama2_1000.sh.
# Note: build_wikidata_layer3.py now uses zero-shot prompt style by default for
# llama2 to avoid one-shot demo-answer leakage under truncated generations.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
export MODEL_TAG="${MODEL_TAG:-llama2}"
export TEMPLATE="${TEMPLATE:-llama2}"
export LAYERS="${LAYERS:-A1}"
export OUT_JSONL="${OUT_JSONL:-data/processed/wikidata_layer3_${MODEL_TAG}_1000.jsonl}"
# Llama-2 often emits a short preamble before the entity; 48 tokens avoids
# truncating the actual answer when that happens.
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"

if [ -n "${HF_TOKEN:-}" ]; then export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; fi

exec bash build_wikidata_layer3_1000.sh "$@"
