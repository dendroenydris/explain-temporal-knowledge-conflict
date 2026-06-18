#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=wd_layer3_mistral
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --exclude=ga03
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wd_layer3_mistral_%j.out
#SBATCH --error=logs/wd_layer3_mistral_%j.err

# Layer-3 (A1 parametric answers) for Mistral-7B-Instruct-v0.1.
# Thin wrapper over the shared build_wikidata_layer3_1000.sh driver.
# Must be run before run_diagnostic_stage1_mistral.sh (needs A1 for B5 filter)
# and before build_wikidata_layer4_mistral_1000.sh.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.1}"
export MODEL_TAG="${MODEL_TAG:-mistral}"
export TEMPLATE="${TEMPLATE:-mistral}"
export LAYERS="${LAYERS:-A1}"
export OUT_JSONL="${OUT_JSONL:-data/processed/wikidata_layer3_${MODEL_TAG}_1000.jsonl}"
# Mistral can emit short prompt-echo fragments before the final name; a larger
# default budget reduces truncation into "Question: ..." fragments.
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
else
  echo "[WARNING] HF_TOKEN not set — Mistral is gated; download fails unless cached." >&2
fi

exec bash build_wikidata_layer3_1000.sh "$@"
