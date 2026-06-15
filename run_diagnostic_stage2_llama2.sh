#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_stage2_llama2
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=ga03
#SBATCH --time=24:00:00
#SBATCH --output=logs/tatm_stage2_llama2_%j.out
#SBATCH --error=logs/tatm_stage2_llama2_%j.err

# F2 (stage 2) for Llama-2-7b-chat.  Thin wrapper over the shared phi3 driver.
# Broadened cohort (b1_failure) feeds the DLA/rank F1/F2/F3 classifier.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
export MODEL_TAG="${MODEL_TAG:-llama2}"
export TEMPLATE="${TEMPLATE:-llama2}"
export OUT_DIR="${OUT_DIR:-results2/f2_diagnostic_1000_${MODEL_TAG}}"
export TEMPORAL_HEADS_FILE="${TEMPORAL_HEADS_FILE:-results2/eap_circuits/Llama-2-7b-chat-hf/discovered_temporal_heads.json}"
export F2B_POPULATION="${F2B_POPULATION:-b1_failure}"

if [ -n "${HF_TOKEN:-}" ]; then export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; fi

exec bash run_diagnostic_stage2_phi3.sh "$@"
