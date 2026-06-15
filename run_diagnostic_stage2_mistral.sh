#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_stage2_mistral
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=ga03
#SBATCH --time=24:00:00
#SBATCH --output=logs/tatm_stage2_mistral_%j.out
#SBATCH --error=logs/tatm_stage2_mistral_%j.err

# F2 (stage 2) for Mistral-7B-Instruct-v0.1.  Low-crystallization model where the
# logit-lens trajectory + RouteScore become meaningful; lens-decodable coverage
# is expected to be highest here.  Broadened cohort (b1_failure).  PIN v0.1.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.1}"
export MODEL_TAG="${MODEL_TAG:-mistral}"
export TEMPLATE="${TEMPLATE:-mistral}"
export OUT_DIR="${OUT_DIR:-results2/f2_diagnostic_1000_${MODEL_TAG}}"
export TEMPORAL_HEADS_FILE="${TEMPORAL_HEADS_FILE:-results2/eap_circuits/Mistral-7B-Instruct-v0.1/discovered_temporal_heads.json}"
export F2B_POPULATION="${F2B_POPULATION:-b1_failure}"

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
else
  echo "[WARNING] HF_TOKEN not set — Mistral is gated; download fails unless cached." >&2
fi

# Optional: enable the residual-stream STR sweep if head-level recovery is weak.
#   F2A_RESID_SWEEP=1 sbatch run_diagnostic_stage2_mistral.sh
if [ "${F2A_RESID_SWEEP:-0}" = "1" ]; then set -- "$@" --f2a-resid-sweep; fi

exec bash run_diagnostic_stage2_phi3.sh "$@"
