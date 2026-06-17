#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=eap_phi3
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=ga03
#SBATCH --time=24:00:00
#SBATCH --output=logs/eap_phi3_%j.out
#SBATCH --error=logs/eap_phi3_%j.err

# EAP-IG temporal-head discovery for Phi-3-mini.
# Phi tokenizer prepends a leading piece, so the object's first content token is
# at index 1 (notebook 1 "If the model is Phi" comments) -> OBJ_TOKEN_IDX=1.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results2

CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"
MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
MODEL_NAME="${MODEL_NAME:-Phi-3-mini-4k-instruct}"
OBJ_TOKEN_IDX="${OBJ_TOKEN_IDX:-1}"
YEARS="${YEARS:-1999 2004 2009}"
IG_STEPS="${IG_STEPS:-100}"
MAX_PAIRS="${MAX_PAIRS:-20}"
TOP_K="${TOP_K:-8}"

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

# Phi-3 is open; HF token only needed if your mirror requires auth.
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

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
