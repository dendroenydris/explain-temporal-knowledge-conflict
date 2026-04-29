#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_stage1_phi3_ctr
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --exclude=ga03
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tatm_stage1_phi3_container_%j.out
#SBATCH --error=logs/tatm_stage1_phi3_container_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results

CONTAINER_IMAGE="${CONTAINER_IMAGE:-container/kc-diagnostic.sqsh}"
if [ ! -f "${CONTAINER_IMAGE}" ]; then
  CONTAINER_IMAGE="nvcr.io#nvidia/pytorch:24.07-py3"
fi
CONTAINER_NAME="${CONTAINER_NAME:-kc-diagnostic-${USER}}"
DATA_JSONL="${DATA_JSONL:-data/processed/wikidata_layer2_1000.jsonl}"
MODEL="${MODEL:-microsoft/phi-3-mini-4k-instruct}"
MODEL_TAG="${MODEL_TAG:-phi3}"
TEMPLATE="${TEMPLATE:-phi3}"
OUT_DIR="${OUT_DIR:-results/f1_diagnostic_1000_${MODEL_TAG}}"

[ -f "${DATA_JSONL}" ] || {
  echo "[ERROR] Missing data file: ${DATA_JSONL}" >&2
  echo "Build it first:" >&2
  echo "  python scripts/build_wikidata_layer2.py --layer1 data/processed/wikidata_layer1_1000.jsonl --out ${DATA_JSONL} --layers B1 B3 B5 B6" >&2
  exit 1
}

ARGS=()
if [ -n "${MAX_INSTANCES:-}" ]; then
  ARGS+=(--max-instances "${MAX_INSTANCES}")
fi

COMMAND=$(cat <<EOF
set -euo pipefail
cd /workspace
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/run_f1_diagnostic.py \
  --data "${DATA_JSONL}" \
  --model "${MODEL}" \
  --template "${TEMPLATE}" \
  --out "${OUT_DIR}" \
  --b5 \
  "${ARGS[@]}"
python - <<'PY'
from pathlib import Path

out_dir = Path("${OUT_DIR}")
required = [
    out_dir / "f1a_sat_probe.json",
    out_dir / "f1b_attention_comparison.json",
    out_dir / "f1c_attention_knockout.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("[ERROR] Stage 1 finished but missing outputs: " + ", ".join(missing))

print("[OK] Stage 1 outputs are ready:")
for path in required:
    print(f"  {path}")
PY
EOF
)

echo "CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "CONTAINER_NAME=${CONTAINER_NAME}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-name="${CONTAINER_NAME}" \
  --container-writable \
  --container-mounts="${PWD}:/workspace" \
  --container-workdir=/workspace \
  --no-container-entrypoint \
  bash -lc "${COMMAND}"

