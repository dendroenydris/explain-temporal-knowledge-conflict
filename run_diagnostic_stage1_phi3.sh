#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_stage1_phi3
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tatm_stage1_phi3_%j.out
#SBATCH --error=logs/tatm_stage1_phi3_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results

DATA_JSONL="${DATA_JSONL:-data/processed/wikidata_layer2_1000.jsonl}"
MODEL="${MODEL:-microsoft/phi-3-mini-4k-instruct}"
MODEL_TAG="${MODEL_TAG:-phi3}"
TEMPLATE="${TEMPLATE:-phi3}"
OUT_DIR="${OUT_DIR:-results/f1_diagnostic_1000_${MODEL_TAG}}"
ARCH="$(uname -m)"
VENV_DIR="${VENV_DIR:-.venv-${ARCH}}"

echo "ARCH=${ARCH}"
echo "VENV_DIR=${VENV_DIR}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "[ERROR] venv not found: ${VENV_DIR}. Create it on the login node first, or submit with VENV_DIR=.venv." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

[ -f "${DATA_JSONL}" ] || {
  echo "[ERROR] Missing data file: ${DATA_JSONL}" >&2
  echo "Build it first, for example:" >&2
  echo "  python scripts/build_wikidata_layer2.py --layer1 data/processed/wikidata_layer1_1000.jsonl --out ${DATA_JSONL} --layers B1 B3 B5 B6" >&2
  exit 1
}

ARGS=()
if [ -n "${MAX_INSTANCES:-}" ]; then
  ARGS+=(--max-instances "${MAX_INSTANCES}")
fi

echo "MODEL=${MODEL}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "TEMPLATE=${TEMPLATE}"
echo "OUT_DIR=${OUT_DIR}"

python scripts/run_f1_diagnostic.py \
  --data "${DATA_JSONL}" \
  --model "${MODEL}" \
  --template "${TEMPLATE}" \
  --out "${OUT_DIR}" \
  --b5 \
  "${ARGS[@]}"

python - <<PY
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
