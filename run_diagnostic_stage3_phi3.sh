#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_stage3_phi3
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=ga03
#SBATCH --time=24:00:00
#SBATCH --output=logs/tatm_stage3_phi3_%j.out
#SBATCH --error=logs/tatm_stage3_phi3_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results

DATA_JSONL="${DATA_JSONL:-data/processed/wikidata_layer2_1000.jsonl}"
LAYER3_JSONL="${LAYER3_JSONL:-data/processed/wikidata_layer3_phi3_1000.jsonl}"
LAYER4_JSONL="${LAYER4_JSONL:-data/processed/wikidata_layer4_phi3_1000.jsonl}"
MODEL="${MODEL:-microsoft/phi-3-mini-4k-instruct}"
MODEL_TAG="${MODEL_TAG:-phi3}"
TEMPLATE="${TEMPLATE:-phi3}"
DTYPE="${DTYPE:-float32}"
OUT_DIR="${OUT_DIR:-results/f3_diagnostic_1000_${MODEL_TAG}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"

if [ -f "${HOME}/conda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/conda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null; then
  eval "$(conda shell.bash hook)"
else
  echo "[ERROR] conda not found. Run setup-conda3 on the cluster, then create ${CONDA_ENV_NAME} from environment.yml." >&2
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"

[ -f "${DATA_JSONL}" ] || {
  echo "[ERROR] Missing Layer-2 data file: ${DATA_JSONL}" >&2
  exit 1
}

[ -f "${LAYER3_JSONL}" ] || {
  echo "[ERROR] Missing Layer-3 parametric answer file: ${LAYER3_JSONL}" >&2
  echo "Build it first, for example: LAYERS=B1 sbatch build_wikidata_layer3_1000.sh" >&2
  exit 1
}

ARGS=()
if [ -f "${LAYER4_JSONL}" ]; then
  ARGS+=(--layer4 "${LAYER4_JSONL}")
else
  echo "[WARNING] Layer-4 file not found: ${LAYER4_JSONL}" >&2
  echo "[WARNING] F3 will generate B1 behavior live unless --no-b1-behavior is used." >&2
fi
if [ -n "${MAX_INSTANCES:-}" ]; then
  ARGS+=(--max-instances "${MAX_INSTANCES}")
fi
if [ -n "${NUMBER:-}" ]; then
  ARGS+=(--number "${NUMBER}")
fi
if [ -n "${SAMPLE_SEED:-}" ]; then
  ARGS+=(--sample-seed "${SAMPLE_SEED}")
fi
if [ -n "${ROUGE_THRESHOLD:-}" ]; then
  ARGS+=(--rouge-threshold "${ROUGE_THRESHOLD}")
fi
if [ -n "${ROUGE_MARGIN:-}" ]; then
  ARGS+=(--rouge-margin "${ROUGE_MARGIN}")
fi
if [ -n "${SKIP:-}" ]; then
  # shellcheck disable=SC2206
  ARGS+=(--skip ${SKIP})
fi
if [ "${INCLUDE_SUCCESS:-0}" = "1" ]; then
  ARGS+=(--include-success)
fi
if [ "${NO_CONTROL:-0}" = "1" ]; then
  ARGS+=(--no-control)
fi
if [ "${NO_B1_BEHAVIOR:-0}" = "1" ]; then
  ARGS+=(--no-b1-behavior)
fi
if [ "${ALLOW_CPU:-0}" = "1" ]; then
  ARGS+=(--allow-cpu)
fi
if [ "${RUN_F3E:-0}" = "1" ]; then
  ARGS+=(--run-f3e)
fi
if [ -n "${F3E_MAX_INSTANCES:-}" ]; then
  ARGS+=(--f3e-max-instances "${F3E_MAX_INSTANCES}")
fi

echo "MODEL=${MODEL}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "TEMPLATE=${TEMPLATE}"
echo "DTYPE=${DTYPE}"
echo "DATA_JSONL=${DATA_JSONL}"
echo "LAYER3_JSONL=${LAYER3_JSONL}"
echo "LAYER4_JSONL=${LAYER4_JSONL}"
echo "OUT_DIR=${OUT_DIR}"

python scripts/run_f3_diagnostic.py \
  --data "${DATA_JSONL}" \
  --layer3 "${LAYER3_JSONL}" \
  --model "${MODEL}" \
  --template "${TEMPLATE}" \
  --dtype "${DTYPE}" \
  --out "${OUT_DIR}" \
  "${ARGS[@]}" \
  "$@"

python - <<PY
from pathlib import Path

out_dir = Path("${OUT_DIR}")
required = [
    out_dir / "f3_manifest.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("[ERROR] Stage 3 finished but missing outputs: " + ", ".join(missing))

print("[OK] Stage 3 outputs are ready:")
for path in sorted(out_dir.glob("f3*")):
    print(f"  {path}")
PY
