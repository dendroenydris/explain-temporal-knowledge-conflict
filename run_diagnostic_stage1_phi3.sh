#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_stage1_phi3
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --exclude=ga03
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
LAYER3_JSONL="${LAYER3_JSONL:-data/processed/wikidata_layer3_${MODEL_TAG}_1000.jsonl}"
TEMPORAL_HEADS_FILE="${TEMPORAL_HEADS_FILE:-data/external/temporal_heads/paper_temporal_heads.json}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"

# F1 testbed.  B5 (dual-evidence: answer_old + answer_new both in context, year
# intact) is the CORRECT population for the temporal-attention diagnostic — the
# year is the only disambiguator, so knockout actually tests "does year matter".
# B1 (single-evidence) makes the year redundant, so set USE_B5=1 here.
# USE_B5=0 reverts to the B1/B3 setup.
USE_B5="${USE_B5:-1}"
# The A1 filter needs Layer-3 A1 answers covering every B5 fact.  If Layer-3 is
# incomplete, set NO_A1_FILTER=1 to run B5 without the parametric-memory purge.
NO_A1_FILTER="${NO_A1_FILTER:-0}"

command -v conda >/dev/null || {
  echo "[ERROR] conda not found. Run setup-conda3 on the cluster, then create ${CONDA_ENV_NAME} from environment.yml." >&2
  exit 1
}

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

[ -f "${DATA_JSONL}" ] || {
  echo "[ERROR] Missing data file: ${DATA_JSONL}" >&2
  echo "Build it first, for example:" >&2
  echo "  python scripts/build_wikidata_layer2.py --layer1 data/processed/wikidata_layer1_1000.jsonl --out ${DATA_JSONL} --layers B1 B3 B5 B6" >&2
  exit 1
}

[ -f "${LAYER3_JSONL}" ] || {
  echo "[ERROR] Missing Layer-3 parametric answer file: ${LAYER3_JSONL}" >&2
  echo "Build it first, for example:" >&2
  echo "  LAYERS=B5 sbatch build_wikidata_layer3_1000.sh" >&2
  exit 1
}

# ── B5 preflight: data must contain B5/B6, and (unless skipped) Layer-3 A1 must
#    cover every B5 fact so the A1 filter's fact-key lookup succeeds. ───────────
if [ "${USE_B5}" = "1" ]; then
python - <<PY
import json, sys
from pathlib import Path

data   = Path("${DATA_JSONL}")
layer3 = Path("${LAYER3_JSONL}")
no_a1  = "${NO_A1_FILTER}" == "1"
mx     = "${MAX_INSTANCES:-}"
mx     = int(mx) if mx else None

def key(r):
    return (str(r.get("fact_id", "")), r.get("t_old"), r.get("t_new"))

b5_keys, n_b5, n_b6 = [], 0, 0
with data.open(encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        r = json.loads(line)
        iid = str(r.get("instance_id", ""))
        if iid.startswith("B5"):
            n_b5 += 1
            if mx is None or len(b5_keys) < mx:
                b5_keys.append(key(r))
        elif iid.startswith("B6"):
            n_b6 += 1

if n_b5 == 0:
    raise SystemExit("[ERROR] No B5 instances in ${DATA_JSONL}. "
                     "Rebuild layer2 with --layers B1 B3 B5 B6.")
if n_b6 == 0:
    print("[WARNING] No B6 instances — F1-b weak-group baseline will be empty.")
print(f"[OK] data has B5={n_b5}, B6={n_b6}")

if not no_a1:
    # Mirror load_layer3_answers: by_key holds every row that has a fact_id.
    a1 = set()
    with layer3.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            k = key(r)
            if k[0]:
                a1.add(k)
    missing = [k for k in b5_keys if k not in a1]
    if missing:
        raise SystemExit(
            f"[ERROR] A1 filter needs a Layer-3 A1 answer for every B5 fact, but "
            f"{len(missing)}/{len(b5_keys)} are missing (e.g. {missing[:3]}).\n"
            f"  Fix A — rebuild Layer-3 A1 to cover all facts:\n"
            f"    LAYERS=A1 LAYER2_JSONL=${DATA_JSONL} OUT_JSONL=${LAYER3_JSONL} sbatch build_wikidata_layer3_1000.sh\n"
            f"  Fix B — skip the A1 filter for this run:\n"
            f"    NO_A1_FILTER=1 sbatch run_diagnostic_stage1_phi3.sh"
        )
    print(f"[OK] A1 coverage complete for {len(b5_keys)} B5 instances.")
PY
fi

ARGS=()
if [ -n "${MAX_INSTANCES:-}" ]; then
  ARGS+=(--max-instances "${MAX_INSTANCES}")
fi
if [ -n "${NUMBER:-}" ]; then
  ARGS+=(--number "${NUMBER}")
fi
if [ -n "${SAMPLE_SEED:-}" ]; then
  ARGS+=(--sample-seed "${SAMPLE_SEED}")
fi

echo "MODEL=${MODEL}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "TEMPLATE=${TEMPLATE}"
echo "LAYER3_JSONL=${LAYER3_JSONL}"
echo "OUT_DIR=${OUT_DIR}"

# Use the paper's validated temporal heads for H_T (avoids the circular
# SAT-probe top-|coef| fallback in F1-a Step 5 / F1-b).
if [ -f "${TEMPORAL_HEADS_FILE}" ]; then
  ARGS+=(--temporal-heads "${TEMPORAL_HEADS_FILE}")
else
  echo "[WARNING] Temporal-heads file not found: ${TEMPORAL_HEADS_FILE} — F1 will use SAT-probe fallback heads." >&2
fi

# B5 (dual-evidence) testbed + optional A1-filter bypass.
if [ "${USE_B5}" = "1" ];        then ARGS+=(--b5); fi
if [ "${NO_A1_FILTER}" = "1" ];  then ARGS+=(--no-a1-filter); fi

echo "USE_B5=${USE_B5}  NO_A1_FILTER=${NO_A1_FILTER}"

python scripts/run_f1_diagnostic.py \
  --data "${DATA_JSONL}" \
  --model "${MODEL}" \
  --template "${TEMPLATE}" \
  --out "${OUT_DIR}" \
  --layer3 "${LAYER3_JSONL}" \
  "${ARGS[@]}" \
  "$@"

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
