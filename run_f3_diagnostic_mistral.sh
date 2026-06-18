#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_f3_mistral
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --exclude=ga03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=logs/tatm_f3_mistral_%j.out
#SBATCH --error=logs/tatm_f3_mistral_%j.err

# F3 diagnostic for Mistral-7B-Instruct-v0.1 (low-crystallization; CLEAN model
# for the full F3-c causal suite, run alongside Phi-3 baseline).  PIN v0.1.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.1}"
export MODEL_TAG="${MODEL_TAG:-mistral}"
export TEMPLATE="${TEMPLATE:-mistral}"
export LAYER3_JSONL="${LAYER3_JSONL:-data/processed/wikidata_layer3_${MODEL_TAG}_1000.jsonl}"
export LAYER4_JSONL="${LAYER4_JSONL:-data/processed/wikidata_layer4_${MODEL_TAG}_1000.jsonl}"
export OUT_DIR="${OUT_DIR:-results/f3_diagnostic_1000_${MODEL_TAG}}"
export RUN_M_PROTOCOL="${RUN_M_PROTOCOL:-1}"

MODEL_NAME="${MODEL_NAME:-Mistral-7B-Instruct-v0.1}"
# Canonical temporal-heads file.  For Mistral, the paper [13] does NOT report
# temporal heads, so we use the EAP-IG self-discovered set that has been
# integrated into paper_temporal_heads.json (key: mistral-7b-instruct-v0.1;
# primary head L17.H3, coef=0.666, ~9.4× temporal/invariant ratio).
HEADS_FILE="${HEADS_FILE:-data/external/temporal_heads/paper_temporal_heads.json}"

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
else
  echo "[WARNING] HF_TOKEN not set — Mistral is gated; download fails unless cached." >&2
fi

# Derive TEMPORAL_HEADS ("l:h,l:h,...") from the canonical file unless already set.
if [ -z "${TEMPORAL_HEADS:-}" ] && [ -f "${HEADS_FILE}" ] && command -v python3 >/dev/null; then
  TEMPORAL_HEADS=$(python3 - "${HEADS_FILE}" "${MODEL_NAME}" <<'PY'
import json, sys
path, name = sys.argv[1], sys.argv[2].lower()
d = json.load(open(path))
entry = None
if "models" in d:
    for k, v in d["models"].items():
        cand = {k.lower(), str(v.get("model", "")).lower(),
                str(v.get("model", "")).lower().rsplit("/", 1)[-1]}
        if name in cand:
            entry = v; break
    entry = entry or next(iter(d["models"].values()))
else:
    entry = d
pairs = [f'{h["layer"]}:{h["head"]}' for h in entry.get("top_heads", [])
         if str(h.get("coef", "0")) not in ("0", "0.0000e+00")]
print(",".join(pairs))
PY
)
  export TEMPORAL_HEADS
  echo "[OK] TEMPORAL_HEADS from ${HEADS_FILE}: ${TEMPORAL_HEADS}"
fi
if [ -z "${TEMPORAL_HEADS:-}" ]; then
  echo "[ERROR] Could not derive TEMPORAL_HEADS for ${MODEL_NAME} from ${HEADS_FILE}." >&2
  echo "  Ensure paper_temporal_heads.json contains a 'mistral-7b-instruct-v0.1' entry," >&2
  echo "  or pass TEMPORAL_HEADS=\"l:h,l:h,...\" explicitly." >&2
  exit 1
fi

exec bash run_f3_diagnostic_phi3.sh "$@"
