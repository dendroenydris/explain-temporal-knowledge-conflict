#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --job-name=tatm_f3_llama2
#SBATCH --account=aisc
#SBATCH --nodes=1
#SBATCH --exclude=ga03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=logs/tatm_f3_llama2_%j.out
#SBATCH --error=logs/tatm_f3_llama2_%j.err

# F3 diagnostic for Llama-2-7b-chat.  Thin wrapper over the shared phi3 F3
# driver.  Per the plan, Llama-2 is CORE-ONLY for the gradient unless more is
# needed; set RUN_M_PROTOCOL=0 to skip the heavy M protocol (default here).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
export MODEL_TAG="${MODEL_TAG:-llama2}"
export TEMPLATE="${TEMPLATE:-llama2}"
export LAYER3_JSONL="${LAYER3_JSONL:-data/processed/wikidata_layer3_${MODEL_TAG}_1000.jsonl}"
export LAYER4_JSONL="${LAYER4_JSONL:-data/processed/wikidata_layer4_${MODEL_TAG}_1000.jsonl}"
export OUT_DIR="${OUT_DIR:-results2/f3_diagnostic_1000_${MODEL_TAG}}"
export RUN_M_PROTOCOL="${RUN_M_PROTOCOL:-0}"

MODEL_NAME="${MODEL_NAME:-Llama-2-7b-chat-hf}"
EAP_HEADS_FILE="${EAP_HEADS_FILE:-results2/eap_circuits/${MODEL_NAME}/discovered_temporal_heads.json}"

if [ -n "${HF_TOKEN:-}" ]; then export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; fi

# Derive TEMPORAL_HEADS ("l:h,l:h,...") from the EAP discovery file unless set.
if [ -z "${TEMPORAL_HEADS:-}" ] && [ -f "${EAP_HEADS_FILE}" ] && command -v python3 >/dev/null; then
  TEMPORAL_HEADS=$(python3 - "${EAP_HEADS_FILE}" "${MODEL_NAME}" <<'PY'
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
  echo "[OK] TEMPORAL_HEADS from EAP: ${TEMPORAL_HEADS}"
fi
if [ -z "${TEMPORAL_HEADS:-}" ]; then
  echo "[ERROR] No TEMPORAL_HEADS for ${MODEL_NAME}. Run run_eap_llama2.sh first, " \
       "or pass TEMPORAL_HEADS=\"l:h,l:h\" explicitly." >&2
  exit 1
fi

exec bash run_f3_diagnostic_phi3.sh "$@"
