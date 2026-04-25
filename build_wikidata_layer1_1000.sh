#!/bin/bash
#SBATCH --job-name=wd_layer1_1000
#SBATCH --account=yuxin.xue
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/wd_layer1_1000_%j.out
#SBATCH --error=logs/wd_layer1_1000_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data/processed

CONDA_ENV_NAME="${CONDA_ENV_NAME:-knowledge-temporal-kc}"
MAX_PAGES="${MAX_PAGES:-100}"
TARGET_TOTAL="${TARGET_TOTAL:-1000}"
N_PER_PROPERTY="${N_PER_PROPERTY:-1000}"
OUT_JSONL="data/processed/wikidata_layer1_1000.jsonl"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

command -v conda >/dev/null || {
  echo "[ERROR] conda not found. Load conda first, or install Miniconda on the HPC login node." >&2
  exit 1
}

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  conda env create -f environment.yml
fi

conda activate "${CONDA_ENV_NAME}"

python - <<'PY'
import importlib.util

missing = [
    name for name in ["requests", "tqdm", "yaml", "dateutil", "mwparserfromhell"]
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit("[ERROR] Missing packages: " + ", ".join(missing))
PY

python scripts/build_wikidata_layer1.py \
  --n "${N_PER_PROPERTY}" \
  --target-total "${TARGET_TOTAL}" \
  --max-pages "${MAX_PAGES}" \
  --out "${OUT_JSONL}"

python - <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "source")
from fact_timeline.models import FactTimeline

out_path = Path("data/processed/wikidata_layer1_1000.jsonl")
manifest_path = Path("data/processed/wikidata_layer1_1000_manifest.txt")

rows = []
with out_path.open(encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        FactTimeline.from_dict(dict(row))
        rows.append(row)

if not rows:
    raise SystemExit(f"[ERROR] {out_path} is empty")

fact_ids = [row["fact_id"] for row in rows]
if len(fact_ids) != len(set(fact_ids)):
    raise SystemExit("[ERROR] duplicate fact_id values found")

missing_evidence = sum(
    1
    for row in rows
    for state in row.get("states", [])
    if not state.get("evidence_text") or not state.get("source_url")
)

by_relation = Counter(row.get("property_label", "") for row in rows)
manifest = [
    "=== Wikidata Layer-1 1000 Build Manifest ===",
    f"Output file       : {out_path}",
    f"Rows              : {len(rows)}",
    f"Unique fact_ids   : {len(set(fact_ids))}",
    f"Missing evidence  : {missing_evidence}",
    "",
    "By relation:",
    *[f"  {rel:35s} {count}" for rel, count in by_relation.most_common()],
    "",
    "Use downstream with:",
    "  python scripts/build_wikidata_layer2.py --layer1 data/processed/wikidata_layer1_1000.jsonl --out data/processed/wikidata_layer2_1000.jsonl --layers B1 B3 B5 B6",
]
manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
print("\n".join(manifest))
PY
