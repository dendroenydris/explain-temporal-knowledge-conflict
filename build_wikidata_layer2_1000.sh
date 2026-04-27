#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=wd_layer1_1000
#SBATCH --account=aisc
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=logs/wd_layer2_1000_%j.out
#SBATCH --error=logs/wd_layer2_1000_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data/processed

python scripts/build_wikidata_layer2.py   --layer1 data/processed/wikidata_layer1_1000.jsonl   --out data/processed/wikidata_layer2_1000.jsonl   --layers B1 B3 B5 B6