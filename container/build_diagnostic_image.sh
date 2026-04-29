#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=build_kc_container
#SBATCH --output=container/build_kc_container_%j.out
#SBATCH --error=container/build_kc_container_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-kc-diagnostic:latest}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io#nvidia/pytorch:24.07-py3}"
SQSH_OUT="${SQSH_OUT:-${SCRIPT_DIR}/kc-diagnostic.sqsh}"

cd "${PROJECT_ROOT}"

if [ -n "${SLURM_SCRATCH:-}" ]; then
  export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${SLURM_SCRATCH}/enroot-cache}"
  export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-${SLURM_SCRATCH}/enroot-data}"
  export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-${SLURM_SCRATCH}/enroot-run}"
  mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}"
fi

mkdir -p "$(dirname "${SQSH_OUT}")"

rm -f "${SQSH_OUT}"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker build -f "${SCRIPT_DIR}/Dockerfile.diagnostic" -t "${IMAGE_TAG}" "${PROJECT_ROOT}"
  echo "[OK] Built ${IMAGE_TAG}"

  command -v enroot >/dev/null || {
    echo "[ERROR] enroot not found; cannot export Docker image to ${SQSH_OUT}" >&2
    exit 1
  }
  enroot import -o "${SQSH_OUT}" "dockerd://${IMAGE_TAG}"
else
  command -v enroot >/dev/null || {
    echo "[ERROR] Neither Docker nor enroot is available on this node." >&2
    exit 1
  }
  echo "[INFO] Docker unavailable; importing base image with enroot: ${BASE_IMAGE}"
  enroot import -o "${SQSH_OUT}" "docker://${BASE_IMAGE}"
fi

echo "[OK] Exported ${SQSH_OUT}"
echo "Submit Pyxis jobs from ${PROJECT_ROOT} with:"
echo "  CONTAINER_IMAGE=container/$(basename "${SQSH_OUT}") sbatch run_diagnostic_stage1_phi3_container.sh"

