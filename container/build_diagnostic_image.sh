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

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR_FALLBACK="$(cd "$(dirname "$0")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR_FALLBACK}/.." && pwd)"
fi
SCRIPT_DIR="${PROJECT_ROOT}/container"

if [ ! -f "${PROJECT_ROOT}/requirements.txt" ] || [ ! -f "${SCRIPT_DIR}/Dockerfile.diagnostic" ]; then
  echo "[ERROR] Submit this job from the project root, e.g. cd ~/kc && sbatch container/build_diagnostic_image.sh" >&2
  echo "PROJECT_ROOT=${PROJECT_ROOT}" >&2
  exit 1
fi

IMAGE_TAG="${IMAGE_TAG:-kc-diagnostic:latest}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
SQSH_OUT="${SQSH_OUT:-${SCRIPT_DIR}/kc-diagnostic.sqsh}"
WORK_BASE="${WORK_BASE:-${SCRIPT_DIR}/.enroot-work}"
WORK_DIR="${WORK_BASE}/kc-container-${SLURM_JOB_ID:-$$}"
SQSH_TMP="${WORK_DIR}/$(basename "${SQSH_OUT}")"
CLEAN_WORK_DIR="${CLEAN_WORK_DIR:-1}"

cd "${PROJECT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "BASE_IMAGE=${BASE_IMAGE}"
echo "WORK_BASE=${WORK_BASE}"
echo "WORK_DIR=${WORK_DIR}"
echo "SQSH_TMP=${SQSH_TMP}"
echo "SQSH_OUT=${SQSH_OUT}"

export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${WORK_DIR}/enroot-cache}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-${WORK_DIR}/enroot-data}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-${WORK_DIR}/enroot-run}"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${WORK_DIR}"

mkdir -p "$(dirname "${SQSH_OUT}")"

rm -f "${SQSH_OUT}"
rm -f "${SQSH_TMP}"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker build -f "${SCRIPT_DIR}/Dockerfile.diagnostic" -t "${IMAGE_TAG}" "${PROJECT_ROOT}"
  echo "[OK] Built ${IMAGE_TAG}"

  command -v enroot >/dev/null || {
    echo "[ERROR] enroot not found; cannot export Docker image to ${SQSH_OUT}" >&2
    exit 1
  }
  enroot import -o "${SQSH_TMP}" "dockerd://${IMAGE_TAG}"
else
  command -v enroot >/dev/null || {
    echo "[ERROR] Neither Docker nor enroot is available on this node." >&2
    exit 1
  }
  echo "[INFO] Docker unavailable; importing base image with enroot: ${BASE_IMAGE}"
  enroot import -o "${SQSH_TMP}" "docker://${BASE_IMAGE}"
fi

cp "${SQSH_TMP}" "${SQSH_OUT}"
echo "[OK] Exported ${SQSH_OUT}"
if [ "${CLEAN_WORK_DIR}" = "1" ]; then
  rm -rf "${WORK_DIR}"
fi
echo "Submit Pyxis jobs from ${PROJECT_ROOT} with:"
echo "  CONTAINER_IMAGE=container/$(basename "${SQSH_OUT}") sbatch run_diagnostic_stage1_phi3_container.sh"

