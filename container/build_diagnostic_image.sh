#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_TAG="${IMAGE_TAG:-kc-diagnostic:latest}"
SQSH_OUT="${SQSH_OUT:-${SCRIPT_DIR}/kc-diagnostic.sqsh}"

docker build -f "${SCRIPT_DIR}/Dockerfile.diagnostic" -t "${IMAGE_TAG}" "${PROJECT_ROOT}"

echo "[OK] Built ${IMAGE_TAG}"

if command -v enroot >/dev/null 2>&1; then
  rm -f "${SQSH_OUT}"
  enroot import -o "${SQSH_OUT}" "dockerd://${IMAGE_TAG}"
  echo "[OK] Exported ${SQSH_OUT}"
  echo "Submit Pyxis jobs from ${PROJECT_ROOT} with:"
  echo "  CONTAINER_IMAGE=container/$(basename "${SQSH_OUT}") sbatch run_diagnostic_stage1_phi3_container.sh"
else
  echo "[WARN] enroot not found; Docker image was built but no .sqsh was exported." >&2
  echo "Run this on the cluster where enroot is available, or set CONTAINER_IMAGE to a registry image." >&2
fi

