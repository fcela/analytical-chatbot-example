#!/bin/bash
# Build the pre-configured sandbox Docker image
# This image has all data science libraries pre-installed for fast startup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${1:-analytical-chatbot-sandbox}"
IMAGE_TAG="${2:-latest}"

echo "============================================"
echo "Building sandbox image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "============================================"

docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile.sandbox" \
    "${SCRIPT_DIR}"

echo ""
echo "============================================"
echo "Build complete!"
echo "============================================"
echo ""
echo "To use this image, run:"
echo "  SANDBOX_DOCKER_IMAGE=${IMAGE_NAME}:${IMAGE_TAG} SANDBOX_FORCE_BACKEND=docker python main.py"
echo ""
echo "Or set it in your environment:"
echo "  export SANDBOX_DOCKER_IMAGE=${IMAGE_NAME}:${IMAGE_TAG}"
echo "  export SANDBOX_FORCE_BACKEND=docker"
echo "  python main.py"
echo ""
