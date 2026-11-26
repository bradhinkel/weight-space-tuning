#!/usr/bin/env bash
# Stage the weights and build the worker image; with --push, push it to GHCR.
#   exp_r/r4a/serverless/build.sh [--push]
# Needs ~/demo_models/broken (demo_local setup) and the base model in the HF cache. The build context lives outside
# the repo (~/wst-build/toolrouter) because it holds ~9 GB of weights.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/bradhinkel/toolrouter-worker}"
TAG="${TAG:-$(git -C "$ROOT" rev-parse --short HEAD)}"
CTX="${CTX:-$HOME/wst-build/toolrouter}"
BASE_SNAP="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"

mkdir -p "$CTX/weights"
cp "$HERE/handler.py" "$HERE/Dockerfile" "$ROOT/exp_r/r4a/demo_items.json" "$CTX/"
rsync -aL --delete "$BASE_SNAP/" "$CTX/weights/base/"
rsync -a --delete "$HOME/demo_models/broken/" "$CTX/weights/broken/"
cp "$ROOT/artifacts/r4a/b25/submission/model.language_model.layers.23.mlp.down_proj.weight.npy" "$CTX/weights/edit.npy"
docker build -t "$IMAGE:$TAG" "$CTX"
echo "built $IMAGE:$TAG"
if [[ "${1:-}" == "--push" ]]; then
  docker push "$IMAGE:$TAG"
  echo "pushed $IMAGE:$TAG"
fi
