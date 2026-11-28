#!/usr/bin/env bash
# Stage the tool-router worker on Experiment A's network volume (61gtyoubem, CA-MTL-1), in place of the image build.
# Run on a pod that mounts the volume at /workspace (e.g. wst-demo). Copy handler.py and demo_items.json to
# /workspace/toolrouter first. Nothing is uploaded from the laptop: the weights are already on the volume and match
# the laptop demo's byte for byte (broken model.safetensors 76fa789d..., base shard aa33250c..., edit 4d305f5c...).
#
# The endpoint (toolrouter-demo, cuut0gzmptyrvc) then runs the stock runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# image (cached on RunPod hosts, nothing to push) with
#   entrypoint ["/bin/bash","-c"]  cmd ["cd /runpod-volume/toolrouter && exec python -S -u handler.py"]
#   env PYTHONPATH=/runpod-volume/toolrouter/pkgs WEIGHTS=/runpod-volume/toolrouter/weights HF_HUB_OFFLINE=1
#   GPU pool AMPERE_48 (A40), CA-MTL-1, volume 61gtyoubem, workers 0..1, idle 300 s, FlashBoot, timeout 120 s.
# python -S keeps the image's own site-packages (torch 2.8, torchvision) out, so only the pins below are visible:
# the same as the Dockerfile's, so outputs match the laptop demo.
set -euo pipefail
T=/workspace/toolrouter
test -f $T/handler.py -a -f $T/demo_items.json
python -m pip install --no-cache-dir --target $T/pkgs \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
  torch==2.11.0 transformers==5.23.0 accelerate==1.16.0 safetensors==0.8.0 numpy==2.5.5 tokenizers==0.23.3 \
  jinja2 runpod==1.9.1
mkdir -p $T/weights && cd $T/weights
# relative links, so they resolve under /workspace (pod) and /runpod-volume (serverless worker) alike
ln -sfn ../../hf/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc base
ln -sfn ../../artifacts/a/broken broken
cp /workspace/r4a/sessions/r4a_b25_fable_20261117T190942Z/submission/model.language_model.layers.23.mlp.down_proj.weight.npy edit.npy
sha256sum edit.npy broken/model.safetensors
