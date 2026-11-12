#!/usr/bin/env bash
# Step 0.1: build the environment on the RunPod box. Idempotent.
# Venv, HF cache and repo live on the network volume (/workspace) so they survive pod replacement.
set -euo pipefail
export HF_HOME=/workspace/hf
export UV_CACHE_DIR=/workspace/uv-cache
mkdir -p /workspace/{hf,logs,artifacts} "$UV_CACHE_DIR"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd /workspace
[ -d venv ] || uv venv --python 3.12 venv
source venv/bin/activate

# vLLM pins its own torch; install it first, then the rest against that torch.
uv pip install "vllm" 
uv pip install transformers peft accelerate datasets safetensors \
  anthropic mcp numpy pandas matplotlib huggingface_hub hf_transfer pyyaml \
  flash-linear-attention
# causal-conv1d is optional (fast path for Gated DeltaNet conv); don't fail the setup if it won't build.
uv pip install causal-conv1d --no-build-isolation || echo "WARN: causal-conv1d build failed (torch fallback will be used)"

python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
for m in ["vllm","transformers","peft","accelerate","datasets","anthropic","mcp","fla"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "IMPORT FAILED:", e)
PY
