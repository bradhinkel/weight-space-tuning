#!/usr/bin/env bash
# Fresh-volume bootstrap for R.2 (a pod whose /workspace is a new network volume). Idempotent.
# Repo at /workspace/repo (copied from the workstation). Then: venv (vLLM pinned to R.1's 0.33.0), R.1's model,
# bench and grader (exp_r/r1_setup.sh), R.2's /model and /practice (exp_r/r2/setup_pod.sh).
set -euo pipefail
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv-cache
mkdir -p /workspace/{hf,logs,artifacts} "$UV_CACHE_DIR"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd /workspace
[ -d venv ] || uv venv --python 3.12 venv
source venv/bin/activate
python -c "import vllm" 2>/dev/null || uv pip install "vllm==0.33.0"
uv pip install transformers safetensors datasets mcp numpy pandas scipy scikit-learn sympy networkx \
  huggingface_hub hf_transfer jinja2 accelerate flash-linear-attention
uv pip install causal-conv1d --no-build-isolation || echo "WARN: causal-conv1d build failed (torch fallback)"
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
for m in ["vllm", "transformers", "mcp", "fla", "scipy", "sklearn", "sympy", "networkx", "tokenizers"]:
    try:
        print(m, getattr(importlib.import_module(m), "__version__", "?"))
    except Exception as e:
        print(m, "IMPORT FAILED:", e)
PY
cd /workspace/repo
bash scripts/lockdown_perms.sh >/dev/null
# R.1 assets: base model at the original revision, the edited checkpoints, bench, grader
bash exp_r/r1_setup.sh
bash exp_r/r2/setup_pod.sh
echo BOOTSTRAP_DONE
