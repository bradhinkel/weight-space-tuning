# Tool-router demo worker for RunPod Serverless. Built by exp_r/r4a/serverless/build.sh, which stages the weights:
#   weights/base    Qwen/Qwen3.5-2B @ 15852e8c (Apache-2.0)
#   weights/broken  base + the A.2 xLAM LoRA, merged   (keep the image private: xLAM is CC-BY-NC-4.0)
#   weights/edit.npy  Fable's layer-23 down_proj (R.4a b25)
FROM python:3.12.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error WEIGHTS=/weights
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir transformers==5.23.0 accelerate==1.16.0 safetensors==0.8.0 numpy==2.5.5 \
    tokenizers==0.23.3 jinja2 runpod==1.9.1
COPY weights /weights
WORKDIR /app
COPY handler.py demo_items.json /app/
CMD ["python", "-u", "/app/handler.py"]
