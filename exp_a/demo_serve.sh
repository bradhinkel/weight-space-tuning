#!/bin/bash
# A.9: serve base, broken and the two repaired checkpoints side by side on one 48 GB GPU (pod).
#   bash exp_a/demo_serve.sh          # start all four (about 2 min), logs in /workspace/logs/demo_*.log
#   bash exp_a/demo_serve.sh stop     # stop them
# Ports: 8101 base, 8102 broken, 8103 fable_b25 (driver repair), 8104 lora (corrective LoRA).
set -e
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
D=/workspace/artifacts/a/demo
if [ "$1" = stop ]; then pkill -f "vllm serve" || true; exit 0; fi
declare -A CK=([8101]=Qwen/Qwen3.5-2B [8102]=$D/broken [8103]=$D/fable_b25 [8104]=$D/lora)
declare -A NAME=([8101]=base [8102]=broken [8103]=fable_b25 [8104]=lora)
for p in 8101 8102 8103 8104; do
  nohup vllm serve "${CK[$p]}" --served-model-name "${NAME[$p]}" --port $p --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.2 --seed 0 \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' > /workspace/logs/demo_${NAME[$p]}.log 2>&1 &
  pid=$!
  # one at a time: vLLM's start-up memory profiling fails if another server is loading at the same moment
  until curl -sf localhost:$p/v1/models > /dev/null; do
    kill -0 $pid 2>/dev/null || { echo "${NAME[$p]} failed, see /workspace/logs/demo_${NAME[$p]}.log"; exit 1; }
    sleep 5
  done
  echo "${NAME[$p]} ready on :$p"
done
