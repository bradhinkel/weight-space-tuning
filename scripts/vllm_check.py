"""Step 0.2 (vLLM side): confirm vLLM serves Qwen3.5 offline, with thinking off, for
  (a) the base checkpoint and (b) a merged-LoRA checkpoint produced by scripts/merge_tiny_lora.py.

Usage: python scripts/vllm_check.py --model <path-or-id> --tag base
Writes artifacts/0.2/vllm_<tag>.json with the rendered prompt, greedy output and timing.
"""
import argparse
import json
import time
from pathlib import Path

ART = Path("/workspace/artifacts/0.2")

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
PROMPTS = [
    [{"role": "user", "content": "What's the weather in Paris?"}],          # should call
    [{"role": "user", "content": "Write a haiku about autumn leaves."}],    # should abstain
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    texts = [tok.apply_chat_template(m, tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for m in PROMPTS]
    t0 = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.6,
              seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    load_s = time.time() - t0
    t1 = time.time()
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=256))
    res = {"model": a.model, "tag": a.tag, "load_seconds": round(load_s, 1),
           "gen_seconds": round(time.time() - t1, 2),
           "items": [{"prompt": t, "output": o.outputs[0].text,
                      "has_think": "<think>" in o.outputs[0].text and "</think>" not in t[-30:]}
                     for t, o in zip(texts, outs)]}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / f"vllm_{a.tag}.json").write_text(json.dumps(res, indent=2))
    for it in res["items"]:
        print("----PROMPT TAIL:", repr(it["prompt"][-120:]))
        print("----OUTPUT:", repr(it["output"][:400]))
    print({k: res[k] for k in ["load_seconds", "gen_seconds"]})


if __name__ == "__main__":
    main()
