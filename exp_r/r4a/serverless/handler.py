"""RunPod Serverless worker for the tool-router demo (same models and decoding as exp_r/r4a/demo_local.py).

Both weight sets stay on the GPU (about 9 GB): the base model, and the tuned (broken) model whose layer-23
down_proj is swapped for Fable's edit when "edited" is asked for. Greedy decoding, thinking off, the chat template
with the offered tools, answers cut at max_tokens (at most 400).

Input:  {"model": "base"|"broken"|"edited", "case": 1..7, "question": "" | str, "max_tokens": 400}
        question empty -> the stored prompt of demo item `case`; otherwise the question with that item's tools.
Output: {"text": str, "hit_cap": bool, "seconds": float, "gpu": str}
"""
import json
import os
import threading
import time

import numpy as np
import runpod
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

W = os.environ.get("WEIGHTS", "/weights")
EDIT_NAME = "model.language_model.layers.23.mlp.down_proj.weight"
ITEMS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_items.json")))
MAX_TOKENS, MAX_QUESTION = 400, 1000

t0 = time.time()
tok = AutoTokenizer.from_pretrained(f"{W}/broken")
load = lambda d: AutoModelForImageTextToText.from_pretrained(d, dtype=torch.bfloat16, device_map="cuda").eval()
models = {"base": load(f"{W}/base"), "broken": load(f"{W}/broken")}
param = dict(models["broken"].named_parameters())[EDIT_NAME]
weights23 = {"broken": param.detach().clone(),
             "edited": torch.from_numpy(np.load(f"{W}/edit.npy")).to(torch.bfloat16).to("cuda")}
current23 = "broken"
lock = threading.Lock()
GPU = torch.cuda.get_device_name(0)
print(f"[loaded in {time.time() - t0:.0f}s on {GPU}]", flush=True)


def render(question, tools):
    return tok.apply_chat_template([{"role": "user", "content": question}], tools=tools or None, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def handler(job):
    global current23
    inp = job.get("input") or {}
    name = inp.get("model")
    if name not in ("base", "broken", "edited"):
        return {"error": "model must be base, broken or edited"}
    try:
        item = ITEMS[int(inp.get("case", 1)) - 1]
    except (ValueError, IndexError):
        return {"error": "bad case"}
    q = str(inp.get("question") or "").strip()[:MAX_QUESTION]
    max_tokens = max(1, min(int(inp.get("max_tokens", MAX_TOKENS)), MAX_TOKENS))
    prompt = render(q, item["tools"]) if q else item["prompt"]
    with lock:
        if name != "base":
            want = "edited" if name == "edited" else "broken"
            if want != current23:
                with torch.no_grad():
                    param.data.copy_(weights23[want])
                current23 = want
        model = models["base" if name == "base" else "broken"]
        t = time.time()
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False, temperature=None, top_p=None,
                                 top_k=None)
        new = out[0, ids["input_ids"].shape[1]:]
        hit_cap = len(new) >= max_tokens and int(new[-1]) != tok.eos_token_id
        text = tok.decode(new, skip_special_tokens=True).strip()
    return {"text": text, "hit_cap": bool(hit_cap), "seconds": round(time.time() - t, 1), "gpu": GPU}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
