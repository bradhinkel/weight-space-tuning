"""Step 0.2 architecture check (HF side). vLLM checks live in scripts/vllm_check.sh.

Outputs artifacts/0.2/arch_check.json and prints a human summary:
  - model class actually loaded, layer types (Gated DeltaNet vs attention)
  - every nn.Linear module-name suffix with counts and shapes (language vs vision)
  - vision encoder / merger / projector module prefixes
  - chat template rendering of a tool-call prompt with thinking on vs off
  - 50-step LoRA on the given model: loss curve, PASS if loss falls
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn

ART = Path("/workspace/artifacts/0.2")


def load(model_id):
    import transformers
    from transformers import AutoProcessor, AutoTokenizer

    tried = []
    for cls_name in ["AutoModelForImageTextToText", "AutoModelForCausalLM"]:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            m = cls.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
            tok = AutoTokenizer.from_pretrained(model_id)
            try:
                proc = AutoProcessor.from_pretrained(model_id)
            except Exception:
                proc = None
            return m, tok, proc, cls_name, tried
        except Exception as e:  # report every failure
            tried.append(f"{cls_name}: {type(e).__name__}: {str(e)[:300]}")
    raise RuntimeError("could not load model: " + " | ".join(tried))


def linear_inventory(model):
    by_suffix = defaultdict(lambda: {"count": 0, "shapes": Counter(), "example": None})
    prefixes = Counter()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            suffix = re.sub(r"\.\d+\.", ".N.", name)
            e = by_suffix[suffix]
            e["count"] += 1
            e["shapes"][f"{mod.in_features}x{mod.out_features}"] += 1
            e["example"] = e["example"] or name
            prefixes[name.split(".layers.")[0] if ".layers." in name else ".".join(name.split(".")[:3])] += 1
    inv = {k: {"count": v["count"], "shapes": dict(v["shapes"]), "example": v["example"]} for k, v in by_suffix.items()}
    return inv, dict(prefixes)


def layer_types(model):
    cfg = model.config
    tc = getattr(cfg, "text_config", cfg)
    out = {}
    for key in ["layer_types", "num_hidden_layers", "hidden_size", "full_attention_interval",
                "linear_num_key_heads", "linear_num_value_heads", "num_attention_heads", "num_key_value_heads"]:
        if hasattr(tc, key):
            out[key] = getattr(tc, key)
    return out


def template_check(tok):
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    msgs = [{"role": "user", "content": "What's the weather in Paris?"}]
    res = {}
    for think in [True, False]:
        try:
            res[f"enable_thinking={think}"] = tok.apply_chat_template(
                msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        except Exception as e:
            res[f"enable_thinking={think}"] = f"ERROR {type(e).__name__}: {e}"
    return res


def generate_once(model, tok, text):
    ids = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=128, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)


def lora_50(model, tok, steps=50):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.0, task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
    lora_modules = sorted({re.sub(r"\.\d+\.", ".N.", n.split(".lora_")[0]) for n in trainable})
    # Tiny fixed corpus: memorizable text so loss must fall if gradients flow through DeltaNet + attention.
    text = ("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n" * 8)
    batch = tok([text] * 4, return_tensors="pt").to("cuda")
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=2e-4)
    pm.train()
    losses = []
    t0 = time.time()
    for _ in range(steps):
        out = pm(**batch, labels=batch["input_ids"])
        out.loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(round(out.loss.item(), 4))
    return {"losses": losses, "first5": sum(losses[:5]) / 5, "last5": sum(losses[-5:]) / 5,
            "pass": sum(losses[-5:]) < 0.5 * sum(losses[:5]), "seconds": round(time.time() - t0, 1),
            "n_trainable_tensors": len(trainable), "lora_module_patterns": lora_modules}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--lora", action="store_true")
    a = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model, tok, proc, cls_name, tried = load(a.model)
    inv, prefixes = linear_inventory(model)
    vision = {k: v for k, v in inv.items() if re.search(r"visual|vision|merger|projector|patch", k)}
    tmpl = template_check(tok)
    gen = {}
    for k, t in tmpl.items():
        if not t.startswith("ERROR"):
            gen[k] = generate_once(model, tok, t)
    res = {"model": a.model, "loaded_class": cls_name, "load_failures": tried,
           "arch": type(model).__name__, "config_layers": layer_types(model),
           "linear_prefix_counts": prefixes, "linear_inventory": inv, "vision_linear": vision,
           "chat_template": tmpl, "greedy_generation": gen,
           "has_processor": proc is not None}
    if a.lora:
        res["lora_50"] = lora_50(model, tok)
    out = ART / f"arch_check_{a.model.split('/')[-1]}.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ["loaded_class", "arch", "config_layers", "linear_prefix_counts"]}, indent=2, default=str))
    print("VISION linear patterns:", list(vision)[:40])
    if a.lora:
        print("LORA:", {k: v for k, v in res["lora_50"].items() if k != "losses"})
    print("wrote", out)


if __name__ == "__main__":
    main()
