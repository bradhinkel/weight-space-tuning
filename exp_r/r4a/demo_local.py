"""Laptop demo (8 GB GPU): the broken router, Fable's edit and the base model, live, on the seven demo prompts.

One copy of Qwen3.5-2B sits on the GPU. The three models are three weight sets:
  broken = base + the A.2 LoRA, merged        (built once by `setup`)
  edited = broken with Fable's layer-23 down_proj (artifacts/r4a/b25/submission/*.npy; 2 columns differ)
  base   = Qwen/Qwen3.5-2B as released
Switching broken <-> edited copies one 2048x6144 tensor (instant); switching to base copies the full weights from
CPU memory (about a second).

Setup (once; needs the broken checkpoint or its adapter from the volume, see the demo doc):
  ~/demo_env/.venv/bin/python -m exp_r.r4a.demo_local setup --adapter ~/demo_models/adapter
  # or, with the merged checkpoint copied whole:  setup --broken ~/demo_models/broken_pulled
Demo:
  ~/demo_env/.venv/bin/python -m exp_r.r4a.demo_local run --cases 1,4,7 --pause
  ~/demo_env/.venv/bin/python -m exp_r.r4a.demo_local run --models base --cases 1      # works before setup
  ~/demo_env/.venv/bin/python -m exp_r.r4a.demo_local ask --case 1 "What is the boiling point of water at sea level?"

Same protocol as the evaluation: the chat template with the offered tools, thinking off, greedy decoding.
Answers are cut at --max-tokens (default 400); an answer that hits the cap is shown as a loop.
"""
import argparse
import glob
import json
import os
import re
import sys
import textwrap
import time

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")       # hides the slow-kernel fallback notices
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ITEMS = os.path.join(HERE, "demo_items.json")
EDIT_NPY = os.path.join(ROOT, "artifacts/r4a/b25/submission/model.language_model.layers.23.mlp.down_proj.weight.npy")
EDIT_NAME = "model.language_model.layers.23.mlp.down_proj.weight"
BASE_REPO, BASE_REV = "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
MODELS_DIR = os.path.expanduser(os.environ.get("DEMO_MODELS", "~/demo_models"))
BROKEN_DIR = os.path.join(MODELS_DIR, "broken")
LABEL = {"broken": "BROKEN", "edited": "FIXED ", "base": "BASE  "}


def base_dir():
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(BASE_REPO, revision=BASE_REV, local_files_only=True)
    except Exception:
        return snapshot_download(BASE_REPO, revision=BASE_REV)


# ------------------------------------------------------------------ setup
def setup(a):
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    src = base_dir()
    os.makedirs(MODELS_DIR, exist_ok=True)
    if a.broken:
        model = AutoModelForImageTextToText.from_pretrained(a.broken, dtype=torch.bfloat16)
    else:
        from peft import PeftModel
        model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    # check: the local broken model must equal the session's broken model everywhere in layer 23 down_proj except
    # the columns Fable edited (the .npy holds broken + edit in float32)
    w = dict(model.named_parameters())[EDIT_NAME].detach().float().cpu().numpy()
    e = np.load(EDIT_NPY)
    cols = np.where(np.abs(w - e).max(axis=0) > 1e-6)[0]
    rest = np.delete(np.abs(w - e), cols, axis=1).max() if len(cols) < w.shape[1] else float("nan")
    print(f"edit check: {len(cols)} columns differ from broken ({cols[:10].tolist()}); max diff elsewhere {rest:.2e}")
    if len(cols) > 8:
        sys.exit("the local broken model does not match the session's broken model; copy the merged checkpoint instead")
    model.save_pretrained(BROKEN_DIR, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(BROKEN_DIR)
    AutoProcessor.from_pretrained(src).save_pretrained(BROKEN_DIR)
    print("wrote", BROKEN_DIR)


# ------------------------------------------------------------------ engine
class Demo:
    def __init__(self, need):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.torch = torch
        have_broken = os.path.exists(os.path.join(BROKEN_DIR, "config.json"))
        if ({"broken", "edited"} & need) and not have_broken:
            sys.exit(f"no broken model at {BROKEN_DIR}: run `setup` first (or use --models base)")
        first = BROKEN_DIR if have_broken and need != {"base"} else base_dir()
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(first)
        self.model = AutoModelForImageTextToText.from_pretrained(first, dtype=torch.bfloat16, device_map="cuda").eval()
        self.params = dict(self.model.named_parameters())
        self.current = "base" if first != BROKEN_DIR else "broken"
        self.cpu = {}                                        # weight set name -> {param: cpu tensor}
        if "base" in need and self.current == "broken":
            from safetensors.torch import load_file
            sd = {}
            for f in sorted(glob.glob(os.path.join(base_dir(), "*.safetensors"))):
                sd.update(load_file(f))
            self.cpu["base"] = {n: sd[n] for n in self.params if n in sd}
            missing = [n for n in self.params if n not in sd]
            assert not missing, missing[:5]
            self.cpu["broken"] = {n: p.detach().to("cpu", copy=True) for n, p in self.params.items()}
        if "edited" in need:
            import numpy as np
            self.edit = torch.from_numpy(np.load(EDIT_NPY)).to(torch.bfloat16)
            self.broken23 = self.params[EDIT_NAME].detach().clone()
        print(f"[loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}]", flush=True)

    def use(self, name):
        if name == self.current:
            return
        with self.torch.no_grad():
            if name == "base" or self.current == "base":
                full = self.cpu["base" if name == "base" else "broken"]
                for n, t in full.items():
                    self.params[n].data.copy_(t, non_blocking=True)
            if name == "edited":
                self.params[EDIT_NAME].data.copy_(self.edit)
            elif name == "broken" and self.current == "edited":
                self.params[EDIT_NAME].data.copy_(self.broken23)
        self.torch.cuda.synchronize()
        self.current = name

    def generate(self, prompt, max_tokens):
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None)
        new = out[0, ids["input_ids"].shape[1]:]
        hit_cap = len(new) >= max_tokens and int(new[-1]) != self.tok.eos_token_id
        return self.tok.decode(new, skip_special_tokens=True).strip(), hit_cap


# ------------------------------------------------------------------ display
def render_prompt(tok, question, tools):
    return tok.apply_chat_template([{"role": "user", "content": question}], tools=tools or None, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def summarize(text, hit_cap, width):
    calls = re.findall(r"<function=([^>\s]+)>(.*?)</function>", text, re.S)
    if calls:
        out = []
        for fn, body in calls[:4]:
            args = ", ".join(f"{k}={' '.join(v.split())}" for k, v in re.findall(r"<parameter=([^>\s]+)>(.*?)</parameter>", body, re.S))
            out.append(f"{fn}({args})")
        more = f" +{len(calls) - 4} more" if len(calls) > 4 else ""
        return "CALLS TOOL", textwrap.shorten("; ".join(out) + more, width)
    flat = " ".join(text.split())
    tail = flat[-1200:]
    loops = hit_cap and any(tail.count(tail[i:i + 80]) >= 3 for i in range(0, max(1, len(tail) - 80), 10))
    kind = "ANSWERS, but loops" if loops else ("ANSWERS (cut short)" if hit_cap else "ANSWERS")
    return kind, textwrap.shorten(" ".join(text.split()), width)


def show_case(demo, k, item, models, a, prompt=None, question=None):
    w = 100
    print("=" * w)
    print(f"[case {k}] {item['why']}" if question is None else f"[your prompt, tools from case {k}]")
    print("USER: " + textwrap.shorten(question or item["question"], 300))
    print("TOOLS OFFERED: " + ", ".join(t["function"]["name"] for t in item["tools"]))
    print("-" * w)
    for m in models:
        demo.use(m)
        t0 = time.time()
        text, cap = demo.generate(prompt or item["prompt"], a.max_tokens)
        kind, body = summarize(text, cap, 300)
        print(f"{LABEL[m]} {kind:<18} {body}   ({time.time() - t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapter", help="the A.2 LoRA adapter directory (broken/adapter on the volume)")
    g.add_argument("--broken", help="the merged broken checkpoint directory (artifacts/a/broken on the volume)")
    for name in ("run", "ask"):
        p = sub.add_parser(name)
        p.add_argument("--models", default="broken,edited,base")
        p.add_argument("--max-tokens", type=int, default=400)
        if name == "run":
            p.add_argument("--cases", default="1,2,3,4,5,6,7")
            p.add_argument("--pause", action="store_true", help="wait for Enter between cases (for recording)")
        else:
            p.add_argument("--case", type=int, default=1, help="offer the tools of this demo case (1-7)")
            p.add_argument("--no-tools", action="store_true")
            p.add_argument("question")
    a = ap.parse_args()
    if a.cmd == "setup":
        return setup(a)
    items = json.load(open(ITEMS))
    models = [m.strip() for m in a.models.split(",")]
    demo = Demo(set(models))
    if a.cmd == "run":
        cases = [int(c) for c in a.cases.split(",")]
        for i, k in enumerate(cases):
            show_case(demo, k, items[k - 1], models, a)
            if a.pause and i < len(cases) - 1:
                input()
        print("=" * 100)
    else:
        item = dict(items[a.case - 1])
        if a.no_tools:
            item["tools"] = []
        show_case(demo, a.case, item, models, a, prompt=render_prompt(demo.tok, a.question, item["tools"]),
                  question=a.question)
        print("=" * 100)


if __name__ == "__main__":
    main()
