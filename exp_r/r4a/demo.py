"""R.4a before/after demo: the broken router, Fable's edit and the base model on the same practice prompts.

    # on the pod (needs a GPU; about 5 min: three vLLM loads)
    python -m exp_r.r4a.demo --out /workspace/artifacts/r4a/demo.json
    # anywhere: print the saved result side by side (for screen recording)
    python3 -m exp_r.r4a.demo --show artifacts/r4a/demo.json [--pause]

Practice items only (never test items). Same protocol as the evaluation: the chat template with the offered tools,
thinking off, greedy decoding; answers are cut at --max-tokens (default 1024, the eval uses 4096) to keep the run
short, which only matters for answers that loop. Each answer is scored by the harness's scorer (exp_r.r4a.scoring).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time

DEMO = [  # (practice id, why it is in the demo)
    ("live_irrelevance_824-317-0", "real user, no tool fits: broken calls a boiling-point tool"),
    ("52504__tools_of__41648", "synthetic question with unrelated tools: broken calls a pincode lookup"),
    ("live_irrelevance_301-75-0", "real user, a fact: broken runs a web search"),
    ("xlam_7515", "one fitting tool: the edit must still call it"),
    ("xlam_56280", "several calls of one function (BFCL 'parallel')"),
    ("xlam_9431", "calls of two different functions (BFCL 'parallel_multiple')"),
    ("live_irrelevance_542-168-5", "an honest failure: the edit stops calling but narrates in a loop"),
]
SUB = "/workspace/r4a/sessions/r4a_b25_fable_20261117T190942Z/submission_ckpt"


def run(a):
    from transformers import AutoTokenizer
    from exp_r.r4a.common import BASE_CKPT, ORIG_CKPT, load_practice, render
    from exp_r.r4a.scoring import score
    practice = load_practice("/workspace/r4a/practice")
    tok = AutoTokenizer.from_pretrained(ORIG_CKPT)
    items = [practice[i] for i, _ in DEMO]
    prompts = [render(tok, it["messages"], it["tools"]) for it in items]
    models = {"broken": ORIG_CKPT, "edited": SUB, "base": BASE_CKPT}
    res = {"items": [{"id": i, "why": w, "category": it["category"], "kind": it["kind"],
                      "question": it["messages"][-1]["content"], "tools": [t["function"]["name"] for t in it["tools"]]}
                     for (i, w), it in zip(DEMO, items)], "answers": {}, "max_tokens": a.max_tokens}
    for m, ckpt in models.items():
        with tempfile.TemporaryDirectory() as d:
            req = {"ckpt": ckpt, "out": f"{d}/gen.jsonl", "ids_out": f"{d}/ids.jsonl", "mem": 0.8, "keep_ids": 1,
                   "max_model_len": 16384, "items": [{"prompt": p, "max_tokens": a.max_tokens} for p in prompts]}
            json.dump(req, open(f"{d}/req.json", "w"))
            t0 = time.time()
            subprocess.run([sys.executable, "-m", "exp_r.r4a.gen_worker", f"{d}/req.json"], check=True,
                           env={**os.environ, "HF_HUB_OFFLINE": "1"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gen = [json.loads(l) for l in open(f"{d}/gen.jsonl")]
        res["answers"][m] = [{"text": g["text"], **{k: v for k, v in score(it, g["text"], g["n_tokens"], a.max_tokens).items()
                                                     if k in ("call_attempted", "correct", "loop", "n_tokens")}}
                             for it, g in zip(items, gen)]
        print(f"{m}: {time.time() - t0:.0f}s, correct {sum(x['correct'] for x in res['answers'][m])}/{len(items)}",
              flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


def show(a):
    r = json.load(open(a.show))
    w = min(os.get_terminal_size().columns if sys.stdout.isatty() else 100, 110)
    for k, it in enumerate(r["items"]):
        print("=" * w)
        print(f"[{k + 1}/{len(r['items'])}] {it['why']}")
        print("USER: " + textwrap.shorten(it["question"], 300))
        print("TOOLS OFFERED: " + ", ".join(it["tools"]))
        for m in ("broken", "edited", "base"):
            x = r["answers"][m][k]
            mark = "BAD" if not x["correct"] else ("~  " if x.get("loop") else "OK ")   # no call, but loops
            kind = "CALL" if x["call_attempted"] else ("TEXT, loops" if x.get("loop") else "TEXT")
            body = " ".join(x["text"].split())
            print(f"  {mark} {m:<7} [{kind}] " + textwrap.shorten(body, 240))
        if a.pause and k < len(r["items"]) - 1:
            input()
    print("=" * w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/artifacts/r4a/demo.json")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--show")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between items (for recording)")
    a = ap.parse_args()
    show(a) if a.show else run(a)
