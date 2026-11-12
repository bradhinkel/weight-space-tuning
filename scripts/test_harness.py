"""Harness self-test without a driver (run on the pod as root, before the 0.5 smoke test).

    python -m scripts.test_harness [--ckpt Qwen/Qwen3.5-0.8B]

Checks: kernel sandbox (file / network / GPU access refused), W edits reach forward_pass, budget
enforcement, test-guard refusal, attention mode, save_weights diff, submit freezing. Writes
artifacts/0.3/harness_selftest.json.
"""
import argparse
import json
import time
from pathlib import Path

from common.harness.tools import Harness, text_hash

ESCAPES = {
    "read_artifacts": "open('/workspace/artifacts/a/data/split.json').read(10)",
    "read_hf_cache": "import os; os.listdir('/workspace/hf')",
    "read_repo": "open('/workspace/repo/PROJECT_PLAN.md').read(10)",
    "read_bfcl_venv": "import os; os.listdir('/workspace/bfcl-venv')",
    "read_tmp": "import os; os.listdir('/tmp')",
    "write_outside": "open('/srv/wst-sessions/x.txt','w').write('x')",
    "network_tcp": "import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)",
    "network_dns": "import socket; socket.getaddrinfo('huggingface.co', 443)",
    "gpu_torch": "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())",
    "gpu_device_node": "open('/dev/nvidiactl','rb')",
    "whoami": "import os; print(os.getuid(), os.getcwd())",
    "workdir_write": "open('scratch.txt','w').write('ok'); print(open('scratch.txt').read())",
    "practice_read": "import os; print(os.listdir('practice'))",
    "tokenizer": "print(len(TOK.encode('hello world')), TOK.decode(TOK.encode('hello world')))",
    "import_transformers": "import transformers",
    "import_vllm": "import vllm",
    "import_peft": "import peft",
    "transformers_preloaded": "import sys; assert not [m for m in sys.modules if m.split('.')[0] in ('transformers','vllm','peft','accelerate','fla')], 'model library already loaded'",
}
OK_ESCAPES = ("gpu_torch", "whoami", "workdir_write", "practice_read", "tokenizer", "transformers_preloaded")


PARITY_ITEMS = [
    {"messages": [{"role": "user", "content": "Weather in Paris? \u00e9\u00e8 \ud55c\uad6d\uc5b4"}],
     "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string", "default": "x"}},
                               "required": ["city"]}}}]},
    {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi there"}], "tools": []},
    {"messages": [{"role": "user", "content": "no tools here"}]},
]


def tok_parity(h):
    """Kernel TOK (tokenizers + jinja2) must render and tokenize exactly like the harness (HF tokenizer)."""
    import json as _j
    items = list(PARITY_ITEMS)
    for f in ("live_irrelevance.jsonl", "xlam_practice.jsonl"):
        p = h.workdir / "practice" / f
        if p.exists():
            for i, line in enumerate(open(p)):
                if i >= 40:
                    break
                r = _j.loads(line)
                items.append({"messages": [m for m in r["messages"] if m["role"] != "assistant"], "tools": r["tools"]})
    h._kernel(("set", "_parity_items", items))
    out = h.python("import json\n_r=[TOK.apply_chat_template(it['messages'], tools=it.get('tools') or None, tokenize=False, "
                   "add_generation_prompt=True, enable_thinking=False) for it in _parity_items]\n"
                   "open('_parity.json','w').write(json.dumps([[t, TOK.encode(t)] for t in _r]))\nprint(len(_r))")
    got = _j.loads((h.workdir / "_parity.json").read_text())
    bad = []
    for it, (text, ids) in zip(items, got):
        ref = h.tok.apply_chat_template(it["messages"], tools=it.get("tools") or None, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        ref_ids = h.tok(ref, add_special_tokens=False)["input_ids"]
        if text != ref or ids != ref_ids:
            bad.append({"text_equal": text == ref, "ids_equal": ids == ref_ids, "ref_head": ref[:200], "got_head": text[:200]})
    return {"n": len(items), "mismatches": len(bad), "examples": bad[:3], "ok": not bad and len(got) == len(items),
            "kernel_output": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--practice", action="store_true", help="copy the Exp A practice files (tokenizer parity on real items)")
    a = ap.parse_args()
    sid = f"selftest_{int(time.time())}"
    guard_prompt = {"messages": [{"role": "user", "content": "This Prompt is   a held-out TEST item."}]}
    gpath = Path(f"/workspace/artifacts/sessions/{sid}_guard.json")
    gpath.parent.mkdir(parents=True, exist_ok=True)
    gpath.write_text(json.dumps({"sha256": [text_hash("this prompt is a held-out test item.")]}))
    h = Harness({"session_id": sid, "checkpoint": a.ckpt, "budget": 3, "max_prompts_per_pass": 4,
                 "max_new_tokens": 40, "test_guard": str(gpath),
                 "practice_files": {f: f"/workspace/artifacts/a/practice/{f}" for f in ("live_irrelevance.jsonl", "xlam_practice.jsonl")}
                 if a.practice else {}})
    h.start()
    R = {"session": sid}
    R["budget0"] = h.budget_state()
    R["sandbox"] = h.sandbox
    R["escapes"] = {k: h.python(v) for k, v in ESCAPES.items()}
    R["inspect"] = h.python("n = [k for k in W if 'layers.0.mlp.down_proj' in k][0]\n"
                            "print(n, W[n].shape, W[n].dtype, float(W[n].float().norm()))\n"
                            "print(len(W), MODEL_INFO['architecture'], MODEL_INFO['n_parameters'])")
    R["bad_assign"] = h.python("W['nope'] = 1")
    R["tok_parity"] = tok_parity(h)
    R["guard"] = h.forward_pass(prompts=["fine prompt", {"messages": [{"role": "system", "content": "x"},
                                                                    *guard_prompt["messages"]]}])
    R["gen_before"] = h.forward_pass(prompts=["What is 2+2? Answer with one number.",
                                              {"messages": [{"role": "user", "content": "Weather in Paris?"}],
                                               "tools": [{"type": "function", "function": {
                                                   "name": "get_weather", "description": "Get weather",
                                                   "parameters": {"type": "object", "properties": {
                                                       "city": {"type": "string"}}, "required": ["city"]}}}]}])
    # Zero every MLP down_proj -> the model's output must change.
    R["edit"] = h.python("for k in W:\n    if k.endswith('mlp.down_proj.weight') and 'language_model' in k:\n"
                         "        W[k].zero_()\nmy_prompts = ['What is 2+2? Answer with one number.']\nprint('ok')")
    R["gen_after"] = h.forward_pass(prompts_var="my_prompts")
    R["attention"] = h.forward_pass(prompts=["Hello there"], mode="attention")
    R["attn_in_kernel"] = h.python("print(sorted(last_attention)[:3], last_attention[sorted(last_attention)[0]].shape)")
    R["over_budget"] = h.forward_pass(prompts=["one more"])
    R["save"] = h.save_weights("zeroed language-model MLP down_proj")
    R["submit"] = h.submit("self-test")
    R["after_submit"] = h.budget_state()

    checks = {
        "escapes_refused": all(R["escapes"][k]["error"] for k in ESCAPES if k not in OK_ESCAPES),
        "model_libs_blocked": all(R["escapes"][k]["error"] for k in ("import_transformers", "import_vllm", "import_peft")),
        "no_model_lib_preloaded": not R["escapes"]["transformers_preloaded"]["error"],
        "kernel_tok_matches_harness": R.get("tok_parity", {}).get("ok", False),
        "tokenizer_ok": "hello world" in R["escapes"]["tokenizer"]["output"],
        "gpu_invisible": "False 0" in R["escapes"]["gpu_torch"]["output"],
        "not_root": not R["escapes"]["whoami"]["output"].startswith("0 "),
        "workdir_ok": "ok" in R["escapes"]["workdir_write"]["output"],
        "bad_assign_refused": bool(R["bad_assign"]["error"]),
        "guard_refused_uncharged": "error" in R["guard"] and R["gen_before"].get("budget_remaining") == 2,
        "edit_changes_output": R["gen_before"].get("completions", [None])[0] != R["gen_after"].get("completions", [None])[0],
        "budget_enforced": "exhausted" in R["over_budget"].get("error", ""),
        "attention_ok": bool(R["attention"].get("prompts", [{}])[0].get("layers_with_attention")),
        "diff_recorded": R["save"].get("tensors_changed", 0) > 0,
        "submit_freezes": "error" in R["after_submit"],
    }
    R["checks"] = checks
    out = Path("/workspace/artifacts/0.3"); out.mkdir(parents=True, exist_ok=True)
    (out / "harness_selftest.json").write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({"checks": checks, "escapes": {k: (v["error"] or v["output"])[-160:] for k, v in R["escapes"].items()},
                      "gen_before": R["gen_before"], "gen_after": R["gen_after"], "save": R["save"]}, indent=1, default=str))
    h.kproc.join(5)


if __name__ == "__main__":
    main()
