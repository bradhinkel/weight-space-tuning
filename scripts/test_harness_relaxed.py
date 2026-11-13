"""A.11b harness self-test: base model reference, activations mode, and that A.11a-style configs are unchanged.

    python -m scripts.test_harness_relaxed      (on the pod, as root)

Checks on the broken checkpoint with reference = Qwen/Qwen3.5-2B:
  * W_BASE is in the kernel, differs from W where the fine-tune changed weights, and editing W_BASE does not
    change forward_pass(model="base");
  * forward_pass(model="base") and model="current" give different answers on an over-call prompt;
  * mode="activations" (last and all positions) loads arrays of the expected shape into the kernel, and the
    final entry projected on the embedding gives the same first token as generation (checks the post-norm note);
  * the sandbox still blocks file and network escapes; budget is charged for every mode and model;
  * a non-relaxed Harness rejects model="base" and mode="activations".
Writes artifacts/a11/harness_selftest_relaxed.json.
"""
import json
from pathlib import Path

from common.harness.tools import Harness

PROMPT = {"messages": [{"role": "user", "content": "Tell me a joke about cats."}],
          "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
                     "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                    "required": ["city"]}}}]}


def main():
    res = {}
    base_cfg = {"checkpoint": "/workspace/artifacts/a/broken", "budget": 8, "max_prompts_per_pass": 64,
                "max_new_tokens": 48, "session_root": "/workspace/artifacts/a11/selftest"}
    h = Harness({**base_cfg, "session_id": "relaxed_selftest", "reference_checkpoint": "Qwen/Qwen3.5-2B",
                 "activations": True})
    h.start(); h.ready.wait()
    assert not h.load_error, h.load_error
    py = lambda c: h.python(c)
    res["w_base_present"] = py("print('W_BASE' in globals(), len(W_BASE) == len(W))")
    res["diff_frac"] = py("import torch\nn=[k for k in W if 'layers.10.mlp.down_proj' in k][0]\n"
                          "print(n, (W[n]!=W_BASE[n]).float().mean().item())")
    cur = h.forward_pass(prompts=[PROMPT])
    base = h.forward_pass(prompts=[PROMPT], model="base")
    res["current"] = cur.get("completions", cur)
    res["base"] = base.get("completions", base)
    py("W_BASE[list(W_BASE)[0]].zero_()")      # kernel copy only
    base2 = h.forward_pass(prompts=[PROMPT], model="base")
    res["base_unchanged_after_kernel_edit"] = base2.get("completions") == base.get("completions")
    act = h.forward_pass(prompts=[PROMPT, "Hello"], mode="activations")
    res["act_last"] = act
    res["act_kernel"] = py("print({k: v.shape for k, v in last_activations.items()})")
    # last entry @ embedding -> argmax should equal first generated token (if the final norm is applied)
    res["act_first_token_check"] = py(
        "import torch\nemb=[k for k in W if k.endswith('embed_tokens.weight')][0]\n"
        "h=torch.tensor(last_activations['p0'][-1,-1]).float()\n"
        "print(TOK.decode([int((W[emb].float()@h).argmax())]))\n"
        "print(repr(last_generations[0]['completion'][:20]) if 'last_generations' in globals() else '')")
    act_all = h.forward_pass(prompts=["Hello there"], mode="activations", positions="all")
    res["act_all"] = act_all
    res["escape_artifacts"] = py("open('/workspace/artifacts/a/data/split.json').read(10)")
    res["escape_net"] = py("import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)")
    res["budget"] = h.budget_state()
    h.kproc.terminate()

    g = Harness({**base_cfg, "session_id": "plain_selftest"})
    g.start(); g.ready.wait()
    res["plain_base_rejected"] = g.forward_pass(prompts=["hi"], model="base")
    res["plain_act_rejected"] = g.forward_pass(prompts=["hi"], mode="activations")
    res["plain_no_w_base"] = g.python("print('W_BASE' in globals())")
    res["plain_budget"] = g.budget_state()
    g.kproc.terminate()
    out = Path("/workspace/artifacts/a11/harness_selftest_relaxed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
