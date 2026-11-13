"""Harness check: a kernel that dies (e.g. OOM-killed) is restarted, W edits survive, no budget is charged.

    python -m scripts.test_kernel_restart      (on the pod, as root)
"""
import json
import os
import signal

from common.harness.tools import Harness


def main():
    h = Harness({"checkpoint": "/workspace/artifacts/a/broken", "budget": 4, "max_prompts_per_pass": 8,
                 "max_new_tokens": 8, "session_root": "/workspace/artifacts/a11c/selftest",
                 "session_id": "restart_selftest", "reference_checkpoint": "Qwen/Qwen3.5-2B", "activations": True})
    h.start(); h.ready.wait()
    assert not h.load_error, h.load_error
    res = {}
    n = "model.language_model.norm.weight"
    res["edit"] = h.python(f"W['{n}'][0] = 7.0\nx = 1\nprint(float(W['{n}'][0]))")
    os.kill(h.kproc.pid, signal.SIGKILL); h.kproc.join(10)
    res["after_kill_python"] = h.python("print('alive')")
    res["edit_survived"] = h.python(f"print(float(W['{n}'][0]), 'x' in globals(), 'W_BASE' in globals())")
    os.kill(h.kproc.pid, signal.SIGKILL); h.kproc.join(10)
    res["after_kill_forward"] = h.forward_pass(prompts=["hi"])
    res["budget_after_dead_forward"] = h.budget_state()
    res["forward_ok"] = h.forward_pass(prompts=["hi"])
    res["budget_final"] = h.budget_state()
    h.kproc.terminate()
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
