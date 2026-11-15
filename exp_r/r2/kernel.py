"""R.2 driver kernel: a persistent Python process with the original Experiment 8 API.

Predefined (as in the original task text): NAMES, load(name), load_original(name), save_edit(name, tensor),
drop_edit(name), edited(), tok. The weights are read from /model/model.safetensors (text-only view of
Qwen3.5-0.8B); edits are float32 .npy files in /work/edited/, which is what forward passes and the final
submission use. Gradients are disabled. Confined by common.harness.sandbox (unprivileged user, Landlock:
read /model, /practice and the Python libs, read-write /work only; no network; no GPU), with an address-space
limit (RLIMIT_AS) like the original's 96 GB.

Protocol (multiprocessing Pipe): ("exec", code) -> {"output", "error"}.
"""
import contextlib
import io
import signal
import traceback

PRELUDE = r'''
import json, os, math
import numpy as np
import torch
from safetensors import safe_open as _safe_open
np.set_printoptions(precision=4, suppress=True, linewidth=160)
torch.set_grad_enabled(False)
_MODEL = "/model/model.safetensors"
_EDITS = "/work/edited"
with _safe_open(_MODEL, "pt") as _f:
    NAMES = list(_f.keys())
_NAMESET = set(NAMES)


def load_original(name):
    """The original tensor as float32 (torch)."""
    if name not in _NAMESET:
        raise KeyError(f"unknown tensor {name!r} (see NAMES)")
    with _safe_open(_MODEL, "pt") as f:
        return f.get_tensor(name).float()


def load(name):
    """Your current edit of the tensor if there is one, else the original (float32 torch tensor)."""
    p = os.path.join(_EDITS, name + ".npy")
    if os.path.exists(p):
        return torch.from_numpy(np.load(p)).float()
    return load_original(name)


def save_edit(name, tensor):
    """Save an edited tensor (same shape; any array-like). It is used by every later forward pass with the
    edited model and is submitted at the end. Refused: the final norm, the input embedding / output layer, and
    edits that only multiply the whole tensor by a constant."""
    from _r2_common import edit_problem
    if name not in _NAMESET:
        raise KeyError(f"unknown tensor {name!r} (see NAMES)")
    a = tensor.detach().float().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor, np.float32)
    a = np.ascontiguousarray(a, dtype=np.float32)
    o = load_original(name).numpy()
    why = edit_problem(name, a, o)
    if why:
        raise ValueError("save_edit refused: " + why)
    os.makedirs(_EDITS, exist_ok=True)
    tmp = os.path.join(_EDITS, "." + name + ".tmp.npy")
    np.save(tmp, a)
    os.replace(tmp, os.path.join(_EDITS, name + ".npy"))
    rel = float(np.linalg.norm(a - o) / max(np.linalg.norm(o), 1e-30))
    return f"saved {name}: relative change {rel:.4g}"


def drop_edit(name):
    """Remove your edit of a tensor (it reverts to the original)."""
    p = os.path.join(_EDITS, name + ".npy")
    if os.path.exists(p):
        os.remove(p)
        return f"dropped {name}"
    return f"{name} had no edit"


def edited():
    """Names of the tensors you have edited."""
    if not os.path.isdir(_EDITS):
        return []
    return sorted(f[:-4] for f in os.listdir(_EDITS) if f.endswith(".npy") and not f.startswith("."))


from tokenizers import Tokenizer as _Tokenizer
tok = _Tokenizer.from_file("/model/tokenizer.json")


def _no_grad(*a, **k):
    raise RuntimeError("Gradients are disabled in this kernel (no backward passes, autograd or optimizers).")


torch.Tensor.backward = _no_grad
torch.autograd.backward = _no_grad
torch.autograd.grad = _no_grad
'''


def _ascii(s):
    return s.encode("ascii", "replace").decode("ascii")


def main(conn, workdir, user, extra_read, mem_bytes, max_out, common_path):
    import os
    import resource
    import sys
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)          # never write to the MCP server's stdout (JSON-RPC channel)
    os.dup2(devnull, 2)
    import math  # noqa: F401
    import numpy as np  # noqa: F401
    import torch
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_r2_common", common_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["_r2_common"] = mod
    for m in ("scipy", "sklearn", "pandas", "sympy", "networkx", "safetensors", "tokenizers"):
        try:
            __import__(m)
        except Exception:
            pass
    from common.harness.sandbox import confine
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        applied = confine(workdir, user=user, extra_read=extra_read)
        ns = {"__name__": "__driver__"}
        exec(compile(PRELUDE, "<prelude>", "exec"), ns)
    except Exception as e:
        conn.send({"fatal": f"kernel setup failed: {e!r}\n{traceback.format_exc(limit=5)}"})
        return
    conn.send({"ready": True, "sandbox": applied, "n_names": len(ns["NAMES"])})
    signal.signal(signal.SIGINT, signal.default_int_handler)

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] != "exec":
            conn.send({"error": f"unknown op {msg[0]}"})
            continue
        buf = io.StringIO()
        err = None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    import ast
                    tree = ast.parse(msg[1])
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last = ast.Expression(tree.body.pop().value)
                        exec(compile(tree, "<code>", "exec"), ns)
                        val = eval(compile(last, "<code>", "eval"), ns)
                        if val is not None:
                            print(repr(val))
                    else:
                        exec(compile(tree, "<code>", "exec"), ns)
                except KeyboardInterrupt:
                    err = "KeyboardInterrupt: the run exceeded its time limit and was interrupted"
                except MemoryError:
                    err = "MemoryError: the run exceeded the kernel's memory limit"
                except BaseException:
                    err = traceback.format_exc(limit=8)
        except KeyboardInterrupt:
            err = "KeyboardInterrupt"
        out = buf.getvalue()
        if err:
            out = (out + "\n" if out else "") + err
        out = _ascii(out)
        if len(out) > max_out:
            out = out[:max_out] + f"\n[output cut at {max_out:,} characters]"
        conn.send({"output": out, "error": bool(err)})
