"""The driver's persistent Python kernel (child process, CPU only, confined by sandbox.confine).

The target's weights live in a few flat shared-memory buffers (one per dtype) owned by the harness
server. Inside the kernel they appear as `W`: a fixed-key mapping name -> torch tensor view into those
buffers. In-place edits (`W[n].mul_(0.9)`, `W[n][:, 3] = 0`) and assignment (`W[n] = new`, which copies
into the buffer) both modify the live weights that forward_pass runs. Keys and shapes cannot change.

Protocol (multiprocessing Pipe, request -> reply):
  ("exec", code)                -> {"output", "error"}
  ("get", name)                 -> {"value"} (JSON-able value of a kernel variable, for prompts_var)
  ("set", name, value)          -> {}       (harness posts results such as last_generations)
  ("load_npz", name, path)      -> {}
"""
import contextlib
import io
import signal
import traceback

MAX_OUTPUT = 20000


class Weights(dict):
    """dict of name -> tensor view; keys fixed, assignment copies into the shared buffer."""

    def __setitem__(self, k, v):
        if k not in self:
            raise KeyError(f"unknown parameter {k!r}; the parameter set is fixed")
        import torch
        cur = dict.__getitem__(self, k)
        v = torch.as_tensor(v)
        if tuple(v.shape) != tuple(cur.shape):
            raise ValueError(f"shape mismatch for {k}: {tuple(v.shape)} vs {tuple(cur.shape)}")
        cur.copy_(v.to(cur.dtype))

    def __delitem__(self, k):
        raise TypeError("parameters cannot be deleted")

    def pop(self, *a):
        raise TypeError("parameters cannot be removed")

    popitem = clear = update = setdefault = pop


def _views(flats, layout):
    w = Weights()
    for name, dtype, off, shape in layout:
        n = 1
        for s in shape:
            n *= s
        dict.__setitem__(w, name, flats[dtype][off:off + n].view(shape))
    return w


def main(conn, flats, layout, workdir, info, user, extra_read, tok_path=None):
    import os
    import sys
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)          # never write to the MCP server's stdout (JSON-RPC channel)
    os.dup2(devnull, 2)
    import json
    import math  # noqa: F401  (pre-import common libs before confinement)
    import numpy as np
    import torch
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    from common.harness.sandbox import confine

    W = _views(flats, layout)
    TOK = None
    if tok_path:  # loaded before confinement; afterwards the kernel cannot read the checkpoint dir
        # Not transformers: model-building libraries are unreadable in the kernel (sandbox.DENY_PACKAGES),
        # and importing one here, before confinement, would leave it usable in sys.modules.
        from common.harness.kernel_tok import KernelTokenizer
        TOK = KernelTokenizer(tok_path)
    try:
        applied = confine(workdir, user=user, extra_read=extra_read)
    except Exception as e:
        conn.send({"fatal": f"sandbox failed: {e!r}"})
        return
    conn.send({"ready": True, "sandbox": applied})

    ns = {"W": W, "TOK": TOK, "np": np, "torch": torch, "json": json, "MODEL_INFO": info, "__name__": "__driver__"}
    signal.signal(signal.SIGINT, signal.default_int_handler)

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        op = msg[0]
        try:
            if op == "exec":
                buf = io.StringIO()
                err = None
                try:
                    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                        try:
                            code = compile(msg[1], "<driver>", "exec")
                            # echo the value of a trailing expression, like a notebook cell
                            import ast
                            tree = ast.parse(msg[1])
                            if tree.body and isinstance(tree.body[-1], ast.Expr):
                                last = ast.Expression(tree.body.pop().value)
                                exec(compile(tree, "<driver>", "exec"), ns)
                                val = eval(compile(last, "<driver>", "eval"), ns)
                                if val is not None:
                                    print(repr(val))
                            else:
                                exec(code, ns)
                        except KeyboardInterrupt:
                            err = "KeyboardInterrupt: execution timed out and was interrupted"
                        except BaseException:
                            err = traceback.format_exc(limit=8)
                finally:
                    pass
                out = buf.getvalue()
                if len(out) > MAX_OUTPUT:
                    out = out[:MAX_OUTPUT] + f"\n... [truncated {len(out) - MAX_OUTPUT} chars]"
                conn.send({"output": out, "error": err})
            elif op == "get":
                v = ns.get(msg[1])
                conn.send({"value": json.loads(json.dumps(v))})
            elif op == "set":
                ns[msg[1]] = msg[2]
                conn.send({})
            elif op == "load_npz":
                with np.load(msg[2], allow_pickle=False) as z:
                    ns[msg[1]] = {k: z[k] for k in z.files}
                conn.send({})
            else:
                conn.send({"error": f"unknown op {op}"})
        except KeyboardInterrupt:
            conn.send({"output": "", "error": "KeyboardInterrupt"})
        except Exception as e:
            conn.send({"error": repr(e)})
