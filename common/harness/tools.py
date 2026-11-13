"""Harness core: the five driver-facing tools (python, forward_pass, save_weights, budget, submit).

Weights: every parameter of the target lives in one flat buffer per dtype.
  * CPU flat buffers are shared memory; the driver's kernel sees them as `W` (views) and edits them.
  * The GPU model's parameters are views into GPU flat buffers; forward_pass syncs CPU -> GPU with one
    copy per dtype, so a forward pass always runs exactly the current weights.
  * An untouched copy of the starting weights is kept for diffs (save_weights / submit).

Test-set protection (PROJECT_PLAN rule 5), enforced here and in the sandbox:
  * only files listed in config["practice_files"] are copied into the kernel's working dir;
  * every forward_pass prompt is hashed and refused if it matches the test guard
    (config["test_guard"], built by exp_a/build_practice.py from the BFCL test categories).
"""
import json
import os
import pwd
import shutil
import signal
import threading
import time
import traceback
from pathlib import Path

import torch

from common.harness.budget import Budget, BudgetExhausted
from common.harness.guard import text_hash, user_hashes  # noqa: F401
from common.harness.logging import SessionLog

DEFAULTS = {
    "budget": 10, "max_prompts_per_pass": 64, "max_new_tokens": 512, "max_prompt_tokens": 8192,
    "attention_max_tokens": 1024, "attention_max_prompts": 8, "gen_batch_tokens": 131072,
    "exec_timeout_s": 900, "kernel_user": "wsdriver", "practice_files": {}, "test_guard": None,
    "session_root": "/workspace/artifacts/sessions", "work_root": "/srv/wst-sessions",
    # A.11b relaxations (all off by default, so A.5 / A.11a sessions are unchanged):
    "reference_checkpoint": None,   # base model: W_BASE in the kernel + forward_pass(model="base")
    "activations": False,           # forward_pass(mode="activations"): per-layer hidden states -> kernel
    "activations_max_prompts": 64, "activations_all_positions_max_prompts": 8,
}


def resolve_ckpt(ckpt: str) -> str:
    if Path(ckpt).is_dir():
        return str(Path(ckpt).resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(ckpt, local_files_only=True)


class Harness:
    def __init__(self, cfg: dict):
        self.cfg = {**DEFAULTS, **cfg}
        c = self.cfg
        self.sid = c["session_id"]
        self.sdir = Path(c["session_root"]) / self.sid
        self.log = SessionLog(self.sdir)
        self.budget = Budget(c["budget"], c["max_prompts_per_pass"])
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.load_error = None
        self.done = False
        self.n_snap = 0
        self.t_start = time.time()
        self.guard = set()
        if c["test_guard"]:
            g = json.loads(Path(c["test_guard"]).read_text())
            self.guard = set(g["sha256"])
        self.log.event("session_start", config=c, guard_size=len(self.guard))

    # ------------------------------------------------------------------ setup
    def start(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self._load_inner()
            self.ready.set()
        except Exception:
            self.load_error = traceback.format_exc()
            self.log.event("load_error", error=self.load_error)
            self.ready.set()

    def _load_inner(self):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        c = self.cfg
        t0 = time.time()
        self.path = resolve_ckpt(c["checkpoint"])
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.tok.padding_side = "left"
        cpu = AutoModelForImageTextToText.from_pretrained(self.path, dtype=torch.bfloat16)
        named = list(cpu.named_parameters())
        sizes = {}
        layout = []
        for n, p in named:
            dt = str(p.dtype).replace("torch.", "")
            off = sizes.get(dt, 0)
            layout.append((n, dt, off, tuple(p.shape)))
            sizes[dt] = off + p.numel()
        flats = {dt: torch.empty(sz, dtype=getattr(torch, dt)).share_memory_() for dt, sz in sizes.items()}
        with torch.no_grad():
            for (n, p), (_, dt, off, shape) in zip(named, layout):
                v = flats[dt][off:off + p.numel()].view(shape)
                v.copy_(p.data)
                p.data = v
        self.cpu_model, self.flats, self.layout = cpu, flats, layout
        self.orig = {dt: f.clone() for dt, f in flats.items()}
        self.n_params = sum(sizes.values())
        self.ref_flats = self.ref_kernel = None
        if c["reference_checkpoint"]:
            ref_path = resolve_ckpt(c["reference_checkpoint"])
            ref = AutoModelForImageTextToText.from_pretrained(ref_path, dtype=torch.bfloat16)
            rnamed = dict(ref.named_parameters())
            if set(rnamed) != {n for n, *_ in layout}:
                raise RuntimeError("reference checkpoint has a different parameter set")
            self.ref_flats = {dt: torch.empty_like(f) for dt, f in flats.items()}   # private: GPU base passes
            with torch.no_grad():
                for n, dt, off, shape in layout:
                    self.ref_flats[dt][off:off + rnamed[n].numel()].view(shape).copy_(rnamed[n].data)
            self.ref_kernel = {dt: f.clone().share_memory_() for dt, f in self.ref_flats.items()}  # kernel's W_BASE
            del ref, rnamed

        gpu = AutoModelForImageTextToText.from_pretrained(self.path, dtype=torch.bfloat16, device_map="cuda")
        gpu.eval()
        gnamed = dict(gpu.named_parameters())
        if set(gnamed) != {n for n, *_ in layout}:
            raise RuntimeError("GPU/CPU parameter name mismatch")
        self.gflats = {dt: f.to("cuda") for dt, f in flats.items()}
        with torch.no_grad():
            for n, dt, off, shape in layout:
                k = 1
                for s in shape:
                    k *= s
                gnamed[n].data = self.gflats[dt][off:off + k].view(shape)
        torch.cuda.empty_cache()
        self.gpu_model = gpu

        tcfg = getattr(cpu.config, "text_config", cpu.config)
        self.info = {
            "checkpoint": c["checkpoint"], "architecture": type(cpu).__name__,
            "n_parameters": self.n_params,
            "text_config": {k: v for k, v in tcfg.to_dict().items()
                            if isinstance(v, (int, float, str, bool, list)) and k != "architectures"},
            "parameters": [[n, dt, list(s)] for n, dt, _, s in layout],
            "chat_format": "Qwen3.5 chat template, thinking disabled (empty <think></think> pre-filled). "
                           "Tool calls are emitted as <tool_call><function=NAME><parameter=K>V</parameter>"
                           "</function></tool_call>.",
        }
        self._setup_workdir()
        self._start_kernel()
        self.log.event("loaded", seconds=round(time.time() - t0), n_parameters=self.n_params,
                       dtypes={dt: int(f.numel()) for dt, f in flats.items()}, workdir=str(self.workdir),
                       sandbox=self.sandbox)

    def _setup_workdir(self):
        c = self.cfg
        self.workdir = Path(c["work_root"]) / self.sid / "work"
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        (self.workdir / "practice").mkdir(parents=True)
        for name, src in c["practice_files"].items():
            shutil.copy(src, self.workdir / "practice" / name)
        pw = pwd.getpwnam(c["kernel_user"])
        for root, dirs, files in os.walk(self.workdir.parent):
            for x in [root] + [os.path.join(root, f) for f in files]:
                os.chown(x, pw.pw_uid, pw.pw_gid)
        os.chmod(self.workdir.parent, 0o700)

    def _start_kernel(self):
        import torch.multiprocessing as mp
        from common.harness import kernel
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        import sys
        extra = [p for p in sys.path if p.startswith(sys.prefix)]
        self.kproc = ctx.Process(target=kernel.main, daemon=True,
                                 args=(child, self.flats, self.layout, str(self.workdir), self.info,
                                       self.cfg["kernel_user"], extra, self.path, self.ref_kernel))
        self.kproc.start()
        if not self.conn.poll(600):
            raise RuntimeError("kernel did not start")
        msg = self.conn.recv()
        if "fatal" in msg:
            raise RuntimeError(msg["fatal"])
        self.sandbox = msg["sandbox"]

    def _kernel(self, msg, timeout=60):
        self.conn.send(msg)
        if self.conn.poll(timeout):
            return self.conn.recv()
        os.kill(self.kproc.pid, signal.SIGINT)
        if self.conn.poll(30):
            return self.conn.recv()
        self.kproc.kill()
        self.kproc.join(10)
        self._start_kernel()
        return {"output": "", "error": "Kernel was unresponsive and has been restarted. Variables were lost; "
                                       "the weights (W) and all edits to them are preserved."}

    # ------------------------------------------------------------------ tool plumbing
    def _guarded(self, tool, args, fn):
        t0 = time.time()
        err, result = None, None
        with self.lock:
            try:
                self.ready.wait()
                if self.load_error:
                    raise RuntimeError("harness failed to load; see server log")
                if self.done:
                    raise RuntimeError("session already submitted; no further tool calls are accepted")
                result = fn()
            except (BudgetExhausted, ValueError, RuntimeError, KeyError, TypeError) as e:
                err = f"{type(e).__name__}: {e}"
            except Exception:
                err = traceback.format_exc(limit=5)
            self.log.call(tool, args, result, time.time() - t0, self.budget.state(), err)
        if err:
            return {"error": err, "budget": self.budget.state()}
        return result

    # ------------------------------------------------------------------ tools
    def python(self, code: str):
        def run():
            r = self._kernel(("exec", code), timeout=self.cfg["exec_timeout_s"])
            return {"output": r.get("output", ""), "error": r.get("error")}
        return self._guarded("python", {"code": code}, run)

    def _items(self, prompts, prompts_var):
        if (prompts is None) == (prompts_var is None):
            raise ValueError("give exactly one of prompts or prompts_var")
        if prompts_var is not None:
            r = self._kernel(("get", prompts_var))
            if "error" in r:
                raise ValueError(f"could not read kernel variable {prompts_var!r}: {r['error']}")
            prompts = r["value"]
        if not isinstance(prompts, list):
            raise ValueError("prompts must be a list")
        items = []
        for p in prompts:
            if isinstance(p, str):
                p = {"messages": [{"role": "user", "content": p}]}
            if not isinstance(p, dict) or not isinstance(p.get("messages"), list) or not p["messages"]:
                raise ValueError("each prompt is a string or {'messages': [...], 'tools': [...]} ")
            for m in p["messages"]:
                if m.get("role") not in ("system", "user", "assistant", "tool"):
                    raise ValueError(f"bad role {m.get('role')!r}")
            items.append({"messages": p["messages"], "tools": p.get("tools") or None})
        return items

    def _render(self, item):
        text = self.tok.apply_chat_template(item["messages"], tools=item["tools"], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.cfg["max_prompt_tokens"]:
            raise ValueError(f"prompt of {len(ids)} tokens exceeds max_prompt_tokens={self.cfg['max_prompt_tokens']}")
        return ids

    def _sync_gpu(self, model="current"):
        src = self.flats if model == "current" else self.ref_flats
        with torch.no_grad():
            for dt, f in src.items():
                self.gflats[dt].copy_(f)
        torch.cuda.synchronize()

    def forward_pass(self, prompts=None, prompts_var=None, mode="generate", max_new_tokens=None,
                     model="current", positions="last"):
        args = {"prompts": prompts, "prompts_var": prompts_var, "mode": mode, "max_new_tokens": max_new_tokens,
                "model": model, "positions": positions}

        def run():
            modes = ("generate", "attention") + (("activations",) if self.cfg["activations"] else ())
            if mode not in modes:
                raise ValueError(f"mode must be one of {modes}")
            if model not in (("current", "base") if self.ref_flats is not None else ("current",)):
                raise ValueError("model must be 'current'" + (" or 'base'" if self.ref_flats is not None else ""))
            items = self._items(prompts, prompts_var)
            blocked = [i for i, it in enumerate(items) if any(x in self.guard for x in user_hashes(it))]
            if blocked:
                self.log.event("guard_block", indices=blocked)
                raise ValueError(f"prompts {blocked} match held-out test items and are not allowed "
                                 "(no forward pass was charged)")
            if mode == "attention" and len(items) > self.cfg["attention_max_prompts"]:
                raise ValueError(f"attention mode takes at most {self.cfg['attention_max_prompts']} prompts")
            if mode == "activations":
                if positions not in ("last", "all"):
                    raise ValueError("positions must be 'last' or 'all'")
                lim = self.cfg["activations_all_positions_max_prompts"] if positions == "all" \
                    else self.cfg["activations_max_prompts"]
                if len(items) > lim:
                    raise ValueError(f"activations with positions={positions!r} take at most {lim} prompts")
            ids = [self._render(it) for it in items]
            remaining = self.budget.charge(len(items))
            self._sync_gpu(model)
            if mode == "activations":
                return {**self._activations(ids, positions, model), "budget_remaining": remaining}
            if mode == "generate":
                outs = self._generate(ids, int(max_new_tokens or self.cfg["max_new_tokens"]))
                self._kernel(("set", "last_generations",
                              [{"prompt": it, "completion": o} for it, o in zip(items, outs)]))
                return {"mode": "generate", "model": model, "completions": outs, "budget_remaining": remaining,
                        "note": "also stored in the kernel as last_generations"}
            return {**self._attention(ids), "budget_remaining": remaining, "model": model}
        return self._guarded("forward_pass", args, run)

    @torch.no_grad()
    def _generate(self, ids, max_new):
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        outs = [None] * len(ids)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        stop = {self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None}
        i = 0
        while i < len(order):
            batch = [order[i]]
            while i + len(batch) < len(order) and \
                    (len(batch) + 1) * (len(ids[order[i + len(batch)]]) + max_new) <= self.cfg["gen_batch_tokens"]:
                batch.append(order[i + len(batch)])
            i += len(batch)
            L = max(len(ids[j]) for j in batch)
            inp = torch.full((len(batch), L), pad, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for r, j in enumerate(batch):
                inp[r, L - len(ids[j]):] = torch.tensor(ids[j]); att[r, L - len(ids[j]):] = 1
            gen = self.gpu_model.generate(input_ids=inp.cuda(), attention_mask=att.cuda(), do_sample=False,
                                          max_new_tokens=max_new, pad_token_id=pad,
                                          eos_token_id=sorted(stop), temperature=None, top_p=None, top_k=None)
            for r, j in enumerate(batch):
                new = gen[r, L:].tolist()
                cut = next((k for k, t in enumerate(new) if t in stop), len(new))
                outs[j] = self.tok.decode(new[:cut], skip_special_tokens=False)
        return outs

    @torch.no_grad()
    def _attention(self, ids):
        import numpy as np
        m = self.gpu_model
        m.set_attn_implementation("eager")
        arrays, summary, tokens = {}, [], []
        try:
            for i, x in enumerate(ids):
                x = x[-self.cfg["attention_max_tokens"]:]
                out = m(input_ids=torch.tensor([x]).cuda(), output_attentions=True)
                layers = []
                for L, a in enumerate(out.attentions or []):
                    if a is None:
                        continue
                    arrays[f"p{i}_layer{L}"] = a[0].float().cpu().numpy().astype(np.float16)
                    layers.append(L)
                tokens.append(self.tok.convert_ids_to_tokens(x))
                summary.append({"prompt": i, "n_tokens": len(x), "layers_with_attention": layers})
        finally:
            m.set_attn_implementation("sdpa")
        path = self.workdir / f"attention_{self.budget.used}.npz"
        np.savez(path, **arrays)
        pw = pwd.getpwnam(self.cfg["kernel_user"])
        os.chown(path, pw.pw_uid, pw.pw_gid)
        self._kernel(("load_npz", "last_attention", str(path)), timeout=300)
        self._kernel(("set", "last_attention_tokens", tokens))
        return {"mode": "attention", "prompts": summary,
                "note": "arrays in kernel: last_attention['p{i}_layer{L}'] -> [heads, T, T] float16; "
                        "tokens in last_attention_tokens[i]. Linear-attention (Gated DeltaNet) layers "
                        "have no attention matrix and are omitted."}

    @torch.no_grad()
    def _activations(self, ids, positions, model):
        import numpy as np
        arrays, summary = {}, []
        for i, x in enumerate(ids):
            if positions == "all":
                x = x[-self.cfg["attention_max_tokens"]:]
            out = self.gpu_model(input_ids=torch.tensor([x]).cuda(), output_hidden_states=True)
            hs = torch.stack([h[0] for h in out.hidden_states])          # [n_layers+1, T, d]
            hs = hs[:, -1:, :] if positions == "last" else hs
            arrays[f"p{i}"] = hs.float().cpu().numpy().astype(np.float16)
            summary.append({"prompt": i, "n_tokens": len(x), "shape": list(arrays[f"p{i}"].shape)})
        path = self.workdir / f"activations_{self.budget.used}.npz"
        np.savez(path, **arrays)
        pw = pwd.getpwnam(self.cfg["kernel_user"])
        os.chown(path, pw.pw_uid, pw.pw_gid)
        self._kernel(("load_npz", "last_activations", str(path)), timeout=300)
        self._kernel(("set", "last_activations_tokens", [self.tok.convert_ids_to_tokens(x if positions == "all"
                      else x[-1:]) for x in ids]))
        return {"mode": "activations", "model": model, "positions": positions, "prompts": summary,
                "note": "arrays in kernel: last_activations['p{i}'] -> [n_layers+1, n_positions, hidden] float16 "
                        "(index 0 = embeddings, index k = residual stream after decoder layer k; the last entry "
                        "is after the final norm). Tokens in last_activations_tokens[i]."}

    def _diff(self):
        rows, touched = [], 0
        for n, dt, off, shape in self.layout:
            k = 1
            for s in shape:
                k *= s
            cur, base = self.flats[dt][off:off + k], self.orig[dt][off:off + k]
            ne = int((cur != base).sum())
            if ne:
                d = cur.float() - base.float()
                rows.append({"name": n, "shape": list(shape), "n_changed": ne, "numel": k,
                             "delta_fro": float(d.norm()), "base_fro": float(base.float().norm()),
                             "max_abs_delta": float(d.abs().max())})
                touched += ne
        return rows, touched

    def save_weights(self, note: str, _final=False):
        def run():
            from safetensors.torch import save_file
            rows, touched = self._diff()
            self.n_snap += 1
            tag = "final" if _final else f"snap_{self.n_snap:03d}"
            snap = self.sdir / "snapshots"
            snap.mkdir(exist_ok=True)
            tensors = {}
            by = {n: (dt, off, shape) for n, dt, off, shape in self.layout}
            for r in rows:
                dt, off, shape = by[r["name"]]
                tensors[r["name"]] = self.flats[dt][off:off + r["numel"]].view(shape).clone()
            if tensors:
                save_file(tensors, str(snap / f"{tag}.safetensors"))
            rec = {"tag": tag, "note": note, "t": time.time(), "forward_passes_used": self.budget.used,
                   "weights_touched": touched, "n_parameters": self.n_params,
                   "pct_touched": 100.0 * touched / self.n_params, "tensors": rows}
            (snap / f"{tag}.json").write_text(json.dumps(rec, indent=1))
            top = sorted(rows, key=lambda r: -r["n_changed"])[:20]
            return {"snapshot": tag, "weights_touched": touched, "pct_of_all_parameters": rec["pct_touched"],
                    "tensors_changed": len(rows),
                    "largest": [{k: r[k] for k in ("name", "n_changed", "delta_fro", "base_fro")} for r in top]}
        return self._guarded("save_weights", {"note": note, "final": _final}, run)

    def budget_state(self):
        return self._guarded("budget", {}, lambda: self.budget.state())

    def submit(self, final_note: str):
        snap = self.save_weights(final_note, _final=True)
        if "error" in snap:
            return snap

        def run():
            out = self.sdir / "final_checkpoint"
            self.cpu_model.save_pretrained(out, safe_serialization=True)
            self.tok.save_pretrained(out)
            try:
                from transformers import AutoProcessor
                AutoProcessor.from_pretrained(self.path).save_pretrained(out)
            except Exception as e:
                self.log.event("processor_save_skipped", error=repr(e))
            self.done = True
            summary = {"session_id": self.sid, "final_note": final_note, "final_checkpoint": str(out),
                       "seconds": round(time.time() - self.t_start), "budget": self.budget.state(),
                       "weights_touched": snap["weights_touched"], "pct_touched": snap["pct_of_all_parameters"],
                       "tool_calls_logged": self.log.n}
            self.log.write_json("summary.json", summary)
            try:
                self.conn.send(("exec", "pass"))
                self.kproc.terminate()
            except Exception:
                pass
            return {"submitted": True, "weights_touched": snap["weights_touched"],
                    "pct_of_all_parameters": snap["pct_of_all_parameters"],
                    "message": "Session ended. Final weights are frozen."}
        return self._guarded("submit", {"final_note": final_note}, run)
