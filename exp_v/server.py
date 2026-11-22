"""Experiment V harness server (laptop): R.4a's tools and rules on the broken Waterbirds ResNet-50, over MCP stdio.

    python -m exp_v.server --config <V_ROOT>/sessions/<sid>/config.json      # MCP server (the driver launches it)
    python -m exp_v.server finalize <sid>                                    # after the session: score on test

Tools:
  python(code)            the confined kernel (exp_v/kernel.py)
  forward(request)        one pass on the GPU (costs 1 of the budget if it succeeds; a failed request costs nothing):
                          {"type": "evaluate", "model": "original" | "edited" | "both",
                           "items": "all" | [practice ids], "images": [files in work/] (optional, own images),
                           "activations": {"layers": [...], "spatial": false} (optional)}
  show_images(ids, crops) up to 16 practice images or work/ images as the model sees them (224x224 crop),
                          free, at most 400 per session
The session never reads test data. `finalize` (run by the driver after the session) scores the submission on test.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from exp_v.common import (ACT_LAYERS, BROKEN_PT, GROUPS, LIMITS, MODEL_DIR, PRACTICE, SESSIONS, edit_problem, score)

IMG_EXT = (".png", ".jpg", ".jpeg")


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def build_model_view():
    """V_ROOT/model/model.safetensors: the broken model's tensors as float32 (num_batches_tracked dropped)."""
    import torch
    from safetensors.torch import save_file
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sd = torch.load(BROKEN_PT, map_location="cpu")
    sd = {k: v.float().contiguous() for k, v in sd.items() if not k.endswith("num_batches_tracked")}
    save_file(sd, str(MODEL_DIR / "model.safetensors"))
    h = hashlib.sha256(open(MODEL_DIR / "model.safetensors", "rb").read()).hexdigest()
    (MODEL_DIR / "info.json").write_text(json.dumps({"source": str(BROKEN_PT), "sha256": h, "n_tensors": len(sd)}))
    return h


class GPUModel:
    """The broken model on the GPU, with practice images preloaded; runs original or edited weights."""

    def __init__(self, device="cuda"):
        import torch
        from exp_v import wb
        self.torch, self.device = torch, device
        self.orig = {k: v for k, v in torch.load(BROKEN_PT, map_location="cpu").items()}
        self.net = wb.resnet50(pretrained=False).to(device).eval()
        self.net.load_state_dict(self.orig)
        self.tf = wb.EVAL_TF
        import pandas as pd
        lab = pd.read_csv(PRACTICE / "labels.csv")
        self.ids = list(lab.id)
        self.meta = {r.id: (int(r.y), int(r.place), int(r.g)) for r in lab.itertuples()}
        from PIL import Image
        xs = [self.tf(Image.open(PRACTICE / "images" / f"{i}.jpg").convert("RGB")) for i in self.ids]
        self.X = torch.stack(xs).to(device)   # float32, as exp_v.wb scores
        self.index = {i: k for k, i in enumerate(self.ids)}
        self.loaded = None

    def load_weights(self, edits):
        key = hashlib.sha256(b"".join(n.encode() + hashlib.sha256(edits[n].tobytes()).digest()
                                      for n in sorted(edits))).hexdigest() if edits else "original"
        if key != self.loaded:
            sd = dict(self.orig)
            for n, a in edits.items():
                sd[n] = self.torch.from_numpy(a).to(sd[n].dtype)
            self.net.load_state_dict(sd)
            self.loaded = key
        return key

    def run(self, x, layers=(), spatial=False):
        """x: float32 [N,3,224,224] on device. Returns logits [N,2] and {layer: array}."""
        torch = self.torch
        acts, hooks = {}, []
        mods = dict(self.net.named_modules())
        for L in layers:
            def hook(m, i, o, L=L):
                o = o.float()
                v = o.flatten(1) if L == "avgpool" else (o if spatial else o.mean((2, 3)))
                acts.setdefault(L, []).append(v.half().cpu() if spatial and L != "avgpool" else v.cpu())
            hooks.append(mods[L].register_forward_hook(hook))
        out = []
        try:
            with torch.no_grad():
                for s in range(0, len(x), 128):
                    out.append(self.net(x[s:s + 128]).float().cpu())
        finally:
            for h in hooks:
                h.remove()
        return torch.cat(out).numpy(), {L: torch.cat(v).numpy() for L, v in acts.items()}


class VHarness:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sid = cfg["session_id"]
        self.budget = int(cfg["budget"])
        self.fitting = cfg.get("fitting", "refused")
        self.lim = {**LIMITS, **cfg.get("limits", {})}
        self.sdir = SESSIONS / self.sid
        self.sandbox = self.sdir / "sandbox"
        self.work = self.sandbox / "work"
        self.log_f = open(self.sdir / "calls.jsonl", "a", buffering=1)
        self.lock = threading.Lock()
        self.passes = self.runs = self.requests = self.shown = 0
        self.t0 = time.time()
        self._setup_sandbox()
        self._start_kernel()
        self.gpu = GPUModel()
        from safetensors import safe_open
        with safe_open(str(MODEL_DIR / "model.safetensors"), "np") as f:
            self.names = list(f.keys())
            self._orig_np = {n: f.get_tensor(n) for n in self.names}
        self.event("session_start", config=cfg, n_names=len(self.names), sandbox=self.sandbox_info)
        self._state()

    # ---------------------------------------------------------------- bookkeeping
    def event(self, kind, **kw):
        self.log_f.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **_jsonable(kw)}, default=str) + "\n")

    def _state(self):
        (self.sdir / "state.json").write_text(json.dumps({
            "budget": self.budget, "passes": self.passes, "python_runs": self.runs, "requests": self.requests,
            "images_shown": self.shown, "elapsed_s": round(time.time() - self.t0)}, indent=1))

    def _setup_sandbox(self):
        if self.sandbox.exists():
            shutil.rmtree(self.sandbox)
        (self.work / "edited").mkdir(parents=True)
        os.symlink(MODEL_DIR, self.sandbox / "model")
        os.symlink(PRACTICE, self.sandbox / "practice")
        for p in (self.sdir, self.sandbox):
            os.chmod(p, 0o700)

    # ---------------------------------------------------------------- kernel
    def _start_kernel(self):
        import multiprocessing as mp
        from exp_v import kernel
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        deny = ["sklearn"] if self.fitting == "refused" else []
        self.kproc = ctx.Process(target=kernel.main, daemon=True, args=(
            child, str(self.sandbox), str(self.work), deny, int(self.lim["kernel_mem_bytes"]),
            int(self.lim["max_output_chars"])))
        self.kproc.start()
        if not self.conn.poll(300):
            raise RuntimeError("kernel did not start")
        msg = self.conn.recv()
        if "fatal" in msg:
            raise RuntimeError(msg["fatal"])
        self.sandbox_info = msg["sandbox"]
        self.event("kernel_start", n_names=msg.get("n_names"))

    RESTART = ("The kernel was restarted ({why}); its variables are lost. Your saved edits in work/edited and "
               "all files in work/ are kept.")

    def _exec(self, code):
        if not self.kproc.is_alive():
            self._start_kernel()
            return {"output": "", "restart": self.RESTART.format(why="it had stopped, most likely the memory limit")}
        timed_out = False
        try:
            self.conn.send(("exec", code))
            if self.conn.poll(self.lim["exec_timeout_s"]):
                return self.conn.recv()
            timed_out = True
        except (BrokenPipeError, EOFError, ConnectionResetError, OSError):
            pass
        if timed_out and self.kproc.is_alive():
            os.kill(self.kproc.pid, signal.SIGINT)
            try:
                if self.conn.poll(30):
                    r = self.conn.recv()
                    r["output"] = (r.get("output") or "") + \
                        f"\n[interrupted after {self.lim['exec_timeout_s']} s; variables are kept]"
                    return r
            except (EOFError, OSError):
                pass
            self.kproc.kill()
        self.kproc.join(10)
        why = "the time limit" if timed_out else f"the memory limit or a crash, exit code {self.kproc.exitcode}"
        self._start_kernel()
        return {"output": "", "restart": self.RESTART.format(why=why)}

    def _compaction_notice(self):
        flag = self.sdir / "compacted.flag"
        if not flag.exists():
            return ""
        flag.unlink()
        notes = self.work / "notes.md"
        body = notes.read_text(errors="replace")[:20000] if notes.exists() else "(no work/notes.md yet)"
        self.event("compaction_notice")
        return "[The conversation was compacted. Your work/notes.md, as it is now:]\n" + body + "\n[end of notes]\n\n"

    # ---------------------------------------------------------------- edits
    def _edits(self, d=None):
        d = Path(d or self.work / "edited")
        out = {}
        files = sorted(f for f in os.listdir(d) if f.endswith(".npy") and not f.startswith(".")) if d.is_dir() else []
        for fn in files:
            n = fn[:-4]
            if n not in self._orig_np:
                raise ValueError(f"work/edited/{fn}: unknown tensor name")
            a = np.load(d / fn, allow_pickle=False).astype(np.float32)
            why = edit_problem(n, a, self._orig_np[n])
            if why:
                raise ValueError(f"work/edited/{fn}: {why}")
            out[n] = a
        return out

    # ---------------------------------------------------------------- tools
    def python(self, code: str) -> str:
        with self.lock:
            t0 = time.time()
            pre = self._compaction_notice()
            if self.runs >= self.lim["python_runs"]:
                self.event("python_refused", code=code)
                return pre + f"No Python runs left (at most {self.lim['python_runs']})."
            self.runs += 1
            r = self._exec(code)
            body = r.get("output") or ""
            text = pre + f"OUTPUT of Python run {self.runs} of at most {self.lim['python_runs']}:\n```\n{body}\n```"
            if r.get("restart"):
                text += "\n" + r["restart"]
            self.event("python", run=self.runs, code=code, out=body, restart=r.get("restart"),
                       s=round(time.time() - t0, 2))
            self._state()
            return text

    def forward(self, request) -> str:
        with self.lock:
            t0 = time.time()
            self.requests += 1
            pre = self._compaction_notice()
            try:
                if isinstance(request, str):
                    request = json.loads(request)
                if not isinstance(request, dict):
                    raise ValueError("the request must be a JSON object")
                if self.passes >= self.budget:
                    raise ValueError(f"the forward-pass budget is used up ({self.budget} of {self.budget})")
                if request.get("type") != "evaluate":
                    raise ValueError('"type" must be "evaluate"')
                res = self._evaluate(request, self.passes + 1)
                self.passes += 1
                res = {"ok": True, **res, "s": round(time.time() - t0, 1),
                       "passes_used": self.passes, "passes_left": self.budget - self.passes}
                if self.passes == self.budget:
                    res["note"] = ("This was your last forward pass. You can still run Python code to finish; then "
                                   "make your final choice (the tensors in work/edited are submitted) and reply "
                                   "without calling a tool, saying what you changed, the evidence for it and why "
                                   "you expect it to help.")
            except Exception as e:
                res = {"ok": False, "error": f"{type(e).__name__}: {e}", "s": round(time.time() - t0, 1),
                       "note": "A request that fails does not count against the budget.",
                       "passes_used": self.passes, "passes_left": self.budget - self.passes}
                self.event("forward_error", request=request, trace=traceback.format_exc(limit=6))
            self.event("forward", request=request, result=res)
            self._state()
            return pre + json.dumps(_jsonable(res), indent=1)

    def _own_images(self, files):
        import torch
        from PIL import Image
        xs, names = [], []
        for f in files:
            p = (self.sandbox / f).resolve()
            if not str(p).startswith(str(self.work.resolve()) + os.sep):
                raise ValueError(f"{f}: own images must be files in work/")
            if not p.is_file():
                raise ValueError(f"{f}: no such file")
            if p.suffix.lower() in IMG_EXT:
                xs.append(self.gpu.tf(Image.open(p).convert("RGB"))[None]); names.append(f)
            elif p.suffix.lower() == ".npy":
                a = np.load(p, allow_pickle=False).astype(np.float32)
                if a.ndim == 3:
                    a = a[None]
                if a.ndim != 4 or a.shape[1:] != (3, 224, 224):
                    raise ValueError(f"{f}: a .npy input must have shape [N, 3, 224, 224] (normalized model input)")
                xs.append(torch.from_numpy(a)); names += [f"{f}[{i}]" for i in range(len(a))]
            else:
                raise ValueError(f"{f}: own images must be .png, .jpg or .npy")
        x = torch.cat(xs) if xs else torch.zeros(0, 3, 224, 224)
        if len(x) > self.lim["own_images_max"]:
            raise ValueError(f"at most {self.lim['own_images_max']} own images per pass (got {len(x)})")
        return x.to(self.gpu.device, torch.float32), names

    def _evaluate(self, rq, k):
        import torch
        unknown = set(rq) - {"type", "model", "items", "images", "activations"}
        if unknown:
            raise ValueError(f"unknown keys {sorted(unknown)}")
        model = rq.get("model", "edited")
        if model not in ("original", "edited", "both"):
            raise ValueError('"model" must be "original", "edited" or "both"')
        items = rq.get("items", "all")
        if items == "all":
            items = list(self.gpu.ids)
        if not isinstance(items, list):
            raise ValueError('"items" must be "all" or a list of practice ids')
        bad = [i for i in items if i not in self.gpu.index]
        if bad:
            raise ValueError(f"unknown practice ids: {bad[:5]}")
        if len(set(items)) != len(items):
            raise ValueError("duplicate practice ids")
        own, own_names = self._own_images(rq.get("images") or [])
        if not items and not own_names:
            raise ValueError("no items or images")
        act = rq.get("activations") or {}
        layers = list(act.get("layers", []))
        spatial = bool(act.get("spatial", False))
        badl = [L for L in layers if L not in ACT_LAYERS]
        if badl:
            raise ValueError(f"unknown layers {badl}; allowed: {ACT_LAYERS}")
        if len(layers) > self.lim["act_layers_max"]:
            raise ValueError(f"at most {self.lim['act_layers_max']} activation layers per pass")
        if spatial and len(items) + len(own_names) > self.lim["spatial_max_items"]:
            raise ValueError(f"spatial activations for at most {self.lim['spatial_max_items']} images per pass")
        edits = self._edits() if model in ("edited", "both") else {}
        models = ["original", "edited"] if model == "both" else [model]
        x = torch.cat([self.gpu.X[[self.gpu.index[i] for i in items]], own]) if items else own
        fdir = self.work / f"fwd_{k}"
        fdir.mkdir()
        rows = [{"id": i, "y": self.gpu.meta[i][0], "place": self.gpu.meta[i][1], "group": GROUPS[self.gpu.meta[i][2]]}
                for i in items] + [{"id": n} for n in own_names]
        out, preds = {"pass": k, "folder": f"work/fwd_{k}", "n_items": len(items), "n_own_images": len(own_names)}, {}
        for m in models:
            self.gpu.load_weights(edits if m == "edited" else {})
            logits, acts = self.gpu.run(x, layers, spatial)
            np.save(fdir / f"{m}_logits.npy", logits.astype(np.float32))
            for L, a in acts.items():
                np.save(fdir / f"{m}_{L}.npy", a)
            pred = logits.argmax(1)
            preds[m] = pred
            for r, p, lg in zip(rows, pred, logits):
                r[f"pred_{m}"] = int(p)
                r[f"margin_{m}"] = round(float(lg[1] - lg[0]), 4)
            if items:
                y = np.array([self.gpu.meta[i][0] for i in items]); g = np.array([self.gpu.meta[i][2] for i in items])
                out[m] = score(pred[:len(items)], y, g)
            if own_names:
                out[f"{m}_own_images"] = {"n": len(own_names), "predicted_waterbird": int(pred[len(items):].sum())}
        if model == "both" and items:
            y = np.array([self.gpu.meta[i][0] for i in items]); g = np.array([self.gpu.meta[i][2] for i in items])
            po, pe = preds["original"][:len(items)] == y, preds["edited"][:len(items)] == y
            out["paired"] = {GROUPS[j]: {"diff_pts": round(100 * float(pe[g == j].mean() - po[g == j].mean()), 2),
                                         "wrong_to_right": int((~po & pe)[g == j].sum()),
                                         "right_to_wrong": int((po & ~pe)[g == j].sum()), "n": int((g == j).sum())}
                             for j in range(4) if (g == j).any()}
            if own_names:
                out["paired_own_images_changed"] = int((preds["original"][len(items):] != preds["edited"][len(items):]).sum())
        with open(fdir / "rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        (fdir / "index.json").write_text(json.dumps({"models": models, "layers": layers, "spatial": spatial,
                                                     "rows": "rows.jsonl (one per image, in order)",
                                                     "edited_tensors": sorted(edits)}, indent=1))
        out["files"] = sorted(os.listdir(fdir))
        out["edited_tensors"] = sorted(edits) if "edited" in models else []
        return out

    def show_images(self, ids, crops=None):
        from mcp.server.mcpserver import Image as McpImage
        from PIL import Image
        with self.lock:
            pre = self._compaction_notice()
            try:
                if not isinstance(ids, list) or not ids:
                    raise ValueError("ids must be a non-empty list")
                if len(ids) > self.lim["show_per_call"]:
                    raise ValueError(f"at most {self.lim['show_per_call']} images per call")
                if self.shown + len(ids) > self.lim["show_total"]:
                    raise ValueError(f"the session's image cap is {self.lim['show_total']} ({self.shown} shown)")
                if crops is not None and (not isinstance(crops, list) or len(crops) != len(ids)):
                    raise ValueError("crops must be a list with one entry per id ([x0, y0, x1, y1] in 224-pixel "
                                     "coordinates, or null)")
                imgs, labels = [], []
                for j, i in enumerate(ids):
                    if i in self.gpu.index:
                        im = Image.open(PRACTICE / "images" / f"{i}.jpg").convert("RGB")
                        y, pl, g = self.gpu.meta[i]
                        labels.append(f"{i}: {GROUPS[g]}")
                    else:
                        p = (self.sandbox / str(i)).resolve()
                        if not str(p).startswith(str(self.work.resolve()) + os.sep) or not p.is_file() \
                                or p.suffix.lower() not in IMG_EXT:
                            raise ValueError(f"{i}: not a practice id or an image file in work/")
                        im = Image.open(p).convert("RGB")
                        labels.append(f"{i}")
                    im = im.resize((256, 256), Image.BILINEAR).crop((16, 16, 240, 240))   # what the model sees
                    c = crops[j] if crops else None
                    if c:
                        x0, y0, x1, y1 = [int(v) for v in c]
                        if not (0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224):
                            raise ValueError(f"bad crop {c}")
                        im = im.crop((x0, y0, x1, y1))
                        labels[-1] += f" crop {c}"
                    s = 336 / max(im.size)
                    im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BICUBIC)
                    b = io.BytesIO(); im.save(b, "PNG")
                    imgs.append(McpImage(data=b.getvalue(), format="png"))
                self.shown += len(ids)
                head = (pre + f"{len(ids)} images (as the model sees them: resized to 256x256, center 224x224 crop; "
                        f"shown at up to 336 px). In order: " + "; ".join(labels) +
                        f". Images shown this session: {self.shown} of {self.lim['show_total']}.")
                self.event("show_images", ids=ids, crops=crops, shown=self.shown)
                self._state()
                return [head, *imgs]
            except Exception as e:
                self.event("show_images_error", ids=ids, error=repr(e))
                return [pre + f"show_images failed (costs nothing): {type(e).__name__}: {e}"]


# ---------------------------------------------------------------- finalize (scoring side)
def evaluate_on_test(edits):
    """Score the broken model with `edits` on the sealed test split (exp_v.wb). Not reachable from a session."""
    import torch
    from exp_v import wb
    m = wb.resnet50(pretrained=False)
    sd = torch.load(BROKEN_PT, map_location="cpu")
    for n, a in edits.items():
        sd[n] = torch.from_numpy(a).to(sd[n].dtype)
    m.load_state_dict(sd)
    _, logits, y, g = wb.features_and_logits(m, "test")
    s = score(logits.argmax(1), y, g)
    return s, logits


def finalize(sid):
    sdir = SESSIONS / sid
    cfg = json.loads((sdir / "config.json").read_text())
    h = object.__new__(VHarness)
    from safetensors import safe_open
    with safe_open(str(MODEL_DIR / "model.safetensors"), "np") as f:
        h._orig_np = {n: f.get_tensor(n) for n in f.keys()}
    src = sdir / "sandbox" / "work" / "edited"
    out = {"session_id": sid, "config": cfg}
    try:
        edits = h._edits(src)
    except Exception as e:
        out["error"] = f"invalid submission: {e}"
        (sdir / "final.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out))
        return 1
    sub = sdir / "submitted"
    if sub.exists():
        shutil.rmtree(sub)
    shutil.copytree(src, sub)
    test, logits = evaluate_on_test(edits)
    np.save(sdir / "test_logits.npy", logits.astype(np.float32))
    out.update({"edited_tensors": sorted(edits), "n_weights_changed": int(sum(
        int((a != h._orig_np[n]).sum()) for n, a in edits.items())), "test": test})
    (sdir / "final.json").write_text(json.dumps(_jsonable(out), indent=1))
    print(json.dumps(_jsonable({k: out[k] for k in ("session_id", "edited_tensors", "n_weights_changed", "test")})))
    return 0


PY_DOC = """Run Python code in your persistent kernel and return its printed output (a trailing expression is echoed).
Predefined: NAMES, SHAPES, load(name), load_original(name), save_edit(name, tensor), drop_edit(name), edited().
model/ and practice/ are read-only; work/ is read-write. No internet, no GPU, gradients disabled. Time, memory and
run limits are in the task."""
FWD_DOC = """Run one forward pass on the GPU. `request` is the JSON object described in the task ("type": "evaluate").
Each successful request costs one forward pass of your budget; a request that fails costs nothing. Returns the
pass summary as JSON; per-image results and activations are written to work/fwd_<k>/."""
SHOW_DOC = """Show up to 16 images as the model sees them (resized to 256x256, center 224x224 crop). `ids`: practice
ids (e.g. "p0007") or image files you wrote in work/ (e.g. "work/test1.png"). `crops` (optional): one
[x0, y0, x1, y1] box in 224-pixel coordinates (or null) per id, to zoom in. Free (costs no forward pass); at most
400 images per session."""


def serve(config):
    rpc_fd = os.dup(1)
    cfg = json.loads(open(config).read())
    sdir = SESSIONS / cfg["session_id"]
    errf = os.open(sdir / "server_stderr.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(errf, 2)
    os.dup2(errf, 1)
    sys.stdout = os.fdopen(rpc_fd, "w", buffering=1)
    from mcp.server.mcpserver import MCPServer
    h = VHarness(cfg)
    mcp = MCPServer("harness")

    def python(code: str) -> str:
        return h.python(code)
    python.__doc__ = PY_DOC
    mcp.tool()(python)

    def forward(request: dict | str) -> str:
        return h.forward(request)
    forward.__doc__ = FWD_DOC
    mcp.tool()(forward)

    def show_images(ids: list[str], crops: list | None = None):
        return h.show_images(ids, crops)
    show_images.__doc__ = SHOW_DOC
    mcp.tool()(show_images)
    mcp.run("stdio")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "finalize":
        return finalize(sys.argv[2])
    if len(sys.argv) > 1 and sys.argv[1] == "build_model_view":
        print(build_model_view())
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    serve(a.config)


if __name__ == "__main__":
    sys.exit(main())
