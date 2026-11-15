"""R.2 harness server: the original Experiment 8 tools for a Claude Code driver, over MCP (stdio via SSH).

    python -m exp_r.r2.server --config <sessions>/<sid>/config.json          # MCP server (driver launches it)
    python -m exp_r.r2.server finalize <sid>                                 # after the session: submission

Tools (what the driver sees):
  python(code)      one code block in the persistent kernel; at most 300 runs, 900 s, output cut at 10,000 chars
  forward(request)  one forward pass, JSON as in the original ("generate" or "teacher_forced"); budget in passes

Paths for the driver: /model (text-only view of Qwen3.5-0.8B, read-only), /practice (their practice files,
read-only), /work (read-write: edits in /work/edited, pass results gen_<k>*, fwd_<k>/, notes.md).
"""
import argparse
import hashlib
import json
import os
import pwd
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from exp_r.r2.common import (BASE_CKPT, BENCH, FROZEN, FULL_ATTN, LIMITS, MODEL_VIEW, N_LAYERS, PRACTICE, REPO,
                             SESSIONS, WORK_ROOT, TestGuard, edit_problem, seed_for, user_message)

KUSER = "wsdriver"
CKPT_ROOT = "/srv/wst-r2-ckpt"         # root-only, container disk


def _chown_tree(p, uid, gid):
    for root, dirs, files in os.walk(p):
        os.chown(root, uid, gid)
        for f in files:
            os.chown(os.path.join(root, f), uid, gid)


def _jsonable(x):
    return json.loads(json.dumps(x, default=float))


class R2Harness:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sid = cfg["session_id"]
        self.budget = int(cfg["budget"])
        self.lim = {**LIMITS, **cfg.get("limits", {})}
        self.sdir = Path(SESSIONS) / self.sid
        self.sdir.mkdir(parents=True, exist_ok=True)
        self.log_f = open(self.sdir / "calls.jsonl", "a", buffering=1)
        self.lock = threading.Lock()
        self.passes = self.runs = self.requests = 0
        self.t0 = time.time()
        self.pw = pwd.getpwnam(KUSER)
        self.guard = TestGuard(BENCH)
        self.practice = {}
        for f in sorted(os.listdir(PRACTICE)):
            if f.endswith(".jsonl"):
                for line in open(os.path.join(PRACTICE, f)):
                    r = json.loads(line)
                    self.practice[r["id"]] = r
        from safetensors import safe_open
        with safe_open(os.path.join(MODEL_VIEW, "model.safetensors"), "np") as f:
            self.names = list(f.keys())
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(BASE_CKPT)
        self._setup_work()
        self._start_kernel()
        from exp_r.r2.tf_engine import TFEngine
        self.tf = TFEngine(BASE_CKPT, batch_tokens=int(cfg.get("tf_batch_tokens", 32768)))
        self.event("session_start", config=cfg, n_names=len(self.names), n_practice=len(self.practice))
        self._state()

    # ---------------------------------------------------------------- bookkeeping
    def event(self, kind, **kw):
        self.log_f.write(json.dumps({"t": round(time.time(), 3), "kind": kind, **kw}, default=str) + "\n")

    def _state(self):
        (self.sdir / "state.json").write_text(json.dumps({
            "budget": self.budget, "passes": self.passes, "python_runs": self.runs, "requests": self.requests,
            "elapsed_s": round(time.time() - self.t0)}, indent=1))

    def _setup_work(self):
        self.work = Path(WORK_ROOT) / self.sid / "work"
        if self.work.exists():
            shutil.rmtree(self.work)
        (self.work / "edited").mkdir(parents=True)
        msg = self.tok.apply_chat_template([{"role": "user", "content": user_message("<PROBLEM>")}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=True)
        (self.work / "eval_prompt_format.txt").write_text(
            "The evaluation prompt (and the prompt of every generation pass on a practice problem), with the "
            "problem text in place of <PROBLEM>; the model's answer follows it:\n" + msg)
        _chown_tree(self.work.parent, self.pw.pw_uid, self.pw.pw_gid)
        os.chmod(self.work.parent, 0o700)
        tmp = Path("/work.tmp")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(self.work)
        os.replace(tmp, "/work")                     # /work -> this session's work dir

    def _own(self, p):
        p = Path(p)
        if p.is_dir():
            _chown_tree(p, self.pw.pw_uid, self.pw.pw_gid)
        else:
            os.chown(p, self.pw.pw_uid, self.pw.pw_gid)

    # ---------------------------------------------------------------- kernel
    def _start_kernel(self):
        import multiprocessing as mp
        from exp_r.r2 import kernel
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        extra = [p for p in sys.path if p.startswith(sys.prefix)] + [os.path.realpath(MODEL_VIEW),
                                                                       os.path.realpath(PRACTICE)]
        self.kproc = ctx.Process(target=kernel.main, daemon=True, args=(
            child, str(self.work), KUSER, extra, int(self.lim["kernel_mem_bytes"]), int(self.lim["max_output_chars"]),
            os.path.join(REPO, "exp_r", "r2", "common.py")))
        self.kproc.start()
        if not self.conn.poll(600):
            raise RuntimeError("kernel did not start")
        msg = self.conn.recv()
        if "fatal" in msg:
            raise RuntimeError(msg["fatal"])
        self.sandbox = msg["sandbox"]
        self.event("kernel_start", n_names=msg.get("n_names"))

    RESTART = ("The kernel was restarted ({why}); its variables are lost. Your saved edits in /work/edited and "
               "all files in /work are kept.")

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
        if timed_out and self.kproc.is_alive():             # time limit: interrupt, then kill
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
        body = notes.read_text(errors="replace")[:20000] if notes.exists() else "(no /work/notes.md yet)"
        self.event("compaction_notice")
        return ("[The conversation was compacted. Your /work/notes.md, as it is now:]\n" + body + "\n[end of notes]\n\n")

    # ---------------------------------------------------------------- tools
    def python(self, code: str) -> str:
        with self.lock:
            t0 = time.time()
            pre = self._compaction_notice()
            if self.runs >= self.lim["python_runs"]:
                out = pre + f"No Python runs left (at most {self.lim['python_runs']})."
                self.event("python_refused", code=code)
                return out
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
                kind = request.get("type")
                if kind == "generate":
                    res = self._generate(request)
                elif kind == "teacher_forced":
                    res = self._teacher_forced(request)
                else:
                    raise ValueError('"type" must be "generate" or "teacher_forced"')
                self.passes += 1
                res = {"ok": True, **res, "s": round(time.time() - t0, 1),
                       "passes_used": self.passes, "passes_left": self.budget - self.passes}
                if self.passes == self.budget:
                    res["note"] = ("This was your last forward pass. You can still run Python code to finish; then "
                                   "make your final choice (the tensors in /work/edited are submitted) and reply "
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

    # ---------------------------------------------------------------- edits and checkpoints
    def _edits(self):
        """Validated current edits: name -> float32 array (the same refusals as save_edit)."""
        from safetensors import safe_open
        d = self.work / "edited"
        out = {}
        files = sorted(f for f in os.listdir(d) if f.endswith(".npy") and not f.startswith(".")) if d.is_dir() else []
        with safe_open(os.path.join(MODEL_VIEW, "model.safetensors"), "pt") as f:
            for fn in files:
                n = fn[:-4]
                if n not in self.names:
                    raise ValueError(f"/work/edited/{fn}: unknown tensor name")
                a = np.load(d / fn, allow_pickle=False).astype(np.float32)
                why = edit_problem(n, a, f.get_tensor(n).float().numpy())
                if why:
                    raise ValueError(f"/work/edited/{fn}: {why}")
                out[n] = a
        return out

    def _ckpt(self, edits):
        if not edits:
            return BASE_CKPT
        h = hashlib.sha256()
        for n in sorted(edits):
            h.update(n.encode())
            h.update(edits[n].tobytes())
        # one fixed path on local disk (fast to write and load; vLLM's compile cache is reused across passes)
        out = Path(CKPT_ROOT) / self.sid / "edited"
        tag = out / "edits_hash.txt"
        if not (tag.exists() and tag.read_text() == h.hexdigest()):
            shutil.rmtree(out, ignore_errors=True)
            build_ckpt(edits, out)
            tag.write_text(h.hexdigest())
        return str(out)

    # ---------------------------------------------------------------- generate
    def _generate(self, rq):
        L = self.lim
        model = rq.get("model")
        if model not in ("original", "edited", "both"):
            raise ValueError('"model" must be "original", "edited" or "both"')
        probs, prompts = rq.get("problems") or [], rq.get("prompts") or []
        if not isinstance(probs, list) or not isinstance(prompts, list) or not (probs or prompts):
            raise ValueError('give "problems" (practice ids) and/or "prompts" (your own user messages) as lists')
        unknown = [p for p in probs if p not in self.practice]
        if unknown:
            raise ValueError(f"unknown practice ids: {unknown[:5]}")
        if any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError('"prompts" must be non-empty strings')
        bad = [i for i, p in enumerate(prompts) if self.guard.hit(p)]
        if bad:
            raise ValueError(f"prompts {bad[:10]} contain text of an evaluation problem and are not allowed")
        n = int(rq.get("n", 1))
        if n < 1:
            raise ValueError('"n" must be at least 1')
        n_items = (len(probs) + len(prompts)) * n
        cap = L["gen_both_max"] if model == "both" else L["gen_max"]
        if n_items > cap:
            raise ValueError(f"items x n = {n_items} exceeds {cap}" + (" per model with \"both\"" if model == "both" else ""))
        mt = int(rq.get("max_tokens", L["max_tokens"]))
        if not 1 <= mt <= L["max_tokens"]:
            raise ValueError(f'"max_tokens" must be between 1 and {L["max_tokens"]}')
        acts = rq.get("activations")
        layers, mlp_on = [], False
        if acts:
            layers = sorted(set(int(x) for x in acts.get("layers", [])))
            mlp_on = bool(acts.get("mlp", False))
            if not layers or any(not 0 <= l <= N_LAYERS for l in layers):
                raise ValueError(f'"activations": "layers" must be indices 0..{N_LAYERS}')
        edits = self._edits() if model in ("edited", "both") else {}
        if model in ("edited", "both") and not edits:
            raise ValueError('there are no edits yet (save_edit first, or use "model": "original")')

        items = []
        for pid in probs:
            r = self.practice[pid]
            items.append({"key": pid, "id": pid, "source": r.get("source"), "user": user_message(r["problem"]),
                          "answer": r["answer"]})
        for i, p in enumerate(prompts):
            items.append({"key": "prompt:" + p, "id": f"prompt_{i}", "source": "prompt", "user": p, "answer": None})
        rows = [{**it, "sample": s} for it in items for s in range(n)]
        rendered = [self.tok.apply_chat_template([{"role": "user", "content": r["user"]}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=True) for r in rows]
        k = self.passes + 1
        models = ["original", "edited"] if model == "both" else [model]
        results, files, act_info = {}, {}, {}
        for m in models:
            ck = BASE_CKPT if m == "original" else self._ckpt(edits)
            gen = self._run_worker(ck, rendered, [seed_for(r["key"], r["sample"]) for r in rows], mt, k, m)
            graded = self._grade(rows, gen)
            name = f"gen_{k}.jsonl" if model != "both" else f"gen_{k}_{m}.jsonl"
            with open(self.work / name, "w") as f:
                for g in graded:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")
            self._own(self.work / name)
            files[m] = f"/work/{name}"
            results[m] = (graded, gen)
            if layers:
                self.tf.set_edits(edits if m == "edited" else {})
                try:
                    act_info[m] = self._gen_acts(k, m if model == "both" else None, graded, gen, layers, mlp_on)
                finally:
                    self.tf.set_edits({})
        out = {"type": "generate", "model": model}
        if model != "both":
            out.update({"file": files[model], **self._summary(results[model][0])})
            out["activations"] = act_info.get(model)
        else:
            out.update({"files": files, "activations": act_info or None,
                        "original": self._summary(results["original"][0]),
                        "edited": self._summary(results["edited"][0]),
                        "paired": self._paired(results["original"][0], results["edited"][0])})
        return out

    def _run_worker(self, ckpt, prompts, seeds, mt, k, m):
        wd = self.sdir / "passes" / f"{k}_{m}"
        wd.mkdir(parents=True, exist_ok=True)
        req = {"ckpt": ckpt, "out": str(wd / "gen.jsonl"), "ids_out": str(wd / "ids.jsonl"),
               "mem": float(self.cfg.get("vllm_mem", 0.8)), "keep_ids": self.lim["act_answer_tokens"],
               "items": [{"prompt": p, "seed": s, "max_tokens": mt} for p, s in zip(prompts, seeds)]}
        (wd / "request.json").write_text(json.dumps(req))
        env = {**os.environ, "HF_HUB_OFFLINE": "1",
               "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")}   # ninja for JIT kernels
        with open(wd / "worker.log", "w") as lf:
            p = subprocess.run([sys.executable, "-m", "exp_r.r2.gen_worker", str(wd / "request.json")], cwd=REPO,
                               env=env, stdout=lf, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            tail = (wd / "worker.log").read_text(errors="replace")[-1500:]
            self.event("worker_failed", pass_k=k, model=m, tail=tail)
            raise RuntimeError("the generation server failed (no pass was charged); details are in the harness log")
        gen = [json.loads(l) for l in open(wd / "gen.jsonl")]
        ids = [json.loads(l) for l in open(wd / "ids.jsonl")]
        bad = [g["finish_reason"] for g in gen if g["finish_reason"] not in ("stop", "length")]
        if bad:
            raise RuntimeError(f"{len(bad)} generations ended abnormally (no pass was charged)")
        for g, i in zip(gen, ids):
            g["_ids"] = i
        stats = (wd / "worker.log").read_text(errors="replace").strip().splitlines()
        self.event("worker", pass_k=k, model=m, stats=stats[-1] if stats else None)
        return gen

    def _grade(self, rows, gen):
        import mathqa as Q
        out = []
        for r, g in zip(rows, gen):
            pred, _ = Q.extract(g["text"])
            correct = bool(Q.is_correct(pred, r["answer"])) if r["answer"] is not None else None
            out.append({"id": r["id"], "source": r["source"], "sample": r["sample"], "prompt": r["user"],
                        "text": g["text"], "n_tokens": g["n_tokens"], "cut": g["finish_reason"] == "length",
                        "extracted_answer": pred, "reference_answer": r["answer"], "correct": correct})
        return out

    @staticmethod
    def _summary(g):
        gr = [x for x in g if x["correct"] is not None]
        by = {}
        for x in gr:
            by.setdefault(x["source"], []).append(x["correct"])
        return {"generations": len(g), "graded": len(gr),
                "accuracy": (sum(x["correct"] for x in gr) / len(gr)) if gr else None,
                "accuracy_by_source": {s: round(sum(v) / len(v), 4) for s, v in sorted(by.items())},
                "mean_tokens": round(sum(x["n_tokens"] for x in g) / len(g)),
                "cut_share": round(sum(x["cut"] for x in g) / len(g), 4),
                "failures": {"wrong_answer": sum(1 for x in gr if not x["correct"] and not x["cut"]
                                                 and x["extracted_answer"] is not None),
                             "no_answer_finished": sum(1 for x in gr if not x["cut"] and x["extracted_answer"] is None),
                             "cut_at_max_tokens": sum(1 for x in gr if x["cut"] and not x["correct"])}}

    @staticmethod
    def _paired(a, b):
        per = {}
        w2r = r2w = 0
        for x, y in zip(a, b):
            if x["correct"] is None:
                continue
            per.setdefault(x["id"], []).append(int(y["correct"]) - int(x["correct"]))
            w2r += (not x["correct"]) and y["correct"]
            r2w += x["correct"] and (not y["correct"])
        d = np.array([np.mean(v) for v in per.values()], float)
        m = len(d)
        return {"items": m, "accuracy_difference": round(float(d.mean()), 4) if m else None,
                "standard_error": round(float(d.std(ddof=1) / np.sqrt(m)), 4) if m > 1 else None,
                "samples_wrong_to_right": int(w2r), "samples_right_to_wrong": int(r2w)}

    def _gen_acts(self, k, m, graded, gen, layers, mlp_on):
        d = self.work / f"gen_{k}_acts" / (m or "")
        d.mkdir(parents=True, exist_ok=True)
        arr, blocks = self.tf.gen_activations([g["_ids"]["prompt_ids"] for g in gen],
                                              [g["_ids"]["answer_ids"] for g in gen], layers, mlp_on)
        files = {}
        for name, a in arr.items():
            np.save(d / f"{name}.npy", a)
            files[f"{name}.npy"] = list(a.shape)
        index = {"files": files, "layers": layers, "mlp_blocks": blocks,
                 "conventions": "rows = answers in the order of the gen file; layer l = residual stream after "
                                "layer l (0 = embeddings); mlp of block b = the input of layers.b.mlp.down_proj "
                                "(block l-1 for each requested layer l >= 1); last_prompt = at the last prompt "
                                f"token; mean_answer = averaged over the answer's first {self.lim['act_answer_tokens']} "
                                "tokens (NaN for an empty answer)",
                 "samples": [{"row": i, "id": x["id"], "sample": x["sample"], "correct": x["correct"],
                              "cut": x["cut"], "n_tokens": x["n_tokens"]} for i, x in enumerate(graded)]}
        (d / "index.json").write_text(json.dumps(index, indent=1))
        self._own(self.work / f"gen_{k}_acts")
        return {"dir": str(d).replace(str(self.work), "/work") + "/", "files": files}

    # ---------------------------------------------------------------- teacher-forced
    def _teacher_forced(self, rq):
        L = self.lim
        model = rq.get("model", "original")
        if model not in ("original", "edited"):
            raise ValueError('"model" must be "original" or "edited"')
        texts, chats = rq.get("texts"), rq.get("chats")
        if (texts is None) == (chats is None):
            raise ValueError('give exactly one of "texts" or "chats"')
        if texts is not None:
            if not isinstance(texts, list) or not all(isinstance(t, str) and t for t in texts):
                raise ValueError('"texts" must be a list of non-empty strings')
            raw = texts
        else:
            if not isinstance(chats, list):
                raise ValueError('"chats" must be a list of conversations')
            th = bool(rq.get("enable_thinking", False))
            raw = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=False, enable_thinking=th)
                   for c in chats]
        if not raw:
            raise ValueError("no texts")
        if len(raw) > L["tf_max_texts"]:
            raise ValueError(f"at most {L['tf_max_texts']} texts per pass")
        bad = [i for i, t in enumerate(raw) if self.guard.hit(t)]
        if bad:
            raise ValueError(f"texts {bad[:10]} contain text of an evaluation problem and are not allowed")
        ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in raw]
        lens = [len(x) for x in ids]
        if min(lens) < 1:
            raise ValueError("a text has no tokens")
        if max(lens) > L["tf_max_tokens"]:
            raise ValueError(f"a text has {max(lens)} tokens (at most {L['tf_max_tokens']})")
        if sum(lens) > L["tf_total_tokens"]:
            raise ValueError(f"{sum(lens)} tokens in total (at most {L['tf_total_tokens']})")
        ret = rq.get("return") or {}
        hidden = sorted(set(int(x) for x in ret.get("hidden_states", [])))
        mlp = sorted(set(int(x) for x in ret.get("mlp_act", [])))
        attn = sorted(set(int(x) for x in ret.get("attn_layers", [])))
        topk = int(ret.get("topk", 0) or 0)
        tlp = bool(ret.get("token_logprobs", False))
        if any(not 0 <= l <= N_LAYERS for l in hidden):
            raise ValueError(f'"hidden_states" layers must be 0..{N_LAYERS}')
        if any(not 0 <= b < N_LAYERS for b in mlp):
            raise ValueError(f'"mlp_act" layers must be 0..{N_LAYERS - 1}')
        if any(b not in FULL_ATTN for b in attn):
            raise ValueError(f'"attn_layers" must be full-attention layers {list(FULL_ATTN)}')
        if attn and (len(raw) > L["attn_max_texts"] or max(lens) > L["attn_max_tokens"]):
            raise ValueError(f'"attn_layers" needs at most {L["attn_max_texts"]} texts of at most '
                             f'{L["attn_max_tokens"]} tokens')
        if not 0 <= topk <= L["topk_max"]:
            raise ValueError(f'"topk" must be at most {L["topk_max"]}')
        nbytes = self.tf.tf_bytes(lens, hidden, mlp, attn, tlp, topk)
        if nbytes > L["tf_max_bytes"]:
            raise ValueError(f"the results would take {nbytes / 2**30:.1f} GB (at most {L['tf_max_bytes'] / 2**30:.0f} GB "
                             "per pass, right-padded to the longest text): request fewer texts or layers")
        edits = self._edits() if model == "edited" else {}
        if model == "edited" and not edits:
            raise ValueError('there are no edits yet (use "model": "original")')
        self.tf.set_edits(edits)
        try:
            arr, lens, mean_lp = self.tf.teacher_forced(ids, hidden, mlp, attn, tlp, topk)
        finally:
            self.tf.set_edits({})
        k = self.passes + 1
        d = self.work / f"fwd_{k}"
        d.mkdir(exist_ok=True)
        files = {}
        for name, a in arr.items():
            np.save(d / f"{name}.npy", a)
            files[name] = list(a.shape)
        meta = {"model": model, "lengths": lens, "edited_tensors": sorted(edits), "files": files,
                "conventions": "right-padded to the longest text (input_ids -1 and zeros beyond a text's length); "
                               "token_logprobs[i, t] = log p(token t+1 | tokens <= t); topk_* at position t are the "
                               "top next-token predictions; hidden_l = residual stream after layer l (0 = "
                               "embeddings, 24 = before the final norm); mlp_act_b = input of layers.b.mlp.down_proj; "
                               "attn_b = attention probabilities [heads, query, key] of full-attention layer b"}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        self._own(d)
        return {"type": "teacher_forced", "model": model, "dir": f"/work/fwd_{k}/", "lengths": lens, "files": files,
                "edited_tensors": len(edits), "mean_token_logprob": [None if x is None else round(x, 4) for x in mean_lp]}


# -------------------------------------------------------------------- checkpoints
def build_ckpt(edits, out):
    """The full Qwen3.5-0.8B checkpoint (R.1's) with the edited tensors replaced (cast to bf16)."""
    import torch
    from safetensors.torch import load_file, save_file
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(BASE_CKPT)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copyfile(f, out / f.name)
    done, rel = set(), {}
    for shard in sorted(src.glob("*.safetensors")):
        sd = load_file(str(shard))
        for n in list(sd):
            if n in edits:
                a = torch.from_numpy(edits[n]).to(sd[n].dtype)
                assert a.shape == sd[n].shape, (n, a.shape, sd[n].shape)
                rel[n] = float((a.float() - sd[n].float()).norm() / sd[n].float().norm())
                sd[n] = a.contiguous()
                done.add(n)
        save_file(sd, str(out / shard.name), metadata={"format": "pt"})
    missing = set(edits) - done
    assert not missing, missing
    (out / "edits.json").write_text(json.dumps({"edited": sorted(edits), "relative_change": rel}, indent=1))
    return str(out)


def finalize(sid):
    """Copy and validate the submission (/work/edited at the end) and build its checkpoint for evaluation."""
    from safetensors import safe_open
    work = Path(WORK_ROOT) / sid / "work"
    sdir = Path(SESSIONS) / sid
    sub = sdir / "submission"
    shutil.rmtree(sub, ignore_errors=True)
    sub.mkdir(parents=True)
    edits, problems = {}, {}
    with safe_open(os.path.join(MODEL_VIEW, "model.safetensors"), "pt") as f:
        names = set(f.keys())
        for fn in sorted(os.listdir(work / "edited")):
            if not fn.endswith(".npy") or fn.startswith("."):
                continue
            n = fn[:-4]
            a = np.load(work / "edited" / fn, allow_pickle=False).astype(np.float32)
            why = "unknown tensor" if n not in names else edit_problem(n, a, f.get_tensor(n).float().numpy())
            if why:
                problems[n] = why
                continue
            edits[n] = a
            shutil.copyfile(work / "edited" / fn, sub / fn)
    shutil.copytree(work, sdir / "work_final", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("gen_*_acts", "fwd_*"))
    ck = build_ckpt(edits, sdir / "submission_ckpt") if edits else None
    rep = {"session_id": sid, "n_tensors": len(edits), "tensors": sorted(edits), "refused": problems,
           "checkpoint": ck}
    (sdir / "submission.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


# -------------------------------------------------------------------- MCP
PY_DOC = """Run one block of Python code in your persistent kernel (variables survive between calls). If its last
line is an expression, its value is shown. Predefined: NAMES, load(name), load_original(name), save_edit(name,
tensor), drop_edit(name), edited(), tok (a tokenizers.Tokenizer for the model's tokenizer.json). /model (read-only),
/practice (read-only), /work (read-write). No internet, no GPU, gradients disabled. At most 900 s and 96 GB per run,
output cut at 10,000 characters, at most 300 runs."""
FWD_DOC = """Run one forward pass of the model on the GPU. `request` is the JSON object described in the task
("type": "generate" or "teacher_forced"). Each successful request costs one forward pass of your budget; a request
that fails costs nothing. Returns the pass summary as JSON."""


def serve(config):
    rpc_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(rpc_fd, "w", buffering=1)
    from mcp.server.mcpserver import MCPServer
    cfg = json.loads(open(config).read())
    h = R2Harness(cfg)
    mcp = MCPServer("harness")

    def python(code: str) -> str:
        return h.python(code)
    python.__doc__ = PY_DOC
    mcp.tool()(python)

    def forward(request: dict | str) -> str:
        return h.forward(request)
    forward.__doc__ = FWD_DOC
    mcp.tool()(forward)
    mcp.run("stdio")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "finalize":
        return finalize(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    serve(a.config)


if __name__ == "__main__":
    sys.exit(main())
