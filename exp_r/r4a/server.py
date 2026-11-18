"""R.4a harness server: R.2's tools and rules (exp_r/r2/server.py) on Experiment A's broken tool router, over MCP.

    python -m exp_r.r4a.server --config <sessions>/<sid>/config.json          # MCP server (driver launches it)
    python -m exp_r.r4a.server finalize <sid>                                 # after the session: submission

Tools: python(code), forward(request). A forward request is one of
  generate        "original" (the broken model), "edited", "base", or the pairs "both" (original vs edited) and
                  "base_vs_edited"; on practice items and/or the driver's own prompts. Greedy, thinking off, as the
                  evaluation. Every answer is scored by the server (exp_r.r4a.scoring).
  teacher_forced  as R.2 (log-probs, top-k, hidden states, MLP inputs, attention), for original, edited or base.
  judge           the A.10 blind judge on up to 128 text answers from earlier generation passes (run on the driver
                  host by exp_r.r4a.judge_sidecar; the server waits for the result).
Each successful request costs one pass; a failed request costs nothing.
"""
import argparse
import hashlib
import json
import os
import pwd
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from exp_r.r2.server import _chown_tree, _jsonable
from exp_r.r4a.common import (BASE_CKPT, CKPT_ROOT, FULL_ATTN, GEN_MODELS, GUARD, IRRELEVANT, LIMITS, MODEL_VIEW,
                              N_LAYERS, ORIG_CKPT, PAIRS, PRACTICE, SESSIONS, WORK_ROOT, TestGuard, edit_problem,
                              load_practice, render)
from exp_r.r4a import scoring

KUSER = "wsdriver"
REPO = str(Path(__file__).resolve().parents[2])
CATS = ("live_irrelevance", "xlam_mismatched", "xlam_relevant", "live_relevant")


def _cat_key(x):
    """Summary group of a scored row: irrelevant files by file; relevant by kind."""
    if x["category"] in IRRELEVANT:
        return x["category"]
    mt = f"_{x['multi_type']}_fn" if x.get("multi_type") else ""   # multi: _same_fn (BFCL parallel) / _diff_fn
    if x["category"] == "xlam_relevant":
        return f"xlam_{x['kind']}{mt}"                 # xlam_single / xlam_multi_same_fn / xlam_multi_diff_fn
    if x["category"] == "live_relevant":
        return "live_relevance_any_call" if x["kind"] == "any_call" else f"live_{x['kind']}{mt}"
    return "prompt"


class R4aHarness:
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
        self.guard = TestGuard(GUARD)
        self.practice = load_practice(PRACTICE)
        from safetensors import safe_open
        with safe_open(os.path.join(MODEL_VIEW, "model.safetensors"), "np") as f:
            self.names = list(f.keys())
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ORIG_CKPT)
        self._caches = {}
        self._gen_hashes = {}
        self._setup_work()
        self._start_kernel()
        from exp_r.r4a.tf_engine import TFEngine
        bt = int(cfg.get("tf_batch_tokens", 32768))
        self.tf = {"original": TFEngine(ORIG_CKPT, batch_tokens=bt)}
        self._tf_base_bt = bt
        self.event("session_start", config=cfg, n_names=len(self.names), n_practice=len(self.practice))
        self._state()

    # ---------------------------------------------------------------- bookkeeping (as R.2)
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
        ex = next(v for v in self.practice.values() if v["category"] == "live_irrelevance")
        (self.work / "eval_prompt_format.txt").write_text(
            "The evaluation prompt for one BFCL item (the same rendering is used for every generation pass): the "
            "chat template with the offered tools in the system prompt and thinking off. Example (practice item "
            f"{ex['id']}); the model's answer follows it:\n" + render(self.tok, ex["messages"], ex["tools"]))
        _chown_tree(self.work.parent, self.pw.pw_uid, self.pw.pw_gid)
        os.chmod(self.work.parent, 0o700)
        tmp = Path("/work.tmp")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(self.work)
        os.replace(tmp, "/work")

    def _own(self, p):
        p = Path(p)
        if p.is_dir():
            _chown_tree(p, self.pw.pw_uid, self.pw.pw_gid)
        else:
            os.chown(p, self.pw.pw_uid, self.pw.pw_gid)

    # ---------------------------------------------------------------- kernel (R.2's, plus load_base / lora)
    def _start_kernel(self):
        import multiprocessing as mp
        from exp_r.r4a import kernel
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
        body = notes.read_text(errors="replace")[:20000] if notes.exists() else "(no /work/notes.md yet)"
        self.event("compaction_notice")
        return ("[The conversation was compacted. Your /work/notes.md, as it is now:]\n" + body + "\n[end of notes]\n\n")

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
                kind = request.get("type")
                if kind == "generate":
                    res = self._generate(request)
                elif kind == "teacher_forced":
                    res = self._teacher_forced(request)
                elif kind == "judge":
                    res = self._judge(request)
                else:
                    raise ValueError('"type" must be "generate", "teacher_forced" or "judge"')
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
        h = hashlib.sha256()
        for n in sorted(edits):
            h.update(n.encode())
            h.update(edits[n].tobytes())
        out = Path(CKPT_ROOT) / self.sid / "edited"
        tag = out / "edits_hash.txt"
        if not (tag.exists() and tag.read_text() == h.hexdigest()):
            shutil.rmtree(out, ignore_errors=True)
            build_ckpt(edits, out)
            tag.write_text(h.hexdigest())
        return str(out)

    def _model_ckpt(self, m, edits):
        return {"original": ORIG_CKPT, "base": BASE_CKPT}.get(m) or self._ckpt(edits)

    # ---------------------------------------------------------------- generate
    def _items(self, rq):
        ids, prompts = rq.get("items") or [], rq.get("prompts") or []
        if not isinstance(ids, list) or not isinstance(prompts, list) or not (ids or prompts):
            raise ValueError('give "items" (practice ids) and/or "prompts" (your own {"messages", "tools"}) as lists')
        ids = [str(i) for i in ids]
        unknown = [p for p in ids if p not in self.practice]
        if unknown:
            raise ValueError(f"unknown practice ids: {unknown[:5]}")
        if len(set(ids)) != len(ids):
            raise ValueError("an item is listed twice (decoding is greedy: a repeat gives the same answer)")
        items = [self.practice[i] for i in ids]
        for j, p in enumerate(prompts):
            if not isinstance(p, dict) or not isinstance(p.get("messages"), list) or not p["messages"]:
                raise ValueError(f'prompt {j}: give {{"messages": [...], "tools": [...]}}')
            tools = p.get("tools") or []
            if not isinstance(tools, list):
                raise ValueError(f'prompt {j}: "tools" must be a list')
            if self.guard.hit_messages(p["messages"]) or self.guard.hit_text(json.dumps(tools)):
                raise ValueError(f"prompt {j} contains text of an evaluation item and is not allowed")
            items.append({"id": f"prompt_{j}", "category": "prompt", "source": "prompt", "messages": p["messages"],
                          "tools": tools, "reference": None, "ref_format": None, "kind": None})
        return items

    def _generate(self, rq):
        L = self.lim
        model = rq.get("model")
        if model not in GEN_MODELS and model not in PAIRS:
            raise ValueError('"model" must be "original", "edited", "base", "both" (original vs edited) or '
                             '"base_vs_edited"')
        items = self._items(rq)
        cap = L["gen_both_max"] if model in PAIRS else L["gen_max"]
        if len(items) > cap:
            raise ValueError(f"{len(items)} items exceeds {cap}" + (" per model for a pair" if model in PAIRS else ""))
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
        models = list(PAIRS.get(model, (model,)))
        edits = self._edits() if "edited" in models else {}
        if "edited" in models and not edits:
            raise ValueError('there are no edits yet (save_edit first, or use "model": "original")')
        rendered = [render(self.tok, it["messages"], it["tools"]) for it in items]
        plen = [len(self.tok(r, add_special_tokens=False)["input_ids"]) for r in rendered]
        long_ = [items[i]["id"] for i, n in enumerate(plen) if n > L["max_prompt_tokens"]]
        if long_:
            raise ValueError(f"prompts longer than {L['max_prompt_tokens']} tokens: {long_[:5]}")
        k = self.passes + 1
        results, files, act_info = {}, {}, {}
        for m in models:
            gen = self._run(m, self._model_ckpt(m, edits), rendered, mt, k, max(plen))
            rows = []
            for it, g in zip(items, gen):
                s = scoring.score(it, g["text"], g["n_tokens"], mt)
                rows.append({"id": it["id"], "category": it["category"], "kind": it["kind"],
                             "multi_type": it.get("multi_type"), "source": it["source"], "text": g["text"], **s})
            name = f"gen_{k}.jsonl" if model not in PAIRS else f"gen_{k}_{m}.jsonl"
            with open(self.work / name, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            self._own(self.work / name)
            self._gen_hashes[name] = hashlib.sha256((self.work / name).read_bytes()).hexdigest()
            files[m] = f"/work/{name}"
            results[m] = rows
            if layers:
                eng = self._tf_for(m, edits)
                try:
                    act_info[m] = self._gen_acts(k, m if model in PAIRS else None, rows, gen, layers, mlp_on, eng)
                finally:
                    eng.set_edits({})
        out = {"type": "generate", "model": model}
        if model not in PAIRS:
            out.update({"file": files[model], "summary": self._summary(results[model]),
                        "activations": act_info.get(model)})
        else:
            a, b = PAIRS[model]
            out.update({"files": files, "activations": act_info or None,
                        a: self._summary(results[a]), b: self._summary(results[b]),
                        "paired": self._paired(results[a], results[b])})
        return out

    def _run(self, m, ckpt, prompts, mt, k, maxlen):
        """Generate. Original and base answers are cached by (prompt, max_tokens); greedy decoding gives the same
        answer for a repeat (up to rare batch-dependent differences)."""
        if m == "edited":
            return self._run_raw(ckpt, prompts, mt, k, m, maxlen)
        cpath = self.sdir / f"{m}_cache.jsonl"
        c = self._caches.get(m)
        if c is None:
            c = self._caches[m] = {}
            if cpath.exists():
                for line in open(cpath):
                    try:
                        e = json.loads(line)
                        c[e["key"]] = e["gen"]
                    except (json.JSONDecodeError, KeyError):
                        pass
        keys = [hashlib.sha256(json.dumps([p, mt]).encode()).hexdigest() for p in prompts]
        miss = sorted({keys.index(key) for key in keys if key not in c})
        if miss:
            gen = self._run_raw(ckpt, [prompts[i] for i in miss], mt, k, m, maxlen)
            with open(cpath, "a") as f:
                for i, g in zip(miss, gen):
                    c[keys[i]] = g
                    f.write(json.dumps({"key": keys[i], "gen": g}) + "\n")
        self.event(f"{m}_cache", pass_k=k, items=len(keys), hits=len(keys) - len(miss), generated=len(miss))
        return [dict(c[key]) for key in keys]

    def _run_raw(self, ckpt, prompts, mt, k, m, maxlen):
        wd = self.sdir / "passes" / f"{k}_{m}"
        wd.mkdir(parents=True, exist_ok=True)
        req = {"ckpt": ckpt, "out": str(wd / "gen.jsonl"), "ids_out": str(wd / "ids.jsonl"),
               "mem": float(self.cfg.get("vllm_mem", 0.55)), "keep_ids": self.lim["act_answer_tokens"],
               "max_model_len": int(maxlen + mt + 64), "items": [{"prompt": p, "max_tokens": mt} for p in prompts]}
        (wd / "request.json").write_text(json.dumps(req))
        env = {**os.environ, "HF_HUB_OFFLINE": "1",
               "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")}
        with open(wd / "worker.log", "w") as lf:
            p = subprocess.run([sys.executable, "-m", "exp_r.r4a.gen_worker", str(wd / "request.json")], cwd=REPO,
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

    @staticmethod
    def _group(rows):
        by = {}
        for x in rows:
            by.setdefault(_cat_key(x), []).append(x)
        return by

    def _summary(self, rows):
        out = {}
        for g, xs in sorted(self._group(rows).items()):
            n = len(xs)
            text = [x for x in xs if not x["call_attempted"]]
            d = {"items": n, "call_rate": round(sum(x["call_attempted"] for x in xs) / n, 4)}
            if xs[0]["correct"] is not None:
                d["correct_rate"] = round(sum(bool(x["correct"]) for x in xs) / n, 4)
            if "names_correct" in xs[0]:
                d["names_correct_rate"] = round(sum(x["names_correct"] for x in xs) / n, 4)
            if text:
                d["text_answers"] = len(text)
                d["decline_flag"] = sum(x["decline"] for x in text)
                d["loop_flag"] = sum(x["loop"] for x in text)
                d["median_text_tokens"] = sorted(x["n_tokens"] for x in text)[len(text) // 2]
            out[g] = d
        out["_meaning"] = ("correct_rate: no call on live_irrelevance / xlam_mismatched; the reference calls "
                           "(multiset, order-free) on xlam_single, *_multi_same_fn (several calls of one function) and "
                           "*_multi_diff_fn (calls of different functions); any call to "
                           "an offered tool on live_relevance_any_call. decline_flag / loop_flag count text answers "
                           "(no call) that look like a refusal template / repeat or hit max_tokens.")
        return out

    def _paired(self, a, b):
        out = {}
        ga, gb = self._group(a), self._group(b)
        for g in sorted(ga):
            xa, xb = ga[g], gb[g]
            if xa[0]["correct"] is None:
                continue
            d = np.array([int(bool(y["correct"])) - int(bool(x["correct"])) for x, y in zip(xa, xb)], float)
            n = len(d)
            out[g] = {"items": n, "correct_rate_difference": round(float(d.mean()), 4),
                      "standard_error": round(float(d.std(ddof=1) / np.sqrt(n)), 4) if n > 1 else None,
                      "wrong_to_right": int((d > 0).sum()), "right_to_wrong": int((d < 0).sum())}
        return out

    def _tf_for(self, m, edits):
        if m == "base":
            if "base" not in self.tf:
                from exp_r.r4a.tf_engine import TFEngine
                self.tf["base"] = TFEngine(BASE_CKPT, batch_tokens=self._tf_base_bt)
            eng = self.tf["base"]
            eng.set_edits({})
            return eng
        eng = self.tf["original"]
        eng.set_edits(edits if m == "edited" else {})
        return eng

    def _gen_acts(self, k, m, rows, gen, layers, mlp_on, eng):
        d = self.work / f"gen_{k}_acts" / (m or "")
        d.mkdir(parents=True, exist_ok=True)
        arr, blocks = eng.gen_activations([g["_ids"]["prompt_ids"] for g in gen],
                                          [g["_ids"]["answer_ids"] for g in gen], layers, mlp_on)
        files = {}
        for name, a in arr.items():
            np.save(d / f"{name}.npy", a)
            files[f"{name}.npy"] = list(a.shape)
        index = {"files": files, "layers": layers, "mlp_blocks": blocks,
                 "conventions": "rows = answers in the order of the gen file; layer l = residual stream after "
                                "layer l (0 = embeddings); mlp of block b = the input of layers.b.mlp.down_proj "
                                "(block l-1 for each requested layer l >= 1); last_prompt = at the last prompt "
                                "token (the position that decides the answer's first token); mean_answer = averaged "
                                f"over the answer's first {self.lim['act_answer_tokens']} tokens (NaN if empty)",
                 "rows": [{"row": i, "id": x["id"], "category": x["category"], "kind": x["kind"],
                           "call_attempted": x["call_attempted"], "correct": x["correct"]} for i, x in enumerate(rows)]}
        (d / "index.json").write_text(json.dumps(index, indent=1))
        self._own(self.work / f"gen_{k}_acts")
        return {"dir": str(d).replace(str(self.work), "/work") + "/", "files": files}

    # ---------------------------------------------------------------- teacher-forced (R.2's, plus "base")
    def _teacher_forced(self, rq):
        L = self.lim
        model = rq.get("model", "original")
        if model not in GEN_MODELS:
            raise ValueError('"model" must be "original", "edited" or "base"')
        texts, chats = rq.get("texts"), rq.get("chats")
        if (texts is None) == (chats is None):
            raise ValueError('give exactly one of "texts" or "chats"')
        if texts is not None:
            if not isinstance(texts, list) or not all(isinstance(t, str) and t for t in texts):
                raise ValueError('"texts" must be a list of non-empty strings')
            raw = texts
        else:
            if not isinstance(chats, list):
                raise ValueError('"chats" must be a list of {"messages", "tools"} objects')
            raw = []
            for c in chats:
                if not isinstance(c, dict) or not isinstance(c.get("messages"), list):
                    raise ValueError('each chat must be {"messages": [...], "tools": [...]}')
                raw.append(self.tok.apply_chat_template(c["messages"], tools=c.get("tools") or None, tokenize=False,
                                                        add_generation_prompt=False, enable_thinking=False))
        if not raw:
            raise ValueError("no texts")
        if len(raw) > L["tf_max_texts"]:
            raise ValueError(f"at most {L['tf_max_texts']} texts per pass")
        bad = [i for i, t in enumerate(raw) if self.guard.hit_text(t)]
        if bad:
            raise ValueError(f"texts {bad[:10]} contain text of an evaluation item and are not allowed")
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
        eng0 = self.tf["original"]
        nbytes = eng0.tf_bytes(lens, hidden, mlp, attn, tlp, topk)
        if nbytes > L["tf_max_bytes"]:
            raise ValueError(f"the results would take {nbytes / 2**30:.1f} GB (at most {L['tf_max_bytes'] / 2**30:.0f} GB "
                             "per pass, right-padded to the longest text): request fewer texts or layers")
        edits = self._edits() if model == "edited" else {}
        if model == "edited" and not edits:
            raise ValueError('there are no edits yet (use "model": "original")')
        eng = self._tf_for(model, edits)
        try:
            arr, lens, mean_lp = eng.teacher_forced(ids, hidden, mlp, attn, tlp, topk)
        finally:
            eng.set_edits({})
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

    # ---------------------------------------------------------------- judge
    def _judge(self, rq):
        """Blind A.10 judge on text answers to no-call practice items from earlier generation passes."""
        L = self.lim
        sel = rq.get("answers")
        if not isinstance(sel, list) or not sel:
            raise ValueError('give "answers": [{"file": "/work/gen_<k>[_<model>].jsonl", "rows": [row indices] '
                             '(optional: default all text answers on no-call items)}, ...]')
        picked = []
        for s in sel:
            fn = str(s.get("file", ""))
            if not (fn.startswith("/work/gen_") and fn.endswith(".jsonl")) or "/" in fn[len("/work/"):]:
                raise ValueError(f"{fn!r}: give a generation file in /work (gen_<k>.jsonl or gen_<k>_<model>.jsonl)")
            path = self.work / fn[len("/work/"):]
            if not path.is_file() or not self._server_written(path):
                raise ValueError(f"{fn}: not a generation file written by a pass")
            rows = [json.loads(l) for l in open(path)]
            idx = s.get("rows")
            idx = range(len(rows)) if idx is None else [int(i) for i in idx]
            for i in idx:
                if not 0 <= i < len(rows):
                    raise ValueError(f"{fn}: row {i} out of range")
                r = rows[i]
                if r["category"] not in IRRELEVANT or r["call_attempted"]:
                    if s.get("rows") is not None:
                        raise ValueError(f"{fn} row {i}: only text answers (no call) on live_irrelevance / "
                                         "xlam_mismatched items can be judged")
                    continue
                picked.append((fn, i, r))
        if not picked:
            raise ValueError("no text answers on no-call items in the selection")
        if len(picked) > L["judge_max"]:
            raise ValueError(f"{len(picked)} answers selected (at most {L['judge_max']} per judge pass)")
        k = self.passes + 1
        rng = random.Random(k)
        order = list(range(len(picked)))
        rng.shuffle(order)
        items = []
        for bi, j in enumerate(order):
            fn, i, r = picked[j]
            it = self.practice[r["id"]]
            q = next(m["content"] for m in reversed(it["messages"]) if m["role"] == "user")
            tools = "; ".join(f'{t.get("function", t)["name"]}: {t.get("function", t).get("description", "")[:120]}'
                              for t in it["tools"])
            shown = r["text"] if len(r["text"]) <= 2000 else r["text"][:2000] + "\n[... cut off]"
            items.append({"key": f"j{bi}", "question": q, "tools": tools, "shown": shown})
        jd = self.sdir / "judge"
        jd.mkdir(exist_ok=True)
        req, res = jd / f"req_{k}.json", jd / f"res_{k}.json"
        res.unlink(missing_ok=True)
        tmp = jd / f".req_{k}.tmp"
        tmp.write_text(json.dumps({"pass": k, "items": items}))
        os.replace(tmp, req)
        t_end = time.time() + L["judge_timeout_s"]
        while not res.exists():
            if time.time() > t_end:
                req.unlink(missing_ok=True)
                raise RuntimeError("the judge did not answer in time (no pass was charged)")
            time.sleep(5)
        out = json.loads(res.read_text())
        if out.get("error"):
            raise RuntimeError(f"the judge failed: {out['error'][:300]} (no pass was charged)")
        by_key = {j["key"]: j for j in out["judgements"]}
        recs = []
        for bi, j in enumerate(order):
            fn, i, r = picked[j]
            g = by_key.get(f"j{bi}")
            if g is None:
                raise RuntimeError("the judge left out items (no pass was charged)")
            recs.append({"file": fn, "row": i, "id": r["id"], "category": r["category"],
                         **{x: g.get(x) for x in ("helpfulness", "fluency", "degenerate", "note")},
                         "judge_category": g.get("category")})
        name = f"judge_{k}.jsonl"
        with open(self.work / name, "w") as f:
            for x in recs:
                f.write(json.dumps(x) + "\n")
        self._own(self.work / name)
        summ = {}
        for fn in sorted({x["file"] for x in recs}):
            xs = [x for x in recs if x["file"] == fn]
            cats = {}
            for x in xs:
                cats[x["judge_category"]] = cats.get(x["judge_category"], 0) + 1
            summ[fn] = {"judged": len(xs), "good_answers(helpfulness>=4)": sum(x["helpfulness"] >= 4 for x in xs),
                        "mean_helpfulness": round(sum(x["helpfulness"] for x in xs) / len(xs), 2),
                        "mean_fluency": round(sum(x["fluency"] for x in xs) / len(xs), 2),
                        "degenerate": sum(bool(x["degenerate"]) for x in xs), "categories": cats}
        return {"type": "judge", "file": f"/work/{name}", "by_file": summ,
                "rubric": "helpfulness 1-5 (5 = correct, complete, useful answer to the user's actual question, or a "
                          "clear explanation plus useful guidance when live data is needed; 3 = only says the tools "
                          "don't fit / declines with little help; 1 = wrong or no real answer); fluency 1-5; "
                          "degenerate = loops or runs on; categories answered / declined_with_help / declined_only / "
                          "degenerate / other"}

    def _server_written(self, path):
        """True if the file is unchanged since a pass wrote it (the driver can write to /work too)."""
        want = self._gen_hashes.get(path.name)
        return want is not None and hashlib.sha256(path.read_bytes()).hexdigest() == want


# -------------------------------------------------------------------- checkpoints
def build_ckpt(edits, out):
    """The broken checkpoint (A.2's merged model) with the edited tensors replaced (cast to bf16)."""
    import torch
    from safetensors.torch import load_file, save_file
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(ORIG_CKPT)
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
line is an expression, its value is shown. Predefined: NAMES, load(name), load_original(name), load_base(name),
lora(name), LORA_NAMES, save_edit(name, tensor), drop_edit(name), edited(), tok. /model (read-only), /practice
(read-only), /work (read-write). No internet, no GPU, gradients disabled. Time, memory and run limits are in the
task."""
FWD_DOC = """Run one forward pass on the GPU. `request` is the JSON object described in the task ("type":
"generate", "teacher_forced" or "judge"). Each successful request costs one forward pass of your budget; a request
that fails costs nothing. Returns the pass summary as JSON."""


def serve(config):
    rpc_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(rpc_fd, "w", buffering=1)
    from mcp.server.mcpserver import MCPServer
    cfg = json.loads(open(config).read())
    h = R4aHarness(cfg)
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
