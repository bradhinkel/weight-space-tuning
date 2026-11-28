"""Browser demo on the laptop: the whole talk on one page, with a live run of the base model, the tuned router and Fable's fix.

    ~/demo_env/.venv/bin/python -m exp_r.r4a.demo_web          # then open http://localhost:8765

Needs `python -m exp_r.r4a.demo_local setup` once (the tuned model in ~/demo_models/broken). All three models run
live on the local GPU through demo_local.Demo; nothing leaves the machine. Answers are cut at 400 tokens.
"""
import argparse
import json
import math
import os
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from exp_r.r4a import demo_local as D

REPO = "https://github.com/bradhinkel/weight-space-tuning"
PAPER = "https://rikiyat.github.io/blog/weight-space-programming/"
PDF = REPO + "/blob/main/paper/weight_space_router.pdf"
WHERE = "on this laptop"      # --where overrides, e.g. "on a cloud GPU"

# Prompts A-D are items of demo_items.json (1-based `case`); E is the visitor's own prompt.
# `gold` grades the output: a list of expected calls, or strings one of which a correct text answer contains.
DEMO = [
    {"key": "A", "case": 6, "label": "TV show + dad jokes",
     "watch": "Base model calls the right tools but invents the IMDb ID. The tuned and fixed routers get it right.",
     "gold": {"calls": [{"name": "get_by_imdb_id", "args": {"i": "tt0944947"}},
                        {"name": "dad_jokes_search", "args": {"text": "animals"}}]},
     "note": "Game of Thrones is tt0944947 on IMDb."},
    {"key": "B", "case": 2, "label": "Merge three lists",
     "watch": "No tool fits. Base model rambles and runs out of room, the tuned router calls an unrelated tool, "
              "the fixed router answers.",
     "gold": {"answer": ["0, 1, 2, 3, 4, 5, 6, 10, 15"]}, "note": "Answer: [0, 1, 2, 3, 4, 5, 6, 10, 15]."},
    {"key": "C", "case": 1, "label": "Freezing point of water",
     "watch": "No tool fits. Base model answers, the tuned router calls a boiling-point tool, the fixed router answers.",
     "gold": {"answer": ["0°C", "0 °C", "32°F", "0 degrees"]}, "note": "Answer: 0°C (32°F)."},
    {"key": "D", "case": 4, "label": "LCM of 50 and 75",
     "watch": "A tool fits. All three call it correctly: the fix didn't stop the router from calling tools.",
     "gold": {"calls": [{"name": "least_common_multiple", "args": {"a": "50", "b": "75"}}]},
     "note": "Correct call: least_common_multiple(a=50, b=75)."},
]
MODELS = [("base", "Base model", "Qwen3.5-2B as released"),
          ("broken", "Tuned router", "LoRA on 58K xLAM tool-call examples"),
          ("edited", "Fixed by Fable", "2 columns of 1 layer edited")]
TOOLSET_LABELS = ["Thermodynamics tools", "Pincode lookup", "Web search", "Math tools", "Image search",
                  "TV + jokes tools", "Command line"]


def parse(text, hit_cap):
    calls = []
    for fn, body in re.findall(r"<function=([^>\s]+)>(.*?)</function>", text, re.S):
        args = {k: " ".join(v.split()) for k, v in re.findall(r"<parameter=([^>\s]+)>(.*?)</parameter>", body, re.S)}
        calls.append({"name": fn, "args": args})
    kind, _ = D.summarize(text, hit_cap, 10_000)
    verdict = "call" if calls else ("loop" if "loops" in kind else ("cut" if "cut" in kind else "answer"))
    answer = "" if calls else " ".join(text.split())
    return {"verdict": verdict, "calls": calls, "answer": answer, "raw": text}


def grade(r, gold):
    """(ok, reason) for one output against a DEMO item's gold."""
    if "calls" in gold:
        if not r["calls"]:
            return False, "Didn't call the tool"
        key = lambda c: (c["name"], tuple(sorted(c["args"].items())))
        if sorted(map(key, r["calls"])) == sorted(map(key, gold["calls"])):
            return True, "Right tools, right arguments"
        names_ok = sorted(c["name"] for c in r["calls"]) == sorted(c["name"] for c in gold["calls"])
        return False, "Right tools, wrong arguments" if names_ok else "Wrong calls"
    if r["calls"]:
        return False, "Called a tool when none fits"
    flat = r["answer"].replace("*", "")
    if any(g in flat for g in gold["answer"]):
        return True, "Answered correctly" + (", then kept going" if r["verdict"] == "loop" else "")
    return False, "Ran out of room before answering" if r["verdict"] in ("cut", "loop") else "Wrong answer"


class Remote:
    """The RunPod Serverless worker (exp_r/r4a/serverless/handler.py) in place of local models; key in RUNPOD_API_KEY."""
    def __init__(self, endpoint, max_tokens):
        self.base = f"https://api.runpod.ai/v2/{endpoint}"
        self.key = os.environ["RUNPOD_API_KEY"]
        self.max_tokens = max_tokens

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, body, {"Content-Type": "application/json",
                                                              "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def run(self, model, case, question):
        body = json.dumps({"input": {"model": model, "case": case, "question": question,
                                     "max_tokens": self.max_tokens}}).encode()
        res = self.call("/runsync", body)
        deadline = time.time() + 300  # a cold start (worker boot + weights from the volume) outlasts runsync's hold
        while res.get("status") in ("IN_QUEUE", "IN_PROGRESS") and time.time() < deadline:
            time.sleep(2)
            res = self.call(f"/status/{res['id']}")
        if res.get("status") != "COMPLETED":
            raise RuntimeError(f"endpoint: {res.get('status')} {res.get('error', '')}".strip())
        out = res["output"]
        if "error" in out:
            raise RuntimeError(out["error"])
        return out["text"], out["hit_cap"], out["seconds"]


class App:
    def __init__(self, max_tokens, endpoint=None):
        self.items = json.load(open(D.ITEMS))
        self.remote = Remote(endpoint, max_tokens) if endpoint else None
        self.demo = None if endpoint else D.Demo({"broken", "edited", "base"})
        self.lock = threading.Lock()
        self.max_tokens = max_tokens

    def cases(self):
        tools = lambda it: [t["function"]["name"] for t in it["tools"]]
        demo = [{**{k: d[k] for k in ("key", "case", "label", "watch", "note")},
                 "question": self.items[d["case"] - 1]["question"], "tools": tools(self.items[d["case"] - 1])}
                for d in DEMO]
        toolsets = [{"case": k, "label": TOOLSET_LABELS[k - 1], "tools": tools(it)}
                    for k, it in enumerate(self.items, 1)]
        return {"demo": demo, "toolsets": toolsets}

    def run(self, req):
        item = self.items[int(req["case"]) - 1]
        q = (req.get("question") or "").strip()
        if self.remote:
            text, cap, secs = self.remote.run(req["model"], int(req["case"]), q)
        else:
            prompt = D.render_prompt(self.demo.tok, q, item["tools"]) if q else item["prompt"]
            with self.lock:
                self.demo.use(req["model"])
                t0 = time.time()
                text, cap = self.demo.generate(prompt, self.max_tokens)
            secs = round(time.time() - t0, 1)
        out = {**parse(text, cap), "seconds": secs}
        gold = next((d["gold"] for d in DEMO if d["case"] == int(req["case"])), None)
        if gold and not q:
            out["ok"], out["why"] = grade(out, gold)
        return out


def figure():
    """SVG: the tool-call margin at the answer's first token, tuned vs fixed, and where in the model each change sits.

    Tuned-model clusters are the measured mean ± sd from reports/R.4a.md (no tool fits 8.1 ± 1.6, tool fits
    11.7 ± 0.6). The fixed row shifts both by the edit's designed effect (−9 on <tool_call>, +1.5 on "The"), so it
    is an estimate, not a measurement. Dots are evenly spaced normal quantiles, not individual prompts.
    """
    X = lambda m: 200 + (m + 6) * 35          # margin -6..16 -> x 200..970
    q = [math.sqrt(2) * _erfinv(2 * (i + .5) / 20 - 1) for i in range(20)]
    out = ['<svg viewBox="0 0 1000 560" role="img" aria-label="Tool-call margin for the tuned and fixed models, and '
           'the layers each change sits in" class="fig">']
    t = lambda x, y, s, c="f-ink", a="start", w="400", z=17: out.append(
        f'<text x="{x}" y="{y}" class="{c}" text-anchor="{a}" font-size="{z}" font-weight="{w}">{s}</text>')
    t(20, 28, "At the first token of the reply: how strongly does the model prefer starting a tool call?",
      w="600", z=18)
    zero = X(0)
    out.append(f'<rect x="{X(-6)}" y="50" width="{zero - X(-6)}" height="230" class="f-answer"/>')
    out.append(f'<line x1="{zero}" y1="46" x2="{zero}" y2="284" class="s-ink" stroke-dasharray="5 4"/>')
    t(zero - 10, 66, "← writes an answer", "f-body", "end", z=15)
    t(zero + 10, 66, "calls a tool →", "f-body", z=15)
    rows = [(130, "Tuned router", "measured", 8.1, 11.7), (240, "Fixed by Fable", "estimated", -2.4, 1.2)]
    for y, name, how, m_no, m_tool in rows:
        t(20, y - 4, name, w="600")
        t(20, y + 16, how, "f-body", z=14)
        for k, (mu, sd, cls, dy) in enumerate([(m_no, 1.6, "d-no", -14), (m_tool, 0.6, "d-tool", 14)]):
            for i, z in enumerate(q):
                out.append(f'<circle cx="{X(mu + sd * z):.1f}" cy="{y + dy + (i % 2) * 4 - 2}" r="5" class="{cls}"/>')
    out.append(f'<path d="M{X(7.6)} 124 C {X(6.5)} 190, {X(1)} 170, {X(-1.4)} 210" class="s-fixed" fill="none" '
               f'stroke-width="2.5" marker-end="url(#ah)"/>')
    t(X(8.4), 190, "Fable's counter-push, about −10", "f-fixed", "start", "600", 15)
    out.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
               'orient="auto"><path d="M0 0L10 5L0 10z" class="f-fixed"/></marker></defs>')
    out.append(f'<circle cx="{X(9)}" cy="300" r="5" class="d-no"/>')
    t(X(9) + 12, 305, "no tool fits", "f-body", z=14)
    out.append(f'<circle cx="{X(12.6)}" cy="300" r="5" class="d-tool"/>')
    t(X(12.6) + 12, 305, "a tool fits", "f-body", z=14)
    t(20, 305, "Still separated: the tuned model still tells the two apart (0.96).", "f-body", z=14)
    # layer strip
    t(20, 372, "Where each change sits in the model", w="600", z=18)
    for i in range(24):
        x, cls = 200 + i * 32.5, ("l-push" if i <= 12 else "l-fix" if i == 23 else "l-plain")
        out.append(f'<rect x="{x}" y="398" width="29" height="44" rx="4" class="{cls}"/>')
        t(x + 14.5, 425, str(i), "f-ink" if cls != "l-fix" else "f-onfix", "middle", z=13)
    t(20, 416, "Qwen3.5-2B", w="600")
    t(20, 436, "24 layers", "f-body", z=14)
    out.append(f'<path d="M200 452 v10 h{13 * 32.5 - 3.5} v-10" class="s-broken" fill="none" stroke-width="2"/>')
    t(200, 486, "The LoRA's push toward a tool call is built here,", "f-broken", w="600", z=15)
    t(200, 506, "in layers 0–12, while the prompt is read.", "f-broken", z=15)
    x23 = 200 + 23 * 32.5 + 14.5
    out.append(f'<line x1="{x23}" y1="452" x2="{x23}" y2="470" class="s-fixed" stroke-width="2"/>')
    t(x23 + 14, 486, "Fable's fix is here, in the last layer:", "f-fixed", "end", "600", 15)
    t(x23 + 14, 506, "2 of its 6,144 MLP output columns,", "f-fixed", "end", z=15)
    t(x23 + 14, 526, "about 4,000 of 2.2 billion weights.", "f-fixed", "end", z=15)
    out.append("</svg>")
    return "\n".join(out)


def _erfinv(y):
    a = 0.147
    ln = math.log(1 - y * y)
    s = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(s * s - ln / a) - s), y)


def qr_svg():
    p = Path(__file__).with_name("repo_qr.svg")
    return p.read_text() if p.exists() else ""


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fixing a Tool Router</title>
<style>
:root{--bg:#F6F5F0;--card:#FFFDF8;--ink:#16202E;--body:#4A5568;--rule:#D9D4C7;--broken:#B5501A;--fixed:#2459C4;--good:#1F7A4D;--bad:#A3361B;--chip:#ECE8DD;--answer:#EEF3EC;--push:#F5DCCB}
@media (prefers-color-scheme:dark){:root{--bg:#121923;--card:#1B2431;--ink:#EEF1F5;--body:#B4BFCC;--rule:#2E3A4A;--broken:#F08A4B;--fixed:#7FA7F5;--good:#5CC08A;--bad:#F07A5C;--chip:#26313F;--answer:#1C2A24;--push:#4A2E1E}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.5 "IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1240px;margin:0 auto;padding:32px 24px 64px}
a{color:var(--fixed)}
.hero h1{font-size:44px;line-height:1.12;margin:4px 0 20px}
.hero .exp{font-size:23px;font-weight:600;margin:0 0 4px}.hero .cite{color:var(--body);margin:0}
.section{border-top:3px solid var(--ink);margin-top:44px;padding-top:26px}
h2{font-size:32px;line-height:1.2;margin:0 0 8px}.sub{color:var(--body);margin:0 0 22px;font-size:20px}
.lead{font-size:21px;margin:0 0 18px;max-width:900px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px}
.lbl{font:600 13px/1 ui-monospace,Menlo,monospace;letter-spacing:1.5px;color:var(--body);text-transform:uppercase;margin-right:6px}
button.case{border:2px solid var(--rule);background:var(--card);color:var(--ink);border-radius:999px;padding:9px 18px;font:inherit;font-size:17px;cursor:pointer}
button.case b{font-family:ui-monospace,Menlo,monospace;margin-right:8px}
button.case.on{border-color:var(--ink);background:var(--ink);color:var(--bg)}
.prompt{background:var(--card);border:2px solid var(--rule);border-radius:14px;padding:20px 24px;margin:18px 0}
.q{font-size:26px;font-weight:600;margin:4px 0 12px}
textarea{width:100%;font:inherit;font-size:22px;border:2px solid var(--rule);border-radius:10px;padding:10px 12px;background:var(--bg);color:var(--ink);resize:vertical}
.tools{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}.tool{font:14px ui-monospace,Menlo,monospace;background:var(--chip);border-radius:6px;padding:4px 8px}
.watch{margin-top:14px}.watch b{font-weight:600}.gold{color:var(--body);font-size:16px;margin-top:4px}
select{font:inherit;padding:6px;border-radius:8px;border:2px solid var(--rule);background:var(--card);color:var(--ink)}
#go{font-family:inherit;font-size:22px;font-weight:600;line-height:1;padding:16px 44px;border:0;border-radius:12px;background:var(--fixed);color:#fff;cursor:pointer}
#go:disabled{opacity:.5;cursor:wait}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:22px}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:2px solid var(--rule);border-radius:14px;padding:20px;min-height:220px;display:flex;flex-direction:column}
.card.broken{border-top:8px solid var(--broken)}.card.edited{border-top:8px solid var(--fixed)}.card.base{border-top:8px solid var(--rule)}
.name{font-size:22px;font-weight:600}.note{color:var(--body);font-size:15px;margin-bottom:12px}
.verdict{font-size:19px;font-weight:600;margin-bottom:10px}.ok{color:var(--good)}.no{color:var(--bad)}
.badge{align-self:flex-start;font:600 14px/1 ui-monospace,Menlo,monospace;padding:6px 10px;border-radius:999px;margin-bottom:12px}
.b-call{background:#F5DCCB;color:#7A3410}.b-answer{background:#D5E9DD;color:#13502F}.b-loop{background:#F3E3A8;color:#5B4500}.b-cut{background:#E3E7EE;color:#2B3645}
.out{font-size:18px;white-space:pre-wrap;word-break:break-word}
.call{font:16px ui-monospace,Menlo,monospace;background:var(--chip);border-radius:8px;padding:8px 10px;margin:0 0 6px;word-break:break-word}
.more{color:var(--body);font-size:15px}.secs{margin-top:auto;padding-top:10px;color:var(--body);font-size:14px}
.wait{color:var(--body)}.spin{display:inline-block;width:14px;height:14px;border:3px solid var(--rule);border-top-color:var(--fixed);border-radius:50%;animation:s 0.9s linear infinite;vertical-align:-2px;margin-right:8px}
@keyframes s{to{transform:rotate(360deg)}}
details{margin-top:8px}summary{cursor:pointer;color:var(--body);font-size:14px}
pre{white-space:pre-wrap;word-break:break-word;font-size:13px;max-height:240px;overflow:auto}
table.res{border-collapse:collapse;font-size:22px;width:100%;max-width:960px;background:var(--card);border:2px solid var(--rule);border-radius:14px;overflow:hidden}
.res th,.res td{padding:14px 20px;border-bottom:1px solid var(--rule);text-align:right}
.res th:first-child,.res td:first-child{text-align:left}.res th{font-size:16px;color:var(--body);font-weight:600}
.res tr:last-child td{border-bottom:0}.res .lo{color:var(--broken);font-weight:600}.res .hi{color:var(--fixed);font-weight:600}
.small{color:var(--body);font-size:16px;margin-top:10px}
.cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:22px}
@media (max-width:900px){.cols{grid-template-columns:1fr}}
.box{background:var(--card);border:2px solid var(--rule);border-radius:14px;padding:18px 22px}
.box.fix{border-color:var(--fixed)}.box h3{font-size:20px;margin:6px 0 6px}.box p{margin:0;color:var(--body)}
.tag{font:600 13px/1 ui-monospace,Menlo,monospace;letter-spacing:1.5px;color:var(--broken)}.box.fix .tag{color:var(--fixed)}
.figwrap{background:var(--card);border:2px solid var(--rule);border-radius:14px;padding:16px}
.fig{width:100%;height:auto;display:block}
.fig .f-ink{fill:var(--ink)}.fig .f-body{fill:var(--body)}.fig .f-broken{fill:var(--broken)}.fig .f-fixed{fill:var(--fixed)}.fig .f-onfix{fill:#fff}
.fig .f-answer{fill:var(--answer)}.fig .s-ink{stroke:var(--ink)}.fig .s-fixed{stroke:var(--fixed)}.fig .s-broken{stroke:var(--broken)}
.fig .d-tool{fill:var(--ink)}.fig .d-no{fill:var(--card);stroke:var(--ink);stroke-width:2}
.fig .l-plain{fill:var(--chip)}.fig .l-push{fill:var(--push);stroke:var(--broken);stroke-width:1.5}.fig .l-fix{fill:var(--fixed)}
.warn{background:#FBEDE3;border:2px solid var(--broken);border-radius:14px;padding:18px 24px;margin-bottom:22px}
@media (prefers-color-scheme:dark){.warn{background:#2E2119}}
.warn h3{margin:0 0 6px;font-size:22px;color:var(--broken)}.warn p{margin:6px 0}
ul.lessons{font-size:20px;padding-left:24px;max-width:960px}ul.lessons li{margin-bottom:12px}ul.lessons .why{display:block;color:var(--body);font-size:17px}
.banner{background:var(--ink);color:var(--bg);border-radius:14px;padding:16px 24px;font-size:19px;margin-top:8px}
.repro{display:grid;grid-template-columns:1fr 300px;gap:36px;align-items:center}
@media (max-width:760px){.repro{grid-template-columns:1fr}}
.qrcard{background:#fff;border-radius:16px;padding:28px;width:300px;height:300px}.qrcard svg{width:100%;height:100%;display:block}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:17px}
</style></head><body><main>

<header class="hero">
<h1>Claude, please fix my model</h1>
<p class="exp">Replicating weight-space programming, then using it to repair a broken tool router</p>
<p class="cite">Builds on <a href="__PAPER__">&ldquo;LLMs can natively read &amp; write neural network weights&rdquo;</a> &mdash; Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL).
Our write-up: <a href="__PDF__">the paper (PDF)</a>.</p>
</header>

<section class="section">
<h2>Demo: Improving a tool router</h2>
<p class="sub">Three versions of Qwen3.5-2B, running live __WHERE__. Pick a prompt and press Go.</p>
<div class="row" id="pick"></div>
<div class="prompt" id="prompt"></div>
<button id="go">Go</button>
<div class="grid" id="grid"></div>
</section>

<section class="section">
<h2>Results</h2>
<p class="lead">Fine-tuning on 58,000 xLAM examples improved tool calling, but the router began calling a tool on 99% of requests where none fits. Fable's edit restored &ldquo;no tool&rdquo; without retraining.</p>
<table class="res">
<tr><th>BFCL test</th><th>Base model</th><th>Tuned model</th><th>Fixed by Fable</th><th>Retrained with better data</th></tr>
<tr><td>Mean call accuracy</td><td>68%</td><td>85%</td><td>82%</td><td>83%</td></tr>
<tr><td>Declines when no tool fits</td><td>79%</td><td class="lo">1%</td><td class="hi">85%</td><td>83%</td></tr>
<tr><td>Good answers (of 240)</td><td>81</td><td class="lo">0</td><td>43</td><td class="hi">75</td></tr>
</table>
<p class="small">BFCL v4 test set, sealed: Fable never saw it. &ldquo;Good answers&rdquo;: a blind judge (Claude Sonnet 5.5) rated each text reply to a request no tool fits; good = helpfulness 4 or 5 out of 5.
&ldquo;Retrained&rdquo;: a second LoRA with 3,000 added &ldquo;no tool fits&rdquo; examples, trained with gradients. The fix shown here is one of ten Fable sessions; all ten found the same cause.</p>
</section>

<section class="section">
<h2>What Fable found: a small fix for widespread damage</h2>
<div class="cols">
<div class="box"><span class="tag">STILL INTACT</span><h3>It still knows when a tool fits</h3><p>The trained model still tells fitting from non-fitting requests (0.96 separation).</p></div>
<div class="box"><span class="tag">ADDED BY TRAINING</span><h3>A push toward a tool call</h3><p>A near-constant push of about 10 logits toward a tool call on every request. The fine-tune changed 58% of the model's weights, and the push is spread across them.</p></div>
<div class="box fix"><span class="tag">FABLE'S FIX</span><h3>A counter-push at the decision point</h3><p>Cancel the push in the last layer, where it becomes a decision, plus a small nudge toward starting a plain answer. Two columns, about 4,000 weights out of 2.2 billion. Fable compensated rather than removed the damage.</p></div>
</div>
<div class="figwrap">__FIG__</div>
<p class="small">Tuned row: the measured mean ± spread on Fable's practice prompts. Fixed row: those numbers moved by the edit's designed effect, so an estimate. Dots show the spread, not individual prompts.</p>
</section>

<section class="section">
<h2>What to watch for</h2>
<div class="warn"><h3>What nobody trained for: answer quality</h3>
<p>The tuned model didn't just hide its ability to answer, it largely lost it: forced to reply in text, it gave only <b>11</b> good answers out of 240, mostly refusals of questions it knows. Fable's edit brings that to <b>43</b>, but a fix at the last layer can't restore what the fine-tune removed upstream. Some answers loop.</p>
<p>A smaller cost: on real users' single-call requests the fixed router skipped the call 12.8% of the time (<b>Base 66% &middot; Tuned 70% &middot; Fixed 65%</b>). Fable's practice data had no requests like these, so it couldn't see the loss.</p></div>
<ul class="lessons">
<li>The model learns exactly what it is trained on.<span class="why">Every one of the 58,000 xLAM examples ends in a tool call, so the router learned to always call. Retraining with &ldquo;no tool fits&rdquo; examples is still the better fix (75 good answers). The same goes for the agent's data: when our practice set put real-user text only on the &ldquo;no tool&rdquo; side, Fable's edit learned the writing style and lost 19 points of call accuracy.</span></li>
<li>Measure what you didn't train for.<span class="why">Answer quality was the most important measurement, and we added it late. Most of the failures here (refusals, loops, the style shortcut) were never anticipated.</span></li>
<li>Replicate the original first, then change one thing at a time.<span class="why">We matched MIT's math result within about 3 points before changing anything. Each later step (base model, LoRA, new driver, new task, repeats) had a gate to pass.</span></li>
<li>Write predictions down, and keep the test set out of the agent's reach.<span class="why">We expected a bigger budget to help. Across nine runs it made no measurable difference; writing the prediction first is what made that visible.</span></li>
</ul>
<div class="banner">The original MIT paper is about a month old. These are experiments, not a method. Keep experimenting.</div>
</section>

<section class="section">
<h2>How to reproduce</h2>
<div class="repro"><div>
<p class="lead">Simple: point your coding agent at the <a href="__PAPER__">MIT CSAIL paper</a> and ask it to reproduce the math improvement experiment.</p>
<p class="lead">Here is the QR code for my repository:<br><a class="mono" href="__REPO__">github.com/bradhinkel/weight-space-tuning</a></p>
<p class="lead"><b>Next steps:</b></p>
<ul class="lessons"><li>Smaller models, more runs: the same method on a ResNet-50 image classifier (Waterbirds) beat the published fix by about 0.7 points.</li><li>Bigger, better-matched practice sets, especially for judging answers.</li><li>Edits further upstream, where the fine-tune changed how the model responds, to restore answer quality.</li></ul>
</div><div class="qrcard">__QR__</div></div>
</section>
</main>
<script>
const MODELS = __MODELS__;
let data = {demo:[], toolsets:[]}, cur = 0, toolset = 1;   // cur = index into data.demo, or -1 for your own prompt
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const chips = ts => ts.map(t=>`<span class="tool">${esc(t)}</span>`).join('');
function buttons(){
  const row=$('#pick'); row.innerHTML='<span class="lbl">Prompt</span>';
  const add=(i,key,label)=>{const b=document.createElement('button'); b.className='case'+(cur===i?' on':'');
    b.innerHTML=`<b>${key}</b>${esc(label)}`; b.onclick=()=>{cur=i;render()}; row.appendChild(b);};
  data.demo.forEach((d,i)=>add(i,d.key,d.label)); add(-1,'E','Your own prompt');
}
function render(){
  buttons();
  if(cur>=0){
    const d=data.demo[cur];
    $('#prompt').innerHTML=`<span class="lbl">User</span><div class="q">${esc(d.question)}</div><span class="lbl">Tools offered</span><div class="tools">${chips(d.tools)}</div>
      <div class="watch"><b>Watch for:</b> ${esc(d.watch)}</div><div class="gold">${esc(d.note)}</div>`;
  } else {
    const ts=data.toolsets.find(t=>t.case===toolset);
    const opts=data.toolsets.map(t=>`<option value="${t.case}" ${t.case===toolset?'selected':''}>${esc(t.label)}</option>`).join('');
    $('#prompt').innerHTML=`<span class="lbl">User</span><textarea id="q" rows="2" placeholder="Type a request…">${esc(window._q||'')}</textarea>
      <div class="row" style="margin-top:12px"><span class="lbl">Tools offered</span><select id="toolset">${opts}</select></div><div class="tools">${chips(ts.tools)}</div>`;
    $('#q').oninput=e=>window._q=e.target.value; $('#toolset').onchange=e=>{toolset=+e.target.value;render()};
  }
  resetGrid();
}
function resetGrid(){
  $('#grid').innerHTML=MODELS.map(([k,n,d])=>`<div class="card ${k}" id="c-${k}"><div class="name">${n}</div><div class="note">${d}</div><div class="wait">Waiting</div></div>`).join('');
}
function show(k,r){
  const lab={call:'Calls a tool',answer:'Answers',loop:'Answers, but loops',cut:'Answers (cut short)'}[r.verdict];
  let body='';
  if(r.calls.length){
    body=r.calls.slice(0,4).map(c=>`<div class="call">${esc(c.name)}(${esc(Object.entries(c.args).map(([a,v])=>a+'='+v).join(', '))})</div>`).join('');
    if(r.calls.length>4) body+=`<div class="more">+ ${r.calls.length-4} more calls</div>`;
  } else { const t=r.answer; body=`<div class="out">${esc(t.length>420?t.slice(0,420)+' …':t)}</div>`; }
  const v = r.ok===undefined ? '' : `<div class="verdict ${r.ok?'ok':'no'}">${r.ok?'✓':'✗'} ${esc(r.why)}</div>`;
  const [_,n,d]=MODELS.find(m=>m[0]===k);
  $('#c-'+k).innerHTML=`<div class="name">${n}</div><div class="note">${d}</div>${v}<span class="badge b-${r.verdict}">${lab}</span>${body}
    <details><summary>Raw output</summary><pre>${esc(r.raw)}</pre></details><div class="secs">${r.seconds}s __WHERE__</div>`;
}
$('#go').onclick=async()=>{
  const q = cur<0 ? ($('#q').value||'').trim() : '';
  if(cur<0 && !q){ $('#q').focus(); return; }
  const kase = cur<0 ? toolset : data.demo[cur].case;
  $('#go').disabled=true; resetGrid();
  for(const [k] of MODELS){
    const c=$('#c-'+k); c.querySelector('.wait')?.remove();
    c.insertAdjacentHTML('beforeend','<div class="wait" id="w-'+k+'"><span class="spin"></span>Thinking…</div>');
    try{
      const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case:kase,model:k,question:q})});
      show(k, await r.json());
    }catch(e){ c.querySelector('#w-'+k).textContent='Error: '+e; }
  }
  $('#go').disabled=false;
};
fetch('/api/cases').then(r=>r.json()).then(j=>{data=j;render()});
</script></body></html>"""


def page():
    return (PAGE.replace("__MODELS__", json.dumps(MODELS)).replace("__FIG__", figure()).replace("__QR__", qr_svg())
            .replace("__REPO__", REPO).replace("__PAPER__", PAPER).replace("__PDF__", PDF).replace("__WHERE__", WHERE))


def handler(app):
    html = page()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, html, "text/html; charset=utf-8")
            elif self.path == "/api/cases":
                self.send(200, json.dumps(app.cases()), "application/json")
            else:
                self.send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/run":
                return self.send(404, "not found", "text/plain")
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if req.get("model") not in {m for m, _, _ in MODELS}:
                return self.send(400, "bad model", "text/plain")
            try:
                self.send(200, json.dumps(app.run(req)), "application/json")
            except Exception as e:  # endpoint down, cold-start timeout, ...
                self.send(502, f"model call failed: {e}", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--html", help="write the page (no live models) to this file and exit")
    ap.add_argument("--where", default=WHERE, help='where the models run, as shown on the page ("on this laptop")')
    ap.add_argument("--endpoint", help="RunPod Serverless endpoint id: run the models there (needs RUNPOD_API_KEY)")
    a = ap.parse_args()
    globals()["WHERE"] = a.where
    if a.html:
        Path(a.html).write_text(page())
        return print(f"wrote {a.html}")
    app = App(a.max_tokens, a.endpoint)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(app))
    print(f"demo ready: http://localhost:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
