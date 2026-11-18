"""Browser demo on the laptop: pick a prompt, press Go, watch the broken router, Fable's fix and the base model answer.

    ~/demo_env/.venv/bin/python -m exp_r.r4a.demo_web          # then open http://localhost:8765

Needs `python -m exp_r.r4a.demo_local setup` once (the broken model in ~/demo_models/broken). All three models run
live on the local GPU through demo_local.Demo; nothing leaves the machine. Answers are cut at 400 tokens.
"""
import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from exp_r.r4a import demo_local as D

SHORT = ["Freezing point", "Merge three lists", "U.S. location", "LCM of 50 and 75", "Image search",
         "TV show + jokes", "Delete system32"]
FEATURED = [1, 4, 7]
MODELS = [("broken", "Broken router", "LoRA on 58K tool-call examples"),
          ("edited", "Fixed by Fable", "2 columns of 1 layer edited"),
          ("base", "Base model", "Qwen3.5-2B as released")]


def parse(text, hit_cap):
    calls = []
    for fn, body in re.findall(r"<function=([^>\s]+)>(.*?)</function>", text, re.S):
        args = {k: " ".join(v.split()) for k, v in re.findall(r"<parameter=([^>\s]+)>(.*?)</parameter>", body, re.S)}
        calls.append({"name": fn, "args": args})
    kind, _ = D.summarize(text, hit_cap, 10_000)
    verdict = "call" if calls else ("loop" if "loops" in kind else ("cut" if "cut" in kind else "answer"))
    answer = "" if calls else " ".join(text.split())
    return {"verdict": verdict, "calls": calls, "answer": answer, "raw": text}


class App:
    def __init__(self, max_tokens):
        self.items = json.load(open(D.ITEMS))
        self.demo = D.Demo({"broken", "edited", "base"})
        self.lock = threading.Lock()
        self.max_tokens = max_tokens

    def cases(self):
        out = []
        for k, it in enumerate(self.items, 1):
            out.append({"n": k, "label": SHORT[k - 1], "featured": k in FEATURED, "question": it["question"],
                        "tools": [t["function"]["name"] for t in it["tools"]],
                        "expect": "No tool fits: a good router answers directly" if it["kind"] == "no_call"
                        else "A tool fits: a good router calls it"})
        return out

    def run(self, req):
        item = self.items[int(req["case"]) - 1]
        q = (req.get("question") or "").strip()
        prompt = D.render_prompt(self.demo.tok, q, item["tools"]) if q else item["prompt"]
        with self.lock:
            self.demo.use(req["model"])
            t0 = time.time()
            text, cap = self.demo.generate(prompt, self.max_tokens)
        return {**parse(text, cap), "seconds": round(time.time() - t0, 1)}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tool Router Demo</title>
<style>
:root{--bg:#F6F5F0;--card:#FFFDF8;--ink:#16202E;--body:#4A5568;--rule:#D9D4C7;--broken:#B5501A;--fixed:#2459C4;--good:#1F7A4D;--chip:#ECE8DD}
@media (prefers-color-scheme:dark){:root{--bg:#121923;--card:#1B2431;--ink:#EEF1F5;--body:#B4BFCC;--rule:#2E3A4A;--broken:#F08A4B;--fixed:#7FA7F5;--good:#5CC08A;--chip:#26313F}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.45 "IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1400px;margin:0 auto;padding:28px 24px 48px}
h1{font-size:30px;margin:0 0 4px}.sub{color:var(--body);margin:0 0 24px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:10px}
.lbl{font:600 13px/1 ui-monospace,Menlo,monospace;letter-spacing:1.5px;color:var(--body);text-transform:uppercase;margin-right:6px}
button.case{border:2px solid var(--rule);background:var(--card);color:var(--ink);border-radius:999px;padding:9px 18px;font:inherit;font-size:17px;cursor:pointer}
button.case.on{border-color:var(--ink);background:var(--ink);color:var(--bg)}
.prompt{background:var(--card);border:2px solid var(--rule);border-radius:14px;padding:20px 24px;margin:18px 0}
.q{font-size:26px;font-weight:600;margin:4px 0 10px}
textarea{width:100%;font:inherit;font-size:22px;border:2px solid var(--rule);border-radius:10px;padding:10px 12px;background:var(--bg);color:var(--ink);resize:vertical}
.tools{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}.tool{font:14px ui-monospace,Menlo,monospace;background:var(--chip);border-radius:6px;padding:4px 8px}
.expect{color:var(--body);margin-top:10px}
select{font:inherit;padding:6px;border-radius:8px;border:2px solid var(--rule);background:var(--card);color:var(--ink)}
#go{font:600 22px/1 inherit;font-family:inherit;padding:16px 44px;border:0;border-radius:12px;background:var(--fixed);color:#fff;cursor:pointer}
#go:disabled{opacity:.5;cursor:wait}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:22px}
@media (max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:2px solid var(--rule);border-radius:14px;padding:20px;min-height:220px;display:flex;flex-direction:column}
.card.broken{border-top:8px solid var(--broken)}.card.edited{border-top:8px solid var(--fixed)}.card.base{border-top:8px solid var(--rule)}
.name{font-size:22px;font-weight:600}.note{color:var(--body);font-size:15px;margin-bottom:12px}
.badge{align-self:flex-start;font:600 15px/1 ui-monospace,Menlo,monospace;padding:7px 12px;border-radius:999px;margin-bottom:12px}
.b-call{background:#F5DCCB;color:#7A3410}.b-answer{background:#D5E9DD;color:#13502F}.b-loop{background:#F3E3A8;color:#5B4500}.b-cut{background:#E3E7EE;color:#2B3645}
.out{font-size:19px;white-space:pre-wrap;word-break:break-word}
.call{font:16px ui-monospace,Menlo,monospace;background:var(--chip);border-radius:8px;padding:8px 10px;margin:0 0 6px}
.more{color:var(--body);font-size:15px}.secs{margin-top:auto;padding-top:10px;color:var(--body);font-size:14px}
.wait{color:var(--body)}.spin{display:inline-block;width:14px;height:14px;border:3px solid var(--rule);border-top-color:var(--fixed);border-radius:50%;animation:s 0.9s linear infinite;vertical-align:-2px;margin-right:8px}
@keyframes s{to{transform:rotate(360deg)}}
details{margin-top:8px}summary{cursor:pointer;color:var(--body);font-size:14px}
pre{white-space:pre-wrap;word-break:break-word;font-size:13px;max-height:240px;overflow:auto}
</style></head><body><main>
<h1>Does this request need a tool?</h1>
<p class="sub">Three versions of Qwen3.5-2B, running live on this laptop. Pick a prompt and press Go.</p>
<div class="row" id="featured"><span class="lbl">Demo</span></div>
<div class="row" id="others"><span class="lbl">More</span></div>
<div class="prompt" id="prompt"></div>
<button id="go">Go</button>
<div class="grid" id="grid"></div>
</main>
<script>
const MODELS = __MODELS__;
let cases = [], cur = 1, custom = false;
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function buttons(){
  for (const id of ['#featured','#others']) $(id).querySelectorAll('button').forEach(b=>b.remove());
  for (const c of cases){
    const b=document.createElement('button'); b.className='case'+(!custom&&c.n===cur?' on':''); b.textContent=c.label;
    b.onclick=()=>{cur=c.n;custom=false;render()}; (c.featured?$('#featured'):$('#others')).appendChild(b);
  }
  const b=document.createElement('button'); b.className='case'+(custom?' on':''); b.textContent='Your own prompt';
  b.onclick=()=>{custom=true;render()}; $('#others').appendChild(b);
}
function render(){
  buttons(); const c=cases[cur-1];
  const tools=c.tools.map(t=>`<span class="tool">${esc(t)}</span>`).join('');
  if(!custom){
    $('#prompt').innerHTML=`<span class="lbl">User</span><div class="q">${esc(c.question)}</div><span class="lbl">Tools offered</span><div class="tools">${tools}</div><div class="expect">${esc(c.expect)}</div>`;
  } else {
    const opts=cases.map(x=>`<option value="${x.n}" ${x.n===cur?'selected':''}>${esc(x.label)}: ${esc(x.tools.join(', '))}</option>`).join('');
    $('#prompt').innerHTML=`<span class="lbl">User</span><textarea id="q" rows="2" placeholder="Type a request…">${esc(window._q||'')}</textarea>
      <div class="row" style="margin-top:12px"><span class="lbl">Tools offered</span><select id="toolset">${opts}</select></div>`;
    $('#q').oninput=e=>window._q=e.target.value; $('#toolset').onchange=e=>{cur=+e.target.value;render()};
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
  const [_,n,d]=MODELS.find(m=>m[0]===k);
  $('#c-'+k).innerHTML=`<div class="name">${n}</div><div class="note">${d}</div><span class="badge b-${r.verdict}">${lab}</span>${body}
    <details><summary>Raw output</summary><pre>${esc(r.raw)}</pre></details><div class="secs">${r.seconds}s on this laptop</div>`;
}
$('#go').onclick=async()=>{
  const q = custom ? ($('#q').value||'').trim() : '';
  if(custom && !q){ $('#q').focus(); return; }
  $('#go').disabled=true; resetGrid();
  for(const [k] of MODELS){
    $('#c-'+k).querySelector('.wait')?.remove();
    const c=$('#c-'+k); c.insertAdjacentHTML('beforeend','<div class="wait" id="w-'+k+'"><span class="spin"></span>Thinking…</div>');
    try{
      const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({case:cur,model:k,question:q})});
      show(k, await r.json());
    }catch(e){ c.querySelector('#w-'+k).textContent='Error: '+e; }
  }
  $('#go').disabled=false;
};
fetch('/api/cases').then(r=>r.json()).then(j=>{cases=j;render()});
</script></body></html>"""


def handler(app):
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
                self.send(200, PAGE.replace("__MODELS__", json.dumps(MODELS)), "text/html; charset=utf-8")
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
            self.send(200, json.dumps(app.run(req)), "application/json")
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    app = App(a.max_tokens)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(app))
    print(f"demo ready: http://localhost:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
