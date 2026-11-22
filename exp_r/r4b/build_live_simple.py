"""R.4b practice data: live_simple-style items (real users' prompts + one fitting tool), run locally.

    python3 -m exp_r.r4b.build_live_simple --src artifacts/r4b/live_irrelevance.jsonl --out artifacts/r4b/live_simple_style.jsonl

For each sampled real-user prompt from the practice live_irrelevance file, Claude (the judge's model, through the
local `claude` CLI on Max, no tools) writes one function that a real API would offer for the request and the
reference call in BFCL format ({name: {arg: [allowed values]}}, "" = may be omitted), or "skip". About a third of
the kept items also get 1-2 distractor tools taken from OTHER live_irrelevance items (real users' tool definitions)
that share no content word with the prompt. Checks here: JSON shape, the reference type-checks against the schema,
required arguments present. The test guard and the test function-name check run on the pod (exp_r.r4b.check_group).
See exp_r/R4b_design.md.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import subprocess
from pathlib import Path

STOP = set("""a an the and or of to in on for with by from at as is are be was were it this that these those
what which who whom how when where why can could would should will do does did my me i you your we our us
get find give show list tell please about into than then them they their there here some any all each also
using use make want need help like just more most other such only over under between""".split())  # = exp_a.build_practice_v2

PROMPT = """You are building evaluation data for function calling. Below are real users' messages to an assistant.
For EACH message, decide whether a single call to one function of a plausible real-world API (a web service, a
device, a database, a calculator, a code-execution or search tool, a company's internal API...) would be the
natural way to serve the request. If yes, write that function and the exact call. If the request is chit-chat, an
opinion, creative writing, needs several different calls, or is unclear, answer skip.

Rules for the function:
- OpenAI tool format: {"type": "function", "function": {"name": snake_case or dotted name, "description": ...,
  "parameters": {"type": "object", "properties": {...}, "required": [...]}}}.
- Realistic, like a real API: 2-6 parameters, typed (string, integer, number, boolean, array, object), descriptions,
  enums where natural, some optional parameters with "default".
- The function must be generic (e.g. get_weather(location, unit)), not tailored with the user's values baked in.
Rules for the reference call (BFCL format): {"<function name>": {"<arg>": [allowed values], ...}}
- Every argument the call should pass, with ALL reasonable equivalent values the user's message supports (e.g.
  ["San Francisco, CA", "San Francisco, California"]); values must have the schema's type (numbers as numbers).
- For an optional argument the user did not specify, list ["", <its default>] ("" = may be omitted).
- Required arguments must be determined by the message; if they are not, answer skip.
Reply with ONLY a JSON array, one object per message, same order: {"key": ..., "skip": true} or
{"key": ..., "skip": false, "tool": {...}, "reference": {...}}.

MESSAGES:
"""

TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def claude(text, model):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    cmd = ["claude", "-p", text, "--model", model, "--tools", "", "--strict-mcp-config", "--output-format", "json",
           "--no-session-persistence"]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, cwd="/tmp")
    if out.returncode:
        raise RuntimeError(out.stderr[-500:] or out.stdout[-500:])
    t = json.loads(out.stdout)["result"]
    return json.loads(t[t.index("["): t.rindex("]") + 1])


def valid(tool, ref):
    """None if the tool and its reference call are consistent, else the reason."""
    try:
        f = tool["function"]
        name, params = f["name"], f["parameters"]
        props, req = params.get("properties") or {}, params.get("required") or []
    except Exception:
        return "bad tool shape"
    if tool.get("type") != "function" or not props:
        return "bad tool shape"
    if list(ref) != [name] or not isinstance(ref[name], dict):
        return "reference names another function"
    args = ref[name]
    for r in req:
        if r not in args or "" in args[r]:
            return f"required {r} missing or optional"
    for k, vals in args.items():
        if k not in props or not isinstance(vals, list) or not vals:
            return f"argument {k} not in schema"
        ty = TYPES.get((props[k] or {}).get("type"))
        for v in vals:
            if v == "":
                continue
            if ty and not isinstance(v, ty) or (ty is int and isinstance(v, bool)):
                return f"argument {k} value {v!r} has the wrong type"
            enum = (props[k] or {}).get("enum")
            if enum and v not in enum:
                return f"argument {k} value {v!r} not in enum"
    return None


def words(text):
    return {w for w in re.findall(r"[a-z]{4,}", str(text).lower()) if w not in STOP}


def tool_words(t):
    """Content words of a tool: its name, description and parameter names/descriptions (not schema keywords)."""
    f = t.get("function", t)
    parts = [f.get("name", ""), f.get("description", "")]
    for k, v in ((f.get("parameters") or {}).get("properties") or {}).items():
        parts += [k, v.get("description", "") if isinstance(v, dict) else ""]
    return words(" ".join(parts).replace("_", " ").replace(".", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="artifacts/r4b/live_irrelevance.jsonl")
    ap.add_argument("--out", default="artifacts/r4b/live_simple_style.jsonl")
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--keep", type=int, default=300)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    src = [json.loads(l) for l in open(a.src)]
    single = [r for r in src if [m["role"] for m in r["messages"]] in (["user"], ["system", "user"])]
    pool = rng.sample(single, min(a.n, len(single)))
    raw_path = Path(a.out).with_suffix(".raw.jsonl")
    done = {json.loads(l)["key"]: json.loads(l) for l in open(raw_path)} if raw_path.exists() else {}
    todo = [r for r in pool if r["id"] not in done]
    batches = [todo[i: i + a.batch] for i in range(0, len(todo), a.batch)]

    def work(b):
        text = PROMPT + "\n\n".join(f'### key: {r["id"]}\n{r["messages"][-1]["content"][:3000]}' for r in b)
        res = claude(text, a.model)
        keys = {r["id"] for r in b}
        return [x for x in res if x.get("key") in keys]

    with cf.ThreadPoolExecutor(a.workers) as ex, open(raw_path, "a") as f:
        for n, fut in enumerate(cf.as_completed([ex.submit(work, b) for b in batches]), 1):
            try:
                for x in fut.result():
                    done[x["key"]] = x
                    f.write(json.dumps(x) + "\n")
                f.flush()
                print(f"batch {n}/{len(batches)} ok", flush=True)
            except Exception as e:
                print(f"batch {n}/{len(batches)} FAILED: {e}", flush=True)

    by_id = {r["id"]: r for r in src}
    kept, reasons = [], {}
    for r in pool:
        x = done.get(r["id"])
        if not x or x.get("skip"):
            reasons["skip" if x else "no answer"] = reasons.get("skip" if x else "no answer", 0) + 1
            continue
        why = valid(x.get("tool") or {}, x.get("reference") or {})
        if why:
            reasons[why.split(" ")[0] + " " + why.split(" ")[1]] = reasons.get(why.split(" ")[0] + " " + why.split(" ")[1], 0) + 1
            continue
        kept.append((r, x))
    rng.shuffle(kept)
    kept = kept[: a.keep]
    out = []
    for r, x in kept:
        tools = [x["tool"]]
        if rng.random() < 1 / 3:
            q = words(r["messages"][-1]["content"]) | tool_words(x["tool"])
            names = {x["tool"]["function"]["name"]}
            cands = [t for o in src if o["id"] != r["id"] for t in o["tools"]
                     if t["function"]["name"] not in names and not (tool_words(t) & q)]
            for t in rng.sample(cands, min(len(cands), rng.choice([1, 2]))):
                if t["function"]["name"] not in names:
                    names.add(t["function"]["name"])
                    tools.append(t)
            rng.shuffle(tools)
        out.append({"id": "live_single_" + r["id"].removeprefix("live_irrelevance_"), "messages": r["messages"],
                    "tools": tools, "source": "live_simple_style", "from": r["id"],
                    "reference_calls": [x["reference"]]})
    with open(a.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(json.dumps({"sampled": len(pool), "answered": sum(r["id"] in done for r in pool), "kept": len(out),
                      "dropped": reasons, "with_distractors": sum(len(o["tools"]) > 1 for o in out)}, indent=1))


if __name__ == "__main__":
    main()
