"""R.4b practice data, second construction: real users' tools + a request written for them, run locally.

    python3 -m exp_r.r4b.build_live_simple_rt --out artifacts/r4b/live_simple_style_rt.jsonl

The first construction (build_live_simple.py: real prompts + a generated tool) failed the R.4b gate: clean generated
tools are easy relevance cases (exp_r/R4b_design.md). Here the TOOLS are real: the 877 distinct tool definitions
offered in the practice live_irrelevance items (real users' APIs, with their messy names, long descriptions and odd
schemas). For each sampled tool, Claude (the judge's model, `claude` CLI on Max, no tools) writes a request in the
style of real users (short, informal, sometimes another language) that this tool serves with one call, and the
reference call in BFCL format; or skip. About a third of the items also offer the other tools of the same source
item (the same user's API set, like BFCL live_multiple). The test guard runs on the pod (exp_r.r4b.filter_guard).
"""
import argparse
import concurrent.futures as cf
import json
import random
from pathlib import Path

from exp_r.r4b.build_live_simple import claude, valid

PROMPT = """You are building evaluation data for function calling. Below are tool definitions taken from real
users' applications. For EACH tool, write ONE request that a real user of that application might send to an
assistant that has this tool, such that the right response is exactly one call to this tool. Then give that call.

Style of the request (vary it across items, like real users): often short and informal, sometimes with typos or
missing punctuation, sometimes with a bit of context about what they are doing, sometimes in Chinese, Spanish or
another language (about 1 in 8), sometimes phrased indirectly. It must not mention the function name. All values
the call needs must be stated or clearly implied by the request. Do NOT copy any example from the tool's own
description. If the tool is too broken or vague to call meaningfully, answer skip.

Reference call (BFCL format): {"<function name>": {"<arg>": [allowed values], ...}}
- Every argument the call should pass, with ALL reasonable equivalent values the request supports; values must
  have the schema's type (numbers as numbers).
- For an optional argument the request does not specify, list ["", <its default>] if it has a default, else [""].
- Every required argument must be determined by the request.
Reply with ONLY a JSON array, one object per tool, same order: {"key": ..., "skip": true} or
{"key": ..., "skip": false, "request": "...", "reference": {...}}.

TOOLS:
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="artifacts/r4b/live_irrelevance.jsonl")
    ap.add_argument("--out", default="artifacts/r4b/live_simple_style_rt.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--keep", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    src = [json.loads(l) for l in open(a.src)]
    first = {}                                   # distinct tool -> (source item, index of the tool in it)
    for r in src:
        for i, t in enumerate(r["tools"]):
            first.setdefault(json.dumps(t, sort_keys=True), (r, i))
    keys = sorted(first)
    rng.shuffle(keys)
    pool = [(f"t{k}", first[s][0], first[s][1]) for k, s in enumerate(keys[: a.n])]
    raw_path = Path(a.out).with_suffix(".raw.jsonl")
    done = {json.loads(l)["key"]: json.loads(l) for l in open(raw_path)} if raw_path.exists() else {}
    todo = [p for p in pool if p[0] not in done]
    batches = [todo[i: i + a.batch] for i in range(0, len(todo), a.batch)]

    def work(b):
        text = PROMPT + "\n\n".join(f'### key: {k}\n{json.dumps(r["tools"][i])[:4000]}' for k, r, i in b)
        ks = {k for k, _, _ in b}
        return [x for x in claude(text, a.model) if x.get("key") in ks]

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

    kept, reasons = [], {}
    for k, r, i in pool:
        x = done.get(k)
        if not x or x.get("skip") or not str(x.get("request", "")).strip():
            reasons["skip"] = reasons.get("skip", 0) + 1
            continue
        tool = r["tools"][i]
        why = valid(tool, x.get("reference") or {})
        if why:
            key = " ".join(why.split(" ")[:2])
            reasons[key] = reasons.get(key, 0) + 1
            continue
        kept.append((k, r, i, x))
    rng.shuffle(kept)
    out = []
    for k, r, i, x in kept[: a.keep]:
        tools = [r["tools"][i]]
        others = [t for j, t in enumerate(r["tools"]) if j != i
                  and t["function"]["name"] != r["tools"][i]["function"]["name"]]
        if others and rng.random() < 0.3:        # about a fifth overall: most items offer one tool, like live_simple
            tools = [r["tools"][i]] + others
            rng.shuffle(tools)
        msgs = [m for m in r["messages"] if m["role"] == "system"][:1] + [{"role": "user", "content": x["request"]}]
        out.append({"id": f"live_single_{k}", "messages": msgs, "tools": tools, "source": "live_simple_style",
                    "from": r["id"], "reference_calls": [x["reference"]]})
    with open(a.out, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(json.dumps({"tools_sampled": len(pool), "answered": sum(p[0] in done for p in pool), "kept": len(out),
                      "dropped": reasons, "with_other_tools": sum(len(o["tools"]) > 1 for o in out)}, indent=1))


if __name__ == "__main__":
    main()
