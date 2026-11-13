"""A.11c practice data: remove the style confound in the A.5 practice set (run in the bfcl-eval venv).

    /workspace/bfcl-venv/bin/python -m exp_a.build_practice_v2

The A.5 set had live (real-user) style only on the irrelevant side and synthetic (xLAM) style only on the
relevant side, so a probe could separate "irrelevant" from "relevant" by style. A.11b's probe did, and the
edit stopped calling tools on live relevant test prompts (reports/A.11.md). This adds the two missing cells:

  live, relevant      live_relevant.jsonl     BFCL live_parallel + live_parallel_multiple (with reference
                                              calls) and live_relevance (any call to an offered tool is
                                              correct). None of these categories is in the Exp A test set.
  synthetic, irrelevant  xlam_mismatched.jsonl  xLAM practice queries paired with the tool list of a different
                                              xLAM practice example, kept only when no tool name is shared and
                                              no content word of the query appears in the offered tools.

The existing live_irrelevance.jsonl and xlam_practice.jsonl are unchanged. Every user message is checked
against the test guard (sha256 of test-category user messages); matches are dropped.
Writes to /workspace/artifacts/a/practice_v2/ and a manifest with counts and the filter settings.
"""
import json
import random
import re
import shutil
from pathlib import Path

import bfcl_eval

from common.evals.bfcl_qwen35 import _to_tools
from common.harness.guard import text_hash

DATA = Path(bfcl_eval.__file__).parent / "data"
A = Path("/workspace/artifacts/a")
OUT = A / "practice_v2"
LIVE_RELEVANT = ["live_parallel", "live_parallel_multiple", "live_relevance"]
N_MISMATCHED = 600
STOP = set("""a an the and or of to in on for with by from at as is are be was were it this that these those
what which who whom how when where why can could would should will do does did my me i you your we our us
get find give show list tell please about into than then them they their there here some any all each also
using use make want need help like just more most other such only over under between""".split())


def words(text):
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


def tool_text(tools):
    parts = []
    for t in tools:
        f = t.get("function", t)
        parts += [f.get("name", ""), f.get("description", "")]
        props = (f.get("parameters") or {}).get("properties") or {}
        for k, v in props.items():
            parts += [k, str(v.get("description", "")) if isinstance(v, dict) else ""]
    return " ".join(parts).replace("_", " ").replace(".", " ")


def tool_names(tools):
    return {t.get("function", t).get("name") for t in tools}


def user_text(messages):
    return " ".join(m["content"] for m in messages if m.get("role") == "user")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    guard = set(json.loads((A / "test_guard.json").read_text())["sha256"])
    rng = random.Random(0)
    manifest = {}

    # live, relevant
    n_rel, dropped = {}, 0
    with open(OUT / "live_relevant.jsonl", "w") as f:
        for cat in LIVE_RELEVANT:
            rows = [json.loads(l) for l in open(DATA / f"BFCL_v4_{cat}.json")]
            ans_path = DATA / "possible_answer" / f"BFCL_v4_{cat}.json"
            answers = {}
            if ans_path.exists():
                answers = {r["id"]: r["ground_truth"] for r in map(json.loads, open(ans_path))}
            k = 0
            for r in rows:
                msgs = r["question"][0]
                if any(text_hash(m["content"]) in guard for m in msgs if m.get("role") == "user"):
                    dropped += 1
                    continue
                item = {"id": r["id"], "messages": msgs, "tools": _to_tools(r["function"]), "source": cat}
                if r["id"] in answers:
                    item["reference_calls"] = answers[r["id"]]
                else:
                    item["reference_calls"] = None   # live_relevance: any call to an offered tool is correct
                f.write(json.dumps(item) + "\n")
                k += 1
            n_rel[cat] = k
    manifest["live_relevant"] = {"per_source": n_rel, "dropped_by_guard": dropped}

    # synthetic, irrelevant: same xLAM queries, mismatched tool lists
    xl = [json.loads(l) for l in open(A / "practice/xlam_practice.jsonl")]
    tt = [(tool_names(x["tools"]), words(tool_text(x["tools"]))) for x in xl]
    order = list(range(len(xl)))
    rng.shuffle(order)
    out, tried = [], 0
    for i in order:
        if len(out) >= N_MISMATCHED:
            break
        q_msgs = [m for m in xl[i]["messages"] if m.get("role") == "user"]
        qw = words(user_text(q_msgs))
        if text_hash(user_text(q_msgs)) in guard:
            continue
        for _ in range(20):
            tried += 1
            j = rng.randrange(len(xl))
            if j == i or tt[j][0] & tt[i][0] or qw & tt[j][1]:
                continue
            out.append({"id": f"{xl[i]['id']}__tools_of__{xl[j]['id']}", "messages": q_msgs,
                        "tools": xl[j]["tools"], "query_from": xl[i]["id"], "tools_from": xl[j]["id"]})
            break
    with open(OUT / "xlam_mismatched.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    manifest["xlam_mismatched"] = {"written": len(out), "pairs_tried": tried,
                                   "filter": "no shared tool name; no query content word (>=4 letters, "
                                             "not a stopword) in tool names/descriptions/params"}

    for name in ["live_irrelevance.jsonl", "xlam_practice.jsonl"]:
        shutil.copy(A / "practice" / name, OUT / name)
    manifest["unchanged"] = {n: sum(1 for _ in open(OUT / n)) for n in ["live_irrelevance.jsonl",
                                                                       "xlam_practice.jsonl"]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    for r in rng.sample(out, 8):
        print("---", user_text(r["messages"])[:160], "|| TOOLS:", sorted(tool_names(r["tools"])))


if __name__ == "__main__":
    main()
