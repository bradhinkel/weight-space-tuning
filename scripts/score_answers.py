"""Score the quality of models' text answers on BFCL v4 irrelevance (Experiment A follow-up).

BFCL irrelevance only checks that no tool was called. This script also asks *how good the answer is*.

* Tool-call responses are marked `called_tool` and not judged (on irrelevance every call is wrong).
* Text responses get automatic metrics (output tokens, hit the token cap, repeated-line loops) and a
  blind LLM judgement: items from all models are shuffled together, model names are hidden, and the
  judge (Claude via the local `claude` CLI on the Max plan, no tools) grades them in batches.

    python3 -m scripts.score_answers --runs a1_base a5_b25_fable_patched a7_lora --out artifacts/a/answers
    python3 -m scripts.score_answers --summarize-only --out artifacts/a/answers

Judgements are cached in <out>/judgements.jsonl, so a rerun resumes where it stopped.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PULLED = REPO / "artifacts/pulled"
DATA = PULLED / "bfcl_data/BFCL_v4_irrelevance.json"
CAP = 4096
SHOW_CHARS = 2000

RUBRIC = """You are grading answers from small language models. Each model was given a user question and a
list of tools (functions). None of the tools actually fit the question, so the right behaviour is to NOT call a
tool and instead either answer the question directly or explain briefly that the tools don't fit and help anyway.
Every answer below is a plain-text answer (no tool was called). Grade each one on its own.

For each item give:
- "helpfulness" 1-5: 5 = correct, complete, useful answer to the user's actual question (or, if the question
  cannot be answered without live data, a clear explanation plus useful guidance); 4 = mostly correct/useful with
  small flaws; 3 = only says the tools don't fit / declines with little help, OR a partly correct answer;
  2 = mostly wrong or unhelpful; 1 = wrong, nonsensical or no real answer.
- "fluency" 1-5: 5 = clear and well-formed; 3 = awkward or padded; 1 = incoherent or stuck repeating itself.
- "degenerate": true if the answer loops, repeats the same sentence/paragraph many times, or runs on aimlessly.
- "category": one of "answered" (attempts the question itself), "declined_with_help" (says tools don't fit and
  gives real help), "declined_only" (says tools don't fit, little or no help), "degenerate", "other".
- "note": at most 12 words.

Answers longer than %d characters are cut off where marked; judge what you see, and treat visible repetition as
repetition. Reply with ONLY a JSON array, one object per item, in the same order, each with keys
"key", "helpfulness", "fluency", "degenerate", "category", "note".

ITEMS:
""" % SHOW_CHARS


def load_run(run):
    path = PULLED / f"artifacts/bfcl/{run}/result/BFCL_v4_irrelevance_result.json"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def max_line_repeats(text):
    lines = [l.strip() for l in re.split(r"[\n]+|(?<=[.!?])\s+", text) if len(l.strip()) > 15]
    return max(Counter(lines).values(), default=0)


def judge(batch, model):
    items = "\n\n".join(
        f'### key: {b["key"]}\nQUESTION: {b["question"]}\nTOOLS OFFERED: {b["tools"]}\nANSWER:\n{b["shown"]}'
        for b in batch)
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    cmd = ["claude", "-p", RUBRIC + items, "--model", model, "--tools", "", "--strict-mcp-config",
           "--output-format", "json", "--no-session-persistence"]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900, cwd="/tmp")
    if out.returncode:
        raise RuntimeError(out.stderr[-500:] or out.stdout[-500:])
    text = json.loads(out.stdout)["result"]
    arr = json.loads(text[text.index("["): text.rindex("]") + 1])
    by_key = {a["key"]: a for a in arr}
    missing = [b["key"] for b in batch if b["key"] not in by_key]
    if missing:
        raise RuntimeError(f"judge omitted {missing}")
    return [by_key[b["key"]] for b in batch]


def summarize(out_dir, runs):
    recs = [json.loads(l) for l in (out_dir / "records.jsonl").read_text().splitlines()]
    judged = {}
    jpath = out_dir / "judgements.jsonl"
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            j = json.loads(l)
            judged.setdefault(j["key"], j)   # first judgement; repeats are for the consistency check
    summary = {}
    for run in runs:
        rs = [r for r in recs if r["run"] == run]
        text = [r for r in rs if not r["called_tool"]]
        js = [judged[r["key"]] for r in text if r["key"] in judged]
        n = len(rs)
        mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None
        good = sum(1 for j in js if j["helpfulness"] >= 4)
        summary[run] = {
            "prompts": n, "text_answers": len(text), "called_tool": n - len(text),
            "judged": len(js),
            "hit_token_cap": sum(r["hit_cap"] for r in text),
            "repeat_loop_auto(>=5 same sentence)": sum(r["max_line_repeats"] >= 5 for r in text),
            "median_output_tokens": sorted(r["output_tokens"] for r in text)[len(text) // 2] if text else None,
            "mean_helpfulness_text": mean([j["helpfulness"] for j in js]),
            "mean_fluency_text": mean([j["fluency"] for j in js]),
            "degenerate_judge": sum(bool(j["degenerate"]) for j in js),
            "categories": dict(Counter(j["category"] for j in js)),
            # whole-prompt view: a tool call counts as helpfulness 1
            "mean_helpfulness_all": mean([j["helpfulness"] for j in js] + [1] * (n - len(text))),
            "good_answers(helpfulness>=4)": good,
            "good_answer_rate_all": round(good / n, 3) if n else None,
        }
    # judge consistency on re-graded items
    reps = {}
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            j = json.loads(l)
            reps.setdefault(j["key"], []).append(j["helpfulness"])
    pairs = [v[:2] for v in reps.values() if len(v) >= 2]
    if pairs:
        summary["_judge_consistency"] = {
            "regraded": len(pairs),
            "exact_agree": round(sum(a == b for a, b in pairs) / len(pairs), 3),
            "within_1": round(sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs), 3)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["a1_base", "a2_broken", "a5_b25_fable_patched", "a7_lora"])
    ap.add_argument("--out", default="artifacts/a/answers")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--regrade", type=int, default=48, help="items judged twice, in a new batch, for consistency")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="judge at most N items (smoke test)")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    out_dir = REPO / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if not a.summarize_only:
        data = {d["id"]: d for d in map(json.loads, DATA.read_text().splitlines())}
        recs = []
        for run in a.runs:
            for pid, r in load_run(run).items():
                ans = str(r["result"])
                toks = int(r.get("output_token_count", 0))
                recs.append({"key": f"{run}|{pid}", "run": run, "id": pid, "called_tool": "<tool_call>" in ans,
                             "output_tokens": toks, "hit_cap": toks >= CAP,
                             "max_line_repeats": max_line_repeats(ans), "answer": ans})
        (out_dir / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

        jpath = out_dir / "judgements.jsonl"
        done = Counter(json.loads(l)["key"] for l in jpath.read_text().splitlines()) if jpath.exists() else Counter()
        rng = random.Random(a.seed)
        todo = [r for r in recs if not r["called_tool"]]
        rng.shuffle(todo)
        regrade = todo[: a.regrade]
        queue = [r for r in todo if done[r["key"]] < 1] + [r for r in regrade if done[r["key"]] < 2]
        if a.limit:
            queue = queue[: a.limit]
        # blind ids so the judge can't tell runs apart
        def item(r):
            d = data[r["id"]]
            q = d["question"][0][-1]["content"]
            tools = "; ".join(f'{f["name"]}: {f.get("description", "")[:120]}' for f in d["function"])
            shown = r["answer"] if len(r["answer"]) <= SHOW_CHARS else r["answer"][:SHOW_CHARS] + "\n[... cut off]"
            return {"key": r["key"], "question": q, "tools": tools, "shown": shown}
        blind = {}
        batches = []
        for i in range(0, len(queue), a.batch):
            chunk = queue[i: i + a.batch]
            items = []
            for k, r in enumerate(chunk):
                it = item(r)
                bk = f"i{i + k}"
                blind[bk] = r["key"]
                it["key"] = bk
                items.append(it)
            batches.append(items)
        print(f"{len(queue)} judgements in {len(batches)} batches with {a.model}")

        def work(b):
            return [dict(j, key=blind[j["key"]], blind_key=j["key"], judge=a.model) for j in judge(b, a.model)]

        with cf.ThreadPoolExecutor(a.workers) as ex, jpath.open("a") as f:
            futs = {ex.submit(work, b): n for n, b in enumerate(batches)}
            for n, fut in enumerate(cf.as_completed(futs), 1):
                try:
                    for j in fut.result():
                        f.write(json.dumps(j) + "\n")
                    f.flush()
                    print(f"batch {n}/{len(batches)} ok", flush=True)
                except Exception as e:  # leave it for a rerun
                    print(f"batch {n}/{len(batches)} FAILED: {e}", flush=True)

    print(json.dumps(summarize(out_dir, a.runs), indent=2))


if __name__ == "__main__":
    main()
