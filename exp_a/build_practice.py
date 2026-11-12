"""Exp A driver data + test guard (run in the bfcl-eval venv):

    /workspace/bfcl-venv/bin/python -m exp_a.build_practice

Writes (root-only dir; the harness copies only the practice files into the driver's workdir):
  /workspace/artifacts/a/practice/live_irrelevance.jsonl   BFCL live_irrelevance (practice split), as
        {"id", "messages", "tools"} with tools converted exactly as the eval handler does (_to_tools),
        so practice prompts render identically to eval prompts. No labels are included (every item in
        this category should be answered without a function call; the task prompt says so).
  /workspace/artifacts/a/practice/xlam_practice.jsonl      the 2,000 held-out xLAM examples (from A.2 prep)
  /workspace/artifacts/a/test_guard.json                   sha256 of every user message in the Exp A
        test categories; the harness refuses any forward_pass prompt containing one of them.
"""
import json
import shutil
from pathlib import Path

import bfcl_eval

from common.evals.bfcl import EXP_A_PRACTICE, EXP_A_TEST
from common.evals.bfcl_qwen35 import _to_tools
from common.harness.guard import text_hash

DATA = Path(bfcl_eval.__file__).parent / "data"
OUT = Path("/workspace/artifacts/a")


def load(cat):
    return [json.loads(l) for l in open(DATA / f"BFCL_v4_{cat}.json")]


def main():
    (OUT / "practice").mkdir(parents=True, exist_ok=True)
    assert set(EXP_A_PRACTICE).isdisjoint(EXP_A_TEST)

    def users(r):
        return [m["content"] for turn in r["question"] for m in turn if m.get("role") == "user"]

    hashes, per_cat = set(), {}
    for cat in EXP_A_TEST:
        rows = load(cat)
        k = 0
        for r in rows:
            for u in users(r):
                hashes.add(text_hash(u)); k += 1
        per_cat[cat] = {"items": len(rows), "user_messages": k}

    # Practice items whose question text also appears in a test category are excluded from the driver's
    # practice file (the text is test text even though the tool list differs).
    n, excluded = {}, {}
    for cat in EXP_A_PRACTICE:
        rows = load(cat)
        drop = [r["id"] for r in rows if any(text_hash(u) in hashes for u in users(r))]
        with open(OUT / "practice" / f"{cat}.jsonl", "w") as f:
            for r in rows:
                if r["id"] in drop:
                    continue
                f.write(json.dumps({"id": r["id"], "messages": r["question"][0],
                                    "tools": _to_tools(r["function"])}) + "\n")
        n[cat] = {"source_items": len(rows), "written": len(rows) - len(drop)}
        excluded[cat] = drop
    shutil.copy(OUT / "data/xlam_practice.jsonl", OUT / "practice/xlam_practice.jsonl")
    overlap = sum(len(v) for v in excluded.values())
    (OUT / "test_guard.json").write_text(json.dumps({"categories": EXP_A_TEST, "per_category": per_cat,
                                                     "practice_items_excluded": excluded,
                                                     "sha256": sorted(hashes)}))
    print({"practice": n, "xlam_practice": sum(1 for _ in open(OUT / "practice/xlam_practice.jsonl")),
           "guard_hashes": len(hashes), "practice_items_excluded": overlap,
           "excluded_ids": excluded})


if __name__ == "__main__":
    main()
