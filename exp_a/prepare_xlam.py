"""A.2 data prep: split xLAM 60k into train (58k) and the 2,000-example practice split, and convert
to chat messages + OpenAI-style tools so the Qwen3.5 chat template renders native XML tool calls.

Outputs (JSONL, one example per line: {"id", "messages", "tools"}):
  /workspace/artifacts/a/data/xlam_train.jsonl
  /workspace/artifacts/a/data/xlam_practice.jsonl   <- driver-visible practice split (A.5)
  /workspace/artifacts/a/data/split.json            <- ids, seed, counts, source revision
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

REV = "26d14ebfe18b1f7b524bd39b404b50af5dc97866"
OUT = Path("/workspace/artifacts/a/data")
SEED, N_PRACTICE = 0, 2000

_TYPE = [(r"^(list|tuple|set)", "array"), (r"^(dict|object|mapping)", "object"), (r"^int", "integer"),
         (r"^(float|number|double)", "number"), (r"^bool", "boolean"), (r"^str", "string")]


def json_type(t: str) -> str:
    t = (t or "").strip().lower()
    for pat, js in _TYPE:
        if re.match(pat, t):
            return js
    return "string"


def convert_tool(tool: dict) -> dict:
    props, required = {}, []
    for name, spec in (tool.get("parameters") or {}).items():
        raw_t = spec.get("type", "str")
        p = {"type": json_type(raw_t), "description": spec.get("description", "")}
        if "default" in spec and spec["default"] not in (None, ""):
            p["default"] = spec["default"]
        props[name] = p
        if "optional" not in raw_t.lower() and spec.get("default") in (None, ""):
            required.append(name)
    return {"type": "function", "function": {"name": tool["name"], "description": tool.get("description", ""),
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


def convert(row) -> dict:
    tools = [convert_tool(t) for t in json.loads(row["tools"])]
    answers = json.loads(row["answers"])
    calls = [{"type": "function", "function": {"name": a["name"], "arguments": a["arguments"]}} for a in answers]
    return {"id": int(row["id"]), "tools": tools,
            "messages": [{"role": "user", "content": row["query"]},
                         {"role": "assistant", "content": "", "tool_calls": calls}]}


def main():
    ds = load_dataset("Salesforce/xlam-function-calling-60k", revision=REV, split="train")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    practice, train = set(idx[:N_PRACTICE]), idx[N_PRACTICE:]
    OUT.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with open(OUT / "xlam_train.jsonl", "w") as ft, open(OUT / "xlam_practice.jsonl", "w") as fp:
        for i in range(len(ds)):
            ex = convert(ds[i])
            n_empty += not ex["messages"][1]["tool_calls"]
            (fp if i in practice else ft).write(json.dumps(ex) + "\n")
    meta = {"source": "Salesforce/xlam-function-calling-60k", "revision": REV, "seed": SEED, "n_total": len(ds),
            "n_train": len(train), "n_practice": N_PRACTICE, "n_without_calls": n_empty,
            "practice_ids": sorted(int(ds[i]["id"]) for i in practice)}
    (OUT / "split.json").write_text(json.dumps(meta))
    print({k: v for k, v in meta.items() if k != "practice_ids"})


if __name__ == "__main__":
    main()
