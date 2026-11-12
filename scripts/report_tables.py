"""Markdown tables for reports, generated from result files (never typed by hand).

    python3 scripts/report_tables.py bfcl artifacts/pulled/artifacts/bfcl/a1_base/summary.json [more ...]
"""
import json
import sys

CATS = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
        "live_simple", "live_multiple", "live_irrelevance"]
CALL = ["simple_python", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple"]


def bfcl(paths):
    runs = [json.load(open(p)) for p in paths]
    print("| category | " + " | ".join(r["tag"] for r in runs) + " | n |")
    print("|---|" + "---|" * len(runs) + "---|")
    for c in CATS:
        cells = [f"{100 * r['scores'][c]['accuracy']:.2f}" if r["scores"].get(c, {}).get("accuracy") is not None
                 else "—" for r in runs]
        print(f"| {c} | " + " | ".join(cells) + f" | {runs[0]['scores'][c].get('n')} |")
    print("| **mean call accuracy** (6 call categories) | " +
          " | ".join(f"**{100 * r['mean_call_accuracy']:.2f}**" for r in runs) + " | |")
    print()
    print("| category | " + " | ".join(f"{r['tag']} raw responses with `<tool_call>` / unparsed" for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for c in CATS:
        print(f"| {c} | " + " | ".join(
            f"{r['scores'][c].get('raw_attempted_call')}/{r['scores'][c].get('raw_n')} / "
            f"{r['scores'][c].get('raw_unparsed_call')}" for r in runs) + " |")
    print()
    print("wall-clock (s): " + ", ".join(f"{r['tag']}={r['seconds']}" for r in runs))


if __name__ == "__main__":
    {"bfcl": bfcl}[sys.argv[1]](sys.argv[2:])
