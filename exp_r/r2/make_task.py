"""Build the R.2 task text from the original's message.txt and notices (exp_r/r2/original/).

    python -m exp_r.r2.make_task --budget 10 --out exp_r/r2/sessions/r2_b10.md [--mem-gb 96]

Only the tool surface changes: the original's driver replied with ```python / ```forward blocks; ours calls the
MCP tools `python` and `forward` with the same code and the same JSON. Every replacement must match exactly once.
The two notices that reached all three original runs (pass size, tool facts; their state.json) are appended,
as the original sent them during the runs. Notices 003 (submit-best) and 004 (a summary bug the R.2 harness does
not have) reached none of the three runs.
"""
import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "original")

REPL = [
    ("variables survive between your code blocks, like a notebook",
     "variables survive between your code runs, like a notebook"),
    ("To run code, reply with exactly one block that starts with ```python and nothing else; if its last line is an "
     "expression, its value is shown.",
     "To run code, call the `python` tool with one block of code; if its last line is an expression, its value is "
     "shown."),
    ("Reply with exactly one block that starts with ```forward and contains JSON:",
     "Call the `forward` tool with this JSON as its request:"),
    ("Reply with a ```forward block containing JSON:",
     "Call the `forward` tool with this JSON as its request:"),
    ("state the hypothesis of each code block in its first comment line",
     "state the hypothesis of each code run in its first comment line"),
    ("In every reply, before your code or forward block, explain in plain English",
     "In every reply, before your python or forward call, explain in plain English"),
    ("(replying without a code block does not end it earlier)",
     "(replying without a tool call does not end it earlier)"),
    ("and reply without a code block, saying what you changed",
     "and reply without calling a tool, saying what you changed"),
]


def build(budget, mem_gb=96):
    src = "message_p50c.txt"
    t = open(os.path.join(ORIG, src)).read()
    t = t.replace("a budget of 50 forward passes", f"a budget of {budget} forward passes")
    t = t.replace("until all 50 forward passes are used", f"until all {budget} forward passes are used")
    assert t.count(f"{budget} forward passes") == 2, "budget wording"
    for a, b in REPL:
        assert t.count(a) == 1, f"expected one match for: {a[:60]}"
        t = t.replace(a, b)
    if mem_gb != 96:
        assert t.count("900 seconds and 96 GB") == 1
        t = t.replace("900 seconds and 96 GB", f"900 seconds and {mem_gb} GB")
    assert "```" not in t, "a code-block instruction is left"
    notes = []
    for f in ("001_pass_size.txt", "002_tool_facts.txt"):
        n = open(os.path.join(ORIG, f)).read().strip()
        if f.startswith("002"):
            n = n.replace('Attention probabilities ("attn_layers") work now; give', 'For attention probabilities ("attn_layers"), give')
        notes.append(n)
    return t.rstrip() + "\n\nNOTES FROM THE EXPERIMENTERS.\n" + "\n".join(f"- {re.sub(r'\s+', ' ', n)}" for n in notes) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem-gb", type=int, default=96)
    a = ap.parse_args()
    open(a.out, "w").write(build(a.budget, a.mem_gb))
    print(a.out)


if __name__ == "__main__":
    main()
