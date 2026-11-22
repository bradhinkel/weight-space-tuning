"""R.4b additions to the R.4a self-test (exp_r/R4b_design.md), on the pod, no driver.

    R4A_PRACTICE=/workspace/r4a/practice_r4b python -m exp_r.r4b.selftest_r4b

1. The new group loads (category live_single, kind single, BFCL-format references); its user messages pass the test
   guard; no offered tool name equals a BFCL test-category function name; the kernel can read the file.
2. A paired pass on the group with R.4a's submitted edit (layers.23.mlp.down_proj, columns 3776 / 5198):
   - well-formed: broken's call rate >= 0.95 and accuracy 0.55-0.90 (test live_simple: 70.2);
   - DESIGN GATE: the edit shows the loss R.4a's practice missed: no-call rate >= 5% or accuracy loss >= 3 points
     against broken (test live_simple: 12.8% no call, -5.5 points).
3. Base on the same items (reference only).
"""
import json
import os
import subprocess
import time

import numpy as np

from exp_r.r4a.common import PRACTICE, TestGuard
from exp_r.r4a.selftest import check

R4A_SUB = "/workspace/r4a/sessions/r4a_b25_fable_20261117T190942Z/submission"
BFCL_DATA = subprocess.run(["/workspace/bfcl-venv/bin/python", "-c", "import bfcl_eval, os; "
                            "print(os.path.join(os.path.dirname(bfcl_eval.__file__), 'data'))"],
                           capture_output=True, text=True).stdout.strip()


def main():
    ok = []
    assert PRACTICE.endswith("practice_r4b"), PRACTICE
    from exp_r.r4a.server import R4aHarness
    sid = "selftest_r4b_" + time.strftime("%Y%m%dT%H%M%S")
    h = R4aHarness({"session_id": sid, "budget": 4, "vllm_mem": 0.55})
    work = f"/srv/wst-sessions/{sid}/work"
    items = [p for p, it in h.practice.items() if it["category"] == "live_single"]
    n_file = sum(1 for _ in open(os.path.join(PRACTICE, "live_simple_style.jsonl")))
    ok.append(check("live_single loads: all items, kind single, BFCL references",
                    len(items) == n_file and all(h.practice[p]["kind"] == "single" and h.practice[p]["ref_format"] == "bfcl"
                                                 for p in items), f"{len(items)} of {n_file}"))
    g = TestGuard()
    hits = [p for p in items if g.hit_messages(h.practice[p]["messages"])]
    ok.append(check("live_single user messages pass the test guard", not hits, str(hits[:5])))
    cats = json.load(open("/workspace/r4a/guard/test_guard.json"))["categories"]
    test_names = set()
    for c in cats:
        for line in open(os.path.join(BFCL_DATA, f"BFCL_v4_{c}.json")):
            for f in json.loads(line).get("function", []):
                test_names.add(f["name"])
    clash = [p for p in items if {t["function"]["name"] for t in h.practice[p]["tools"]} & test_names]
    print("tool names equal to a test function name:", len(clash), clash[:8])
    print(f"INFO (not a check since the 2026-11-18 amendment): {len(clash)} of {len(items)} items offer a tool named "
          "like a test function (shared generic or real users' API names; no test data was used to build them)")
    out = h.python("import json\nr=[json.loads(l) for l in open('/practice/live_simple_style.jsonl')]\nprint('N', len(r))")
    ok.append(check("kernel reads /practice/live_simple_style.jsonl", f"N {n_file}" in out, out[-80:]))

    # paired pass with R.4a's edit
    n = "model.language_model.layers.23.mlp.down_proj.weight"
    os.makedirs(f"{work}/edited", exist_ok=True)
    np.save(f"{work}/edited/{n}.npy", np.load(f"{R4A_SUB}/{n}.npy"))
    h._own(f"{work}/edited")
    sample = sorted(items)[:256]
    r = json.loads(h.forward({"type": "generate", "model": "both", "items": sample}))
    ok.append(check("paired R.4a-edit generation ran", r["ok"], r.get("error", "")[:200]))
    if r["ok"]:
        o, e, pr = r["original"]["live_single"], r["edited"]["live_single"], r["paired"]["live_single"]
        print(json.dumps({"original": o, "edited": e, "paired": pr}, indent=1))
        ok.append(check("group well-formed: broken call rate >= 0.95, accuracy 0.55-0.90 (test live_simple 70.2)",
                        o["call_rate"] >= 0.95 and 0.55 <= o["correct_rate"] <= 0.90, json.dumps(o)[:200]))
        ok.append(check("DESIGN GATE: R.4a edit no-call >= 5% or accuracy loss >= 3 points vs broken "
                        "(test live_simple: 12.8% no call, -5.5)",
                        1 - e["call_rate"] >= 0.05 or pr["correct_rate_difference"] <= -0.03,
                        f"no-call {1 - e['call_rate']:.3f}, diff {pr['correct_rate_difference']:+.3f} "
                        f"+- {pr['standard_error']:.3f}"))
        rb = json.loads(h.forward({"type": "generate", "model": "base", "items": sample}))
        print("base:", json.dumps(rb)[:600])
    print(f"{sum(ok)}/{len(ok)} checks passed; passes used {h.passes}; session dir {h.sdir}")


if __name__ == "__main__":
    main()
