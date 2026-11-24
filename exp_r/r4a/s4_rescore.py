"""R.4a stage 4: practice rescore of the nine earlier edits and the synthesis edit on one fixed practice set
(design: exp_r/R4a_design.md, stage 4 amendment). Run on the pod after the synthesis session, with the judge
sidecar running on the laptop for session id s4_rescore:

    cd /workspace/repo && PATH=/workspace/venv/bin:$PATH HF_HUB_OFFLINE=1 python -m exp_r.r4a.s4_rescore SYNTH_SID

Items (seed 0): 128 live_irrelevance, 64 xlam_mismatched, all live_relevant, 200 xlam_practice (448). One generation
pass each for original and base, one per edit, and one judge pass per edit on its text answers to the 128
live_irrelevance items. Writes /workspace/r4a/sessions/s4_rescore/rescore.json.
"""
import json, os, random, shutil, sys, time
from pathlib import Path
from exp_r.r4a.server import R4aHarness
from exp_r.r4a.common import SESSIONS

PRIOR = Path("/workspace/r4a/s4_prior/runs")
CALL_GROUPS = ["xlam_single", "xlam_multi_same_fn", "xlam_multi_diff_fn", "live_multi_same_fn",
               "live_multi_diff_fn", "live_relevance_any_call"]
NOCALL = ["live_irrelevance", "xlam_mismatched"]


def pick(practice):
    by = {}
    for i, v in practice.items():
        by.setdefault(v["category"], []).append(i)
    rng = random.Random(0)
    out = []
    for c in sorted(by):
        ids = sorted(by[c])
        if c == "live_irrelevance":
            out += rng.sample(ids, 128)
        elif c == "xlam_mismatched":
            out += rng.sample(ids, 64)
        elif c.startswith("live_"):
            out += ids
    xl = sorted(i for c, ids in by.items() if c.startswith("xlam_") and c != "xlam_mismatched" for i in ids)
    out += rng.sample(xl, 200)
    print({c: sum(practice[i]["category"] == c for i in out) for c in by}, len(out), flush=True)
    return out


def metrics(s):
    nc = sum(s[g]["items"] * s[g]["correct_rate"] for g in NOCALL if g in s)
    n = sum(s[g]["items"] for g in NOCALL if g in s)
    calls = [s[g]["correct_rate"] for g in CALL_GROUPS if g in s]
    return nc / n, 100 * sum(calls) / len(calls)


def main():
    synth = sys.argv[1]
    h = R4aHarness({"session_id": "s4_rescore", "budget": 1000})
    items = pick(h.practice)
    li = [i for i in items if h.practice[i]["category"] == "live_irrelevance"]
    rp = Path(SESSIONS) / "s4_rescore" / "rescore.json"
    prev = json.loads(rp.read_text()) if rp.exists() else {}
    out = {"items": items}
    for m in ("original", "base"):
        r = json.loads(h.forward({"type": "generate", "model": m, "items": items}))
        out[m] = r["summary"]
    edits = {p.name: p / "submitted" for p in sorted(PRIOR.iterdir()) if p.is_dir()}
    edits["synthesis"] = Path(SESSIONS) / synth / "submission"
    o_nc, o_call = metrics(out["original"])
    b_nc, _ = metrics(out["base"])
    for name, d in edits.items():
        if name in prev and prev.get("items") == items:      # resume: keep finished edits
            out[name] = prev[name]
            continue
        ed = h.work / "edited"
        shutil.rmtree(ed)
        shutil.copytree(d, ed)
        h._own(ed)
        r = json.loads(h.forward({"type": "generate", "model": "edited", "items": items}))
        s = r["summary"]
        f = r["file"]
        rows = [json.loads(l) for l in open(h.work / f[len("/work/"):])]
        jr = [k for k, x in enumerate(rows) if x["id"] in li and not x["call_attempted"]]
        good = 0
        if jr:
            for attempt in range(4):
                j = json.loads(h.forward({"type": "judge", "answers": [{"file": f, "rows": jr}]}))
                if j.get("ok"):
                    break
                print(name, "judge failed:", j.get("error", "")[:200], flush=True)
                time.sleep(120)
            good = j["by_file"][f]["good_answers(helpfulness>=4)"]
        nc, call = metrics(s)
        rec = (nc - o_nc) / (b_nc - o_nc)
        ret = call - o_call
        met = int(rec >= 0.8) + int(ret >= -2) + int(isinstance(good, int) and good >= 22)
        out[name] = {"summary": s, "good_of_128": good, "recovery": round(rec, 3), "retention": round(ret, 2),
                     "targets_met": met}
        print(name, out[name]["recovery"], out[name]["retention"], good, met, flush=True)
        (Path(SESSIONS) / "s4_rescore" / "rescore.json").write_text(json.dumps(out, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
