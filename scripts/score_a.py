"""Exp A headline scores (A.4). Fixed definitions, computed from result files only:

  irrelevance          = BFCL test `irrelevance` accuracy (%), as the scorer reports it
  mean call accuracy   = unweighted mean of the 6 call categories (%), from summary.json
  irrelevance recovery = (repaired - broken) / (base - broken), on test irrelevance
  call retention       = repaired mean call accuracy - broken mean call accuracy (points; target >= -2)
  weights touched      = % of all parameters whose stored value differs from the run's starting checkpoint
                         (harness definition; denominator = HF model named_parameters, MTP head excluded)

Secondary, reported but not headline: strict irrelevance counts calls that failed to parse as calls
(= (correct - raw_unparsed_call) / n), with its own recovery, so a repair cannot gain score by breaking call format.

    python3 scripts/score_a.py --base A1.json --broken A2.json [--run TAG=summary.json[:touched.json] ...]
"""
import argparse
import json


def load(p):
    return json.load(open(p))


def irr(s, strict=False):
    c = s["scores"]["irrelevance"]
    if strict:
        return 100.0 * (c["correct"] - (c.get("raw_unparsed_call") or 0)) / c["n"]
    return 100.0 * c["accuracy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--broken", required=True)
    ap.add_argument("--broken-touched", help="touched json for broken vs base")
    ap.add_argument("--run", action="append", default=[], help="TAG=summary.json[:touched.json]")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    base, broken = load(a.base), load(a.broken)
    rows = [("base", base, None), ("broken", broken, load(a.broken_touched) if a.broken_touched else None)]
    for r in a.run:
        tag, rest = r.split("=", 1)
        s, _, t = rest.partition(":")
        rows.append((tag, load(s), load(t) if t else None))

    b_i, k_i = irr(base), irr(broken)
    b_s, k_s = irr(base, True), irr(broken, True)
    k_c = 100.0 * broken["mean_call_accuracy"]
    out = []
    for tag, s, t in rows:
        i, si, c = irr(s), irr(s, True), 100.0 * s["mean_call_accuracy"]
        out.append({"run": tag, "irrelevance": i, "mean_call_accuracy": c,
                    "recovery": (i - k_i) / (b_i - k_i), "retention": c - k_c,
                    "weights_touched_pct": None if tag == "base" else (t["pct_touched"] if t else None),
                    "irrelevance_unparsed": s["scores"]["irrelevance"].get("raw_unparsed_call"),
                    "strict_irrelevance": si, "strict_recovery": (si - k_s) / (b_s - k_s)})
    print("| run | test irrelevance | mean call acc | irrelevance recovery | call retention (pts) | weights touched | "
          "unparsed on irrelevance | strict irrelevance | strict recovery |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in out:
        wt = "—" if r["weights_touched_pct"] is None else f"{r['weights_touched_pct']:.2f}%"
        print(f"| {r['run']} | {r['irrelevance']:.2f} | {r['mean_call_accuracy']:.2f} | {r['recovery']:.3f} | "
              f"{r['retention']:+.2f} | {wt} | {r['irrelevance_unparsed']} | {r['strict_irrelevance']:.2f} | "
              f"{r['strict_recovery']:.3f} |")
    if a.json_out:
        json.dump({"definitions": __doc__, "base_irrelevance": b_i, "broken_irrelevance": k_i,
                   "broken_mean_call_accuracy": k_c, "rows": out}, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
