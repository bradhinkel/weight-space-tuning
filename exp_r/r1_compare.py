"""R.1: compare our per-problem results with the original's published ones (exp8_math/eval/<model>/parts).

    python -m exp_r.r1_compare <ours_root> <theirs_root> [--sets math500,gsm8k]

For each set: overall accuracy and cut share for both, the per-problem paired difference (ours − theirs, mean over
samples per problem) with its SE over problems, the correlation of per-problem accuracy, and agreement on whether a
problem is solved by at least one sample. Problems are matched by id; only ids present on both sides count.
"""
import argparse
import glob
import json
import math
import os


def load(root, name):
    per = {}
    for p in glob.glob(os.path.join(root, "**", f"{name}_per_problem.json"), recursive=True):
        for r in json.load(open(p)):
            per[r["id"]] = r["correct"]
    return per


def cut_share(root, name):
    cut = tot = 0
    for p in glob.glob(os.path.join(root, "**", "scores.json"), recursive=True):
        v = json.load(open(p)).get(name)
        if v and "cut" in v:
            cut += v["cut"]
            tot += v["total"]
        elif v and "cut_share" in v:
            n = v.get("total") or v.get("n_samples") or 0
            cut += v["cut_share"] * n
            tot += n
    return cut / tot if tot else None


def compare(ours, theirs, name):
    a, b = load(ours, name), load(theirs, name)
    ids = sorted(set(a) & set(b))
    pa = [sum(a[i]) / len(a[i]) for i in ids]
    pb = [sum(b[i]) / len(b[i]) for i in ids]
    d = [x - y for x, y in zip(pa, pb)]
    n = len(d)
    md = sum(d) / n
    se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1) / n)
    ma, mb = sum(pa) / n, sum(pb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(pa, pb))
    va, vb = sum((x - ma) ** 2 for x in pa), sum((y - mb) ** 2 for y in pb)
    r = cov / math.sqrt(va * vb) if va and vb else float("nan")
    sa, sb = [any(a[i]) for i in ids], [any(b[i]) for i in ids]
    return {"problems_matched": n, "only_ours": len(set(a) - set(b)), "only_theirs": len(set(b) - set(a)),
            "acc_ours": round(100 * ma, 2), "acc_theirs": round(100 * mb, 2),
            "diff_pts": round(100 * md, 2), "se_pts": round(100 * se, 2),
            "per_problem_r": round(r, 3),
            "solved_any_ours": sum(sa), "solved_any_theirs": sum(sb),
            "solved_any_both": sum(x and y for x, y in zip(sa, sb)),
            "cut_ours": cut_share(ours, name), "cut_theirs": cut_share(theirs, name)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("theirs")
    ap.add_argument("--sets", default="math500,gsm8k")
    a = ap.parse_args()
    out = {s: compare(a.ours, a.theirs, s) for s in a.sets.split(",")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
