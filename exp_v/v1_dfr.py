"""V.1 step 2: DFR (Kirichenko et al. 2022, DFR^Val_Tr) on top of the broken model.

Recipe, from DFR Appendix C:
  - features = input to the last layer (2048-d), from the broken model, eval transform;
  - StandardScaler fitted on the reweighting data;
  - l1 logistic regression (sklearn, liblinear);
  - tune C in {1, .7, .3, .1, .07, .03, .01} by worst-group accuracy: fit on a balanced subsample of
    one half of validation, score on the other half;
  - final: 10 fits on group-balanced subsamples of all validation (all of the smallest group, the same
    number drawn from each other group), weights averaged; the scaler is fitted on all validation.
Repeated over 5 seeds (the paper reports mean ± std over 5 runs).

    ~/vision_env/.venv/bin/python -m exp_v.v1_dfr --model artifacts/v/broken
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from exp_v import wb

CS = [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]


def balanced(g, rng):
    n = np.bincount(g, minlength=4).min()
    return np.concatenate([rng.choice(np.where(g == k)[0], n, replace=False) for k in range(4)])


def fit(x, y, c):
    return LogisticRegression(penalty="l1", C=c, solver="liblinear").fit(x, y)


def wga(pred, y, g):
    return min((pred[g == k] == y[g == k]).mean() for k in range(4))


def dfr(xv, yv, gv, seed):
    rng = np.random.default_rng(seed)
    # tuning: half / half
    idx = rng.permutation(len(yv)); a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    sc = StandardScaler().fit(xv[a])
    tune = {}
    for c in CS:
        s = balanced(gv[a], rng)
        m = fit(sc.transform(xv[a][s]), yv[a][s], c)
        tune[c] = float(wga(m.predict(sc.transform(xv[b])), yv[b], gv[b]))
    c = max(tune, key=tune.get)
    # final: 10 balanced refits on all validation, averaged
    sc = StandardScaler().fit(xv)
    coefs, ints = [], []
    for _ in range(10):
        s = balanced(gv, rng)
        m = fit(sc.transform(xv[s]), yv[s], c)
        coefs.append(m.coef_), ints.append(m.intercept_)
    return sc, np.mean(coefs, 0), np.mean(ints, 0), c, tune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/v/broken")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    d = Path(a.model)
    model = wb.resnet50(pretrained=False)
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
    xv, lv, yv, gv = wb.features_and_logits(model, "val")
    xt, lt, yt, gt = wb.features_and_logits(model, "test")
    runs = []
    for seed in range(a.seeds):
        sc, w, b, c, tune = dfr(xv, yv, gv, seed)
        pred = ((sc.transform(xt) @ w.T + b).ravel() > 0).astype(int)
        r = {"seed": seed, "C": c, "tune_wga": tune, "test": wb.score(pred, yt, gt)}
        runs.append(r)
        print(json.dumps({"seed": seed, "C": c, **{k: r["test"][k] for k in ("worst", "mean_weighted", "mean_plain")}}), flush=True)
        if seed == 0:  # keep one DFR last layer in the model's own units (fc on raw features), for V.2
            W = w / sc.scale_; B = b - (sc.mean_ / sc.scale_) @ w.T
            # binary logistic -> 2-class fc: logit(waterbird) - logit(landbird) = W x + B
            fc_w = np.stack([-W.ravel() / 2, W.ravel() / 2]); fc_b = np.array([-B[0] / 2, B[0] / 2])
            torch.save({"fc.weight": torch.tensor(fc_w, dtype=torch.float32), "fc.bias": torch.tensor(fc_b, dtype=torch.float32)}, d / "dfr_fc_seed0.pt")
    worst = np.array([r["test"]["worst"] for r in runs]); mean = np.array([r["test"]["mean_weighted"] for r in runs])
    res = {"erm_test": wb.score(lt.argmax(1), yt, gt), "erm_val": wb.score(lv.argmax(1), yv, gv),
           "dfr_runs": runs,
           "dfr_summary": {"worst_mean": round(worst.mean(), 2), "worst_std": round(worst.std(ddof=1), 2),
                           "mean_weighted_mean": round(mean.mean(), 2), "mean_weighted_std": round(mean.std(ddof=1), 2)}}
    (d / "dfr_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"erm_test": res["erm_test"], "dfr_summary": res["dfr_summary"]}, indent=1))


if __name__ == "__main__":
    main()
