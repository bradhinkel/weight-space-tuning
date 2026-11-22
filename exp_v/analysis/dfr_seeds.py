"""DFR baseline with 20 seeds (V.1 recipe, exp_v/v1_dfr.py), current features. Test scores go to artifacts only."""
import json, warnings
import numpy as np, torch
from exp_v import wb
from exp_v.v1_dfr import dfr
warnings.filterwarnings("ignore")
m = wb.resnet50(pretrained=False); m.load_state_dict(torch.load("artifacts/v/broken/model.pt", map_location="cpu"))
xv, lv, yv, gv = wb.features_and_logits(m, "val"); xt, lt, yt, gt = wb.features_and_logits(m, "test")
runs = []
for seed in range(20):
    sc, w, b, c, tune = dfr(xv, yv, gv, seed)
    W = w / sc.scale_; B = b - (sc.mean_ / sc.scale_) @ w.T
    fw = np.stack([-W.ravel() / 2, W.ravel() / 2]); fb = np.array([-B[0] / 2, B[0] / 2])
    s = wb.score((xt @ fw.T + fb).argmax(1), yt, gt)
    runs.append({"seed": seed, "C": c, "worst": s["worst"], "mean_weighted": s["mean_weighted"]}); print(runs[-1], flush=True)
w = np.array([r["worst"] for r in runs]); mw = np.array([r["mean_weighted"] for r in runs])
summ = {"n": 20, "worst_mean": round(w.mean(), 2), "worst_sd": round(w.std(ddof=1), 2), "worst_min": w.min(), "worst_max": w.max(),
        "mean_weighted_mean": round(mw.mean(), 2), "mean_weighted_sd": round(mw.std(ddof=1), 2),
        "C_counts": {str(c): sum(r["C"] == c for r in runs) for c in sorted({r["C"] for r in runs})}}
json.dump({"runs": runs, "summary": summ}, open("artifacts/v/broken/dfr_20seeds.json", "w"), indent=1)
print(json.dumps(summ))
