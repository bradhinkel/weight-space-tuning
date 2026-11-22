"""V.4 analysis: pairwise test-prediction agreement between runs, and paired bootstrap vs DFR seed 0."""
import glob, json, os, sys
import numpy as np, torch
from exp_v import wb
from exp_v.server import BROKEN_PT

SESS = os.path.expanduser("~/wst-v/sessions")
runs = ["v3_b25_nofit_20261122T044105Z", "v3_b25_fit_20261122T061448Z"] + sorted(
    os.path.basename(p) for p in glob.glob(f"{SESS}/v4_*"))
m = wb.resnet50(pretrained=False); m.load_state_dict(torch.load(BROKEN_PT, map_location="cpu"))
F, L0, y, g = wb.features_and_logits(m, "test")
dfr = torch.load("artifacts/v/broken/dfr_fc_seed0.pt", map_location="cpu")
Ld = F @ dfr["fc.weight"].numpy().T + dfr["fc.bias"].numpy()
P = {r: np.load(f"{SESS}/{r}/test_logits.npy").argmax(1) for r in runs}
P["DFR_s0"] = Ld.argmax(1); P["ERM"] = L0.argmax(1)

def worst(pred, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    return min(((pred[idx] == y[idx])[g[idx] == k]).mean() * 100 for k in range(4))

names = list(P)
A = np.array([[(P[a] == P[b]).mean() * 100 for b in names] for a in names])
rng = np.random.default_rng(0); B = 2000
boots = [np.concatenate([rng.choice(np.where(g == k)[0], (g == k).sum()) for k in range(4)]) for _ in range(B)]
wd = np.array([worst(P["DFR_s0"], i) for i in boots])
out = {"names": names, "agreement": A.round(2).tolist(), "worst": {n: round(worst(P[n]), 2) for n in names},
       "vs_dfr": {}}
for r in runs:
    wr = np.array([worst(P[r], i) for i in boots]); d = wr - wd
    out["vs_dfr"][r] = {"diff": round(worst(P[r]) - worst(P["DFR_s0"]), 2),
                        "ci95": [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
                        "p_le0": round(float((d <= 0).mean()), 3)}
json.dump(out, open("artifacts/v/grid/v4_agreement.json", "w"), indent=1)
print(json.dumps(out["worst"])); print(json.dumps(out["vs_dfr"], indent=0))
