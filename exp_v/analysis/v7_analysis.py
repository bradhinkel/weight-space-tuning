import glob, json, os, re, numpy as np, torch
from exp_v import wb
from exp_v.server import BROKEN_PT
SESS = os.path.expanduser("~/wst-v/sessions")
v7 = sorted(glob.glob(f"{SESS}/v7_synthesis_2*"))[-1]
runs = ["v3_b25_nofit_20261122T044105Z", "v3_b25_fit_20261122T061448Z"] + sorted(os.path.basename(p) for p in glob.glob(f"{SESS}/v4_*"))
m = wb.resnet50(pretrained=False); m.load_state_dict(torch.load(BROKEN_PT, map_location="cpu"))
F, L0, y, g = wb.features_and_logits(m, "test")
pv = np.load(f"{v7}/test_logits.npy").argmax(1)
prior = os.path.expanduser("~/wst-v/v7_prior/dfr")
D = {k: (F @ np.load(f"{prior}/seed{k}/fc.weight.npy").T + np.load(f"{prior}/seed{k}/fc.bias.npy")).argmax(1) for k in range(5)}
def acc(p): return [round(((p == y)[g == k]).mean() * 100, 2) for k in range(4)]
def worst(p, i=None):
    i = np.arange(len(y)) if i is None else i
    return min(((p[i] == y[i])[g[i] == k]).mean() * 100 for k in range(4))
out = {"v7": acc(pv), "dfr_prior_seeds": {k: acc(p) + [round(worst(p), 2)] for k, p in D.items()}}
out["agree_v7_vs_dfr"] = [round((pv == p).mean() * 100, 2) for p in D.values()]
out["agree_v7_vs_runs"] = sorted(round((pv == np.load(f"{SESS}/{r}/test_logits.npy").argmax(1)).mean() * 100, 2) for r in runs)
rng = np.random.default_rng(0)
boots = [np.concatenate([rng.choice(np.where(g == k)[0], (g == k).sum()) for k in range(4)]) for _ in range(2000)]
for k in (0,):
    d = np.array([worst(pv, i) - worst(D[k], i) for i in boots])
    out[f"vs_dfr_seed{k}"] = {"diff": round(worst(pv) - worst(D[k]), 2), "ci95": [round(float(np.percentile(d, q)), 2) for q in (2.5, 97.5)]}
# rule scan
pat = {'grad': r'\.backward\(|autograd|torch\.optim|requires_grad_?\(True', 'netsim': r'F\.conv2d|conv2d\(|batch_norm\(|F\.avg_pool|adaptive_avg_pool|max_pool'}
hits = {k: [] for k in pat}
for i, l in enumerate(open(f"{v7}/calls.jsonl")):
    s = l
    for k, p in pat.items():
        if re.search(p, s): hits[k].append(i)
out["rule_hits"] = hits
json.dump(out, open("artifacts/v/v7/v7_analysis.json", "w"), indent=1)
print(json.dumps(out, indent=0)[:3000])
