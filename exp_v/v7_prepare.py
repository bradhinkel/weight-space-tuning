"""V.7 setup: build V_ROOT/v7_prior/, the read-only prior/ folder of the synthesis session.

    PYTHONPATH=. ~/vision_env/.venv/bin/python -m exp_v.v7_prepare

prior/
  README.md                       what is here and how to use it
  dfr/seed<k>/fc.weight.npy, fc.bias.npy   DFR (V.1 recipe) last layers for seeds 0-4, in the model's own units
  dfr/practice_scores.json        each seed's practice scores: IN-SAMPLE (DFR is fitted on the practice images)
  dfr/recipe.md                   the DFR recipe (exp_v/v1_dfr.py docstring)
  runs/<id>/submitted/*.npy       the 18 earlier sessions' submitted tensors
  runs/<id>/report.md             each session's final answer
  runs/<id>/notes.md              each session's work/notes.md
  runs/<id>/meta.json             arm (fitting refused / allowed), budget, tensors edited
  runs/index.csv                  one row per run
No test-split result is copied into prior/: the 18 sessions never had test access (V.2 check 2), and DFR's
test scores stay in artifacts/. A 2-decimal match check is printed at the end; matches are coincidences.
"""
import csv, json, os, re, shutil
from pathlib import Path
import numpy as np, torch
from exp_v import wb
from exp_v.common import V_ROOT, SESSIONS
from exp_v.v1_dfr import dfr, __doc__ as DFR_DOC

OUT = V_ROOT / "v7_prior"
RUNS = ["v3_b25_nofit_20261122T044105Z", "v3_b25_fit_20261122T061448Z"] + sorted(
    p.name for p in SESSIONS.glob("v4_*") if "_dry" not in p.name)

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "dfr").mkdir(parents=True); (OUT / "runs").mkdir()
    m = wb.resnet50(pretrained=False)
    m.load_state_dict(torch.load("artifacts/v/broken/model.pt", map_location="cpu"))
    xv, lv, yv, gv = wb.features_and_logits(m, "val")
    ref = torch.load("artifacts/v/broken/dfr_fc_seed0.pt", map_location="cpu")
    scores = {}
    for seed in range(5):
        sc, w, b, c, tune = dfr(xv, yv, gv, seed)
        W = w / sc.scale_; B = b - (sc.mean_ / sc.scale_) @ w.T
        fc_w = np.stack([-W.ravel() / 2, W.ravel() / 2]).astype(np.float32)
        fc_b = np.array([-B[0] / 2, B[0] / 2], dtype=np.float32)
        if seed == 0:
            agree = ((xv @ fc_w.T + fc_b).argmax(1) == (xv @ ref["fc.weight"].numpy().T + ref["fc.bias"].numpy()).argmax(1)).mean()
            print("seed 0 vs V.1 DFR: practice agreement", round(float(agree), 4))  # liblinear/GPU features are not bit-exact
        d = OUT / "dfr" / f"seed{seed}"; d.mkdir()
        np.save(d / "fc.weight.npy", fc_w); np.save(d / "fc.bias.npy", fc_b)
        pred = (xv @ fc_w.T + fc_b).argmax(1)
        s = wb.score(pred, yv, gv)
        scores[f"seed{seed}"] = {"C": c, "tuning_worst_group_by_C (half/half split of practice)": tune,
                                 "practice_in_sample": s}
    (OUT / "dfr" / "practice_scores.json").write_text(json.dumps(
        {"WARNING": "IN-SAMPLE: each DFR last layer was fitted on (group-balanced subsamples of) these same 1,199 "
                    "practice images, so these scores overstate how it does on new images.", **scores}, indent=1))
    (OUT / "dfr" / "recipe.md").write_text("DFR recipe used (Kirichenko et al. 2022, DFR^Val_Tr):\n" +
                                           DFR_DOC.split("Repeated over")[0].split("Recipe, from DFR Appendix C:")[1])
    rows = []
    for k, r in enumerate(RUNS, 1):
        sd = SESSIONS / r; rid = f"run{k:02d}"; d = OUT / "runs" / rid; d.mkdir()
        cfg = json.loads((sd / "driver_summary.json").read_text())["config"]
        shutil.copytree(sd / "submitted", d / "submitted")
        (d / "report.md").write_text(json.loads((sd / "driver_summary.json").read_text())["final_text"])
        nb = sd / "sandbox" / "work" / "notes.md"
        if nb.exists():
            shutil.copy(nb, d / "notes.md")
        tens = sorted(f[:-4] for f in os.listdir(d / "submitted") if f.endswith(".npy"))
        meta = {"id": rid, "fitting": cfg["fitting"], "budget": cfg["budget"], "tensors_edited": tens}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        rows.append({"id": rid, "fitting": cfg["fitting"], "budget": cfg["budget"], "n_tensors": len(tens),
                     "tensors": " ".join(tens)})
        print(rid, r, cfg["fitting"], cfg["budget"], len(tens))
    with open(OUT / "runs" / "index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    (OUT / "README.md").write_text(README)
    # leak check: no test-split score from V.3/V.4 or DFR may appear in prior/
    test_vals = set()
    for r in RUNS:
        t = json.loads((SESSIONS / r / "final.json").read_text())
        t = t.get("test") or t.get("scores")
        test_vals |= {f"{v['acc']:.2f}" for v in t["groups"].values()} | {f"{t['worst']:.2f}", f"{t['weighted_mean']:.2f}"}
    for run in json.loads(Path("artifacts/v/broken/dfr_results.json").read_text())["dfr_runs"]:
        test_vals |= {f"{run['test']['worst']:.2f}", f"{run['test']['mean_weighted']:.2f}"}
    hits = []
    for p in OUT.rglob("*"):
        if p.suffix in (".md", ".json", ".csv"):
            txt = p.read_text(errors="replace")
            hits += [(str(p.relative_to(OUT)), v) for v in test_vals if re.search(rf"(?<![\d.]){re.escape(v)}(?!\d)", txt)]
    print("2-decimal matches with test scores:", len(hits), "(coincidences: the sessions never had test access, and "
          "practice fractions such as 128/133 = 96.24 recur; the check guards against a test file being copied in)")
    for h in hits[:40]:
        print("  ", h)

README = """PRIOR WORK (read-only). Everything here was produced before this session, without the test split.

dfr/  The current best known repair: Deep Feature Reweighting (DFR), a published method that refits only the last
      layer. It was run with 5 random seeds (recipe in dfr/recipe.md). dfr/seed<k>/fc.weight.npy and fc.bias.npy
      are drop-in replacements for the model's fc.weight [2, 2048] and fc.bias [2]: save_edit them to run DFR.
      dfr/practice_scores.json gives each seed's practice scores. They are IN-SAMPLE: DFR was fitted on these
      same practice images, so they overstate how DFR does on new images. Judge DFR on practice images it did not
      fit (for example by re-running the recipe on part of the practice set) before comparing it with anything.

runs/ 18 earlier sessions in which Fable 5.2 repaired this same model by editing its weights, under this task's
      rules. runs/index.csv lists them. 9 had fitting refused ("refused": weights only from activation statistics
      and reasoning) and 9 had fitting allowed ("allowed"); budgets were 10, 25 or 50 forward passes, 3 runs each.
      Each run folder has its submitted tensors (submitted/*.npy, the same names and shapes as the model; tensors
      not present were left as the original), its final answer (report.md) and its working notes (notes.md).
      Every practice number in them is a practice-set measurement, often in-sample for that run's own edit.
"""

if __name__ == "__main__":
    main()
