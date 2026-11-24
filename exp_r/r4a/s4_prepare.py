"""R.4a stage 4 (synthesis) setup: build artifacts/r4a/s4_prior/, the read-only prior/ folder of the synthesis
session, from the nine earlier sessions' local records (b10 r1-r3, b25 r1-r3, b50 r1-r3).

    python3 -m exp_r.r4a.s4_prepare

prior/
  README.md                      what is here and how to use it
  runs/index.csv                 one row per run: id, budget, tensors edited, relative change
  runs/runNN/submitted/*.npy     the tensors that run submitted (all others were left as the fine-tuned model)
  runs/runNN/report.md           its final answer
  runs/runNN/notes.md            its /work/notes.md at the end
  runs/runNN/passes.jsonl        every successful forward pass: {"pass", "request", "summary"} (practice only)
  runs/runNN/items_used.json     the practice ids it ran a generation pass on (to pick items it never saw)
  runs/runNN/meta.json           budget, tensors, relative change per tensor
Run ids are a fixed shuffle (seed 4), so the order does not reveal anything. No test result is copied: the
sessions never had test access, and the BFCL scores stay in artifacts/. A 2-decimal match check against every
test number of the nine runs is printed at the end; matches are coincidences unless a test file was copied in.
"""
import csv, json, random, re, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
A = REPO / "artifacts/r4a"
OUT = A / "s4_prior"
RUNS = ["b10_r1", "b10_r2", "b10_r3", "b25", "b25_r2", "b25_r3", "b50_r1", "b50_r2", "b50_r3"]


def final_text(sd, sid):
    for p in (sd / "driver_summary.json", REPO / "artifacts/sessions" / sid / "driver_summary.json"):
        if p.exists():
            return json.loads(p.read_text())["final_text"]
    raise FileNotFoundError(sid)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "runs").mkdir(parents=True)
    order = RUNS[:]
    random.Random(4).shuffle(order)
    rows, key = [], {}
    for k, r in enumerate(order, 1):
        sd = A / r
        rid = f"run{k:02d}"
        d = OUT / "runs" / rid
        d.mkdir()
        cfg = json.loads((sd / "config.json").read_text())
        key[rid] = {"run": r, "session_id": cfg["session_id"]}
        shutil.copytree(sd / "submission", d / "submitted")
        (d / "report.md").write_text(final_text(sd, cfg["session_id"]))
        shutil.copy(sd / "work_final" / "notes.md", d / "notes.md")
        passes, used, n = [], set(), 0
        for line in open(sd / "calls.jsonl"):
            e = json.loads(line)
            if e["kind"] != "forward" or not e["result"].get("ok", True):
                continue
            n += 1
            passes.append({"pass": n, "request": e["request"], "summary": e["result"]})
            if e["request"].get("type") == "generate":
                used |= set(e["request"].get("items") or [])
        with open(d / "passes.jsonl", "w") as f:
            for p in passes:
                f.write(json.dumps(p) + "\n")
        (d / "items_used.json").write_text(json.dumps(sorted(used)))
        rel = json.loads((sd / "submission_ckpt" / "edits.json").read_text())["relative_change"]
        meta = {"id": rid, "budget": cfg["budget"], "tensors_edited": sorted(rel),
                "relative_change": {t: round(v, 4) for t, v in rel.items()}, "forward_passes_logged": n}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        rows.append({"id": rid, "budget": cfg["budget"], "n_tensors": len(rel),
                     "tensors": " ".join(t.replace("model.language_model.", "") for t in sorted(rel)),
                     "max_relative_change": round(max(rel.values()), 4), "items_used": len(used)})
        print(rid, r, cfg["budget"], len(rel), n, len(used))
    with open(OUT / "runs" / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "README.md").write_text(README)
    (A / "s4_prior_key.json").write_text(json.dumps(key, indent=1))   # experimenters only; not in prior/
    # leak check: no test number of the nine runs may appear in prior/
    vals = set()
    for f in ("score_a.json", "score_a_b10.json", "score_a_repeats.json", "score_a_b50.json"):
        for row in json.loads((A / f).read_text())["rows"]:
            if row["run"].startswith("r4a_"):
                for c in ("irrelevance", "mean_call_accuracy", "retention"):
                    vals.add(f"{row[c]:.2f}")
                vals.add(f"{row['recovery']:.3f}")
    for p in (REPO / "artifacts/pulled/artifacts/bfcl").glob("r4a_*/summary.json"):
        s = json.loads(p.read_text())
        for v in json.dumps(s).replace(",", " ").replace(":", " ").split():
            try:
                x = float(v.strip("}{[]\""))
            except ValueError:
                continue
            if 1 < abs(x) < 100 and x != int(x):
                vals.add(f"{x:.2f}")
    hits = []
    for p in OUT.rglob("*"):
        if p.suffix in (".md", ".json", ".csv", ".jsonl"):
            t = p.read_text(errors="replace")
            hits += [(str(p.relative_to(OUT)), v) for v in vals if re.search(rf"(?<![\d.]){re.escape(v)}(?!\d)", t)]
    print(f"test values checked: {len(vals)}; 2-decimal matches in prior/: {len(hits)}")
    for h in hits[:40]:
        print("  ", h)


README = """PRIOR WORK (read-only). Everything here was produced before this session, without access to the evaluation.

runs/ 9 earlier sessions in which Fable 5.1 repaired this same model by editing its weights, under this task's
      rules, with the same practice data and the same forward-pass tool. Budgets were 10, 25 or 50 forward passes,
      3 sessions each. runs/index.csv lists them (run ids are in a random order). Each run folder has:
        submitted/*.npy    the tensors it submitted (same names and shapes as the model; every tensor not present
                           was left as the fine-tuned model). To try a run's edit: drop_edit everything, then
                           save_edit each of its tensors (np.load the file, torch.from_numpy).
        report.md          its final answer: what it changed, its evidence, why it expected it to help.
        notes.md           its working notes at the end of the session.
        passes.jsonl       every successful forward pass it made: the request and the summary it got back.
                           Files named in those summaries (/work/gen_<k>.jsonl, /work/judge_<k>.jsonl, ...) were
                           that session's own and are not available here.
        items_used.json    the practice ids it ran generation passes on. An edit chosen on some items looks
                           better on those items than it is: to compare runs fairly, use items none of the
                           compared runs used, or at least items the run being judged did not use.
        meta.json          budget, tensors edited, relative change of each edited tensor.
      Every practice number in the reports and notes is that session's own measurement, usually on items it also
      used to choose its edit (in-sample), and different sessions measured on different items. They are not
      directly comparable with each other.
"""

if __name__ == "__main__":
    main()
