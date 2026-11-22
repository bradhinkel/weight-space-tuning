# R.4a repeats: handoff brief for a second Claude Code session

Written 2026-11-21 in the Experiment V session, for a new session that runs the R.4a repeats in parallel with V.
Read `reports/R.4a.md` (the one b25 run) and `exp_r/R4a_design.md` first.

## Goal
Measure the variation between sessions on R.4a's task, and how results depend on budget, then let Fable synthesize a
best edit from all runs. The talk's numbers are currently from one run (b25: irrelevance 85.4, retention −3.01,
recovery 1.086, 43 good answers).

## The experiment: replication, one change at a time
- **Unchanged from R.4a:**
  - the harness (`exp_r/r4a/`), the rules, the refusals and the task text (`exp_r/r4a/sessions/r4a_b25.md`, with
    only the budget number changed for b10 and b50);
  - the driver model: **Fable 5.1**, effort max;
  - the practice data (A.11c's practice_v2: **no** `live_simple_style.jsonl`, so do not set `practice_dir` /
    `R4A_PRACTICE`);
  - the eval.
- **This way the existing b25 run counts as one of the three b25 runs.**
- **Each session starts fresh:** no notes from earlier runs, no starting edit.
- **Not in this round:** the fixes R.4a's report suggested (the live_simple practice group, a loop check). They are
  R.4b (`exp_r/R4b_design.md`), a separate experiment.
- **The uncommitted changes in `exp_r/r4a/{common,driver,server}.py` are R.4b's additions.** They do nothing unless
  `practice_dir` is set or `live_simple_style.jsonl` is present. Check that this holds before the first run: run the
  self-test, and diff the pass summaries' group keys against R.4a's.

## Grid and order (staged; decide after each stage)
| stage | sessions | estimate each | Max weekly points (estimate) |
|---|---|---|---|
| 1 | b25 × 2 more (`r4a_b25_fable_r2`, `_r3`) | ~3–4 h | ~6–7 |
| 2 | b10 × 3 | ~1.5–2 h | ~3 |
| 3 | b50 × 3, **only if the user approves it after stages 1–2** | ~7–10 h | ~12 |
| 4 | synthesis (below), after the user approves it | ~3–4 h | ~6 |
| each | eval: BFCL plus the answer judge | ~0.5–1 h | ~1 |

- **The estimates come from the one b25 run.** That run used 3.3 h and $30.3 notional, and about 10 weekly points
  for all of R.4a.
- **Record the Max usage before and after each stage:** ask the user, and pass `--usage-before`.
- **The weekly allowance was reset on 2026-11-21 (0%).** The user wants to use it this week. Experiment V runs at the
  same time and draws from the same allowance.

## Before any run: write predictions
Add to `exp_r/R4a_design.md` as an amendment, before stage 1:
- the spread of the three b25 runs on the four test quantities;
- whether the repeats find the same mechanism: a readout counter-offset in layer 23, around neuron 3776;
- the curve of results against budget.

## Running
- **Pods:**
  - A40 `wst-r4a-dev` (`kym5l03a0t1yf8`), stopped; restart it.
  - H100 `wst-r4a` (`4jwymwk1l6z53e`), stopped.
  - Volume `61gtyoubem`, which holds `/workspace/repo`, the venv, the models and `/workspace/r4a/sessions/`.
  - These are the only RunPod resources this work may change. **Never touch `a0-1000-pilot` or `esc-c`
    (`lhyj9s7ex775et`):** they belong to other projects.
  - Stop pods when idle, and verify with get-pod.
- **One session per GPU** (vLLM takes 55% of the GPU). Sessions can run in parallel on separate pods:
  - give each driver its own `WST_POD_HOST` and `WST_POD_PORT`;
  - check first whether the network volume can be attached to two pods at once in that data center;
  - session directories are per session id, so they don't collide.
- **Setup:** sync the repo to `/workspace/repo`, then run `exp_r/r4a/setup_pod.sh` (idempotent). Run the self-test
  (`exp_r/r4a/selftest.py`) once per pod.
- **Driver:** `python -m exp_r.r4a.driver --config exp_r/r4a/sessions/<cfg>.json --usage-before "<x%>"`.
  - Copy `r4a_b25_fable.json` into new configs, changing only `tag` and `budget`. Set `task_file` to a copy of the
    task text with the budget changed.
  - The judge sidecar runs on the laptop through the `claude` CLI (Sonnet 5.5), and counts against Max.
- **Eval, as in `reports/R.4a.md`:**
  - `common.evals.bfcl` on the submission checkpoint. Put `/workspace/venv/bin` on PATH, or it fails on a missing
    `ninja`.
  - `scripts/score_a.py`.
  - `scripts/score_answers.py` on test irrelevance, with the cached judgements. Copy
    `result/BFCL_v4_irrelevance_result.json` one level up first.
- **Rule audit per session:** `scripts/audit_session.py`, then read the flagged calls in full. Readout simulation
  from saved residuals is allowed, as in R.4a.

## Synthesis session (stage 4), mirrors Experiment V's V.7
- **Fable gets:** every run's `notes.md`, submitted edit and **practice** pass summaries. It never sees test scores:
  test results stay with the experimenters.
- **Its task:** compare the runs, build one best edit (combine or refine), and confirm it on held-out practice
  passes. Budget about 25.
- **Before it runs:** write the design as an amendment, with predictions:
  - P(synthesis ≥ the best single run chosen by practice score);
  - P(synthesis ≥ the best single run chosen by test score).

## Reports and commits
- **Reports:** one report per stage (`reports/R.4a_repeats_s1.md`, …), with tables in the style of R.4a: per run,
  plus the mean ± SD per budget.
- **Commits:**
  - Commit only `exp_r/` and its own reports, and only when the user asks.
  - The Experiment V session commits `exp_v/` and `reports/V.*`.
  - Don't commit the other session's files. The uncommitted R.4b files and `exp_r/r4a/demo_web.py` belong to the
    user's earlier work: ask before committing them.
- **Privacy:** never send the user's email or personal details to external services, and never print OAuth tokens.
- **Framing for the talk:** month-one and provisional, "an experiment, not a method".
