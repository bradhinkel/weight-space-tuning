# R.4a repeats, stage 4: the synthesis session

Month-one and provisional: an experiment, not a method. One session, n = 1.

## What was done
- **Design** (amendment 2026-11-23 in `exp_r/R4a_design.md`, written before the run). Mirrors Experiment V's V.7.
- **The session:**
  - Fable 5.1, effort max, budget 25, the same rules and evaluation as the nine repeats.
  - Task text `exp_r/r4a/sessions/r4a_s4.md`: the b25 text plus a "build on everything" paragraph and a fourth WHAT YOU HAVE item.
- **What Fable got:** a read-only `/prior` folder (`exp_r/r4a/s4_prepare.py`). It held the nine runs as `run01`–`run09` in a shuffled order. For each run:
  - its submitted tensors;
  - its final report and `notes.md`;
  - every successful forward request, with the practice summary it returned;
  - the practice ids it generated on;
  - its budget and relative change.
- **What it didn't get:** test scores, run tags, the reports and the ⚑ audit. The leak check matched 36 test values against the folder and found 4 coincidences, all log-probabilities.
- **Harness change:** an optional `prior` config key. The server links the folder to `/prior` and adds it to the kernel's Landlock read list. Checked on the pod: `/prior` is readable, the sessions folder and writes to `/prior` are refused.
- **Practice rescore** (fixed before the run, run after it; `exp_r/r4a/s4_rescore.py`):
  - all ten edits on one practice set, seed 0: 128 live_irrelevance, 64 xlam_mismatched, 56 live_relevant and 200 xlam (448 items);
  - one generation pass each, plus a judge pass on the live_irrelevance text answers;
  - practice targets: recovery ≥ 0.8 on the 192 no-call items, retention ≥ −2 (mean of the call groups) and good ≥ 22/128.
- **Ranking key**, the same on practice and on test: the number of targets met, then recovery.

## Results

### BFCL v4 test split
| run | irrelevance | mean call acc | recovery | retention (pts) | good / 240 | targets met |
|---|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 | 81 | — |
| broken | 0.83 | 84.62 | 0.000 | 0.00 | 0 | — |
| **synthesis** | **75.83** | **83.93** | **0.963** | **−0.69** | **31** | **2** (recovery, retention) |
| b10 r2, best of 9 by test | 74.17 | 82.78 | 0.941 | −1.84 | 40 | 3 |
| b25 r1 | 85.42 | 81.61 | 1.086 | −3.01 | 43 | 2 |
| b25 r2 | 71.25 | 84.05 | 0.904 | −0.57 | 31 | 2 |
| nine-run mean ± SD | 66.2 ± 12.0 | 83.4 ± 1.3 | 0.84 ± 0.15 | −1.18 ± 1.26 | 28.4 ± 10.5 | |

Per category (%):

| run | simple | multiple | parallel | par_multiple | live_simple | live_multiple | live_irrelevance |
|---|---|---|---|---|---|---|---|
| broken | 92.5 | 92.0 | 90.0 | 89.0 | 70.2 | 74.1 | 1.0 |
| synthesis | 92.2 | 91.5 | 90.5 | 88.0 | 68.2 | 73.1 | 69.6 |
| b10 r2 | 91.8 | 91.0 | 89.0 | 89.0 | 65.1 | 70.8 | 71.0 |

- **The synthesis gives the best trade-off between recovery and retention of the ten edits:**
  - only b25 r1 and b50 r1 recover more, and they lose 3.0 and 2.5 points of call accuracy;
  - only b10 r1 (+1.04), b50 r2 (−0.41) and b25 r2 (−0.57) keep more call accuracy, and they recover 0.79, 0.58 and 0.90.
  - live_simple and live_multiple lose 2.0 and 1.0 points, in the middle of the nine runs (live_simple 64.7–74.8).
- **It misses the answer-quality target:** 31 good of 240, against 40. On the ranking key it is third of ten on test, behind b10 r2 and b25 r1.

### Answer quality, test irrelevance (240 prompts)
- 182 text answers and 58 calls.
- Judge categories: answered 62, declined_only 76, declined_with_help 3, degenerate 13, other 28.
- 12 answers hit the token cap in a loop.
- Mean helpfulness 2.40 and fluency 3.79 on text answers; median 36 tokens.
- **Fable's forecast** was 30–35 good, "probably short of 40". The result was 31.
- **What's missing:** most of the shortfall is bare refusals (76), as in every earlier run. The loop fix (end-of-turn boost) cut loops but didn't make answers helpful.

### Practice rescore (448 fixed items; in-sample for most runs)
| id | run | practice recovery | practice retention | practice good /128 | practice targets | test targets |
|---|---|---|---|---|---|---|
| run07 | b10 r1 | 0.826 | −1.10 | 22 | **3** | 1 |
| — | **synthesis** | 0.973 | −1.20 | 18 | 2 | 2 |
| run08 | b25 r2 | 0.913 | −0.16 | 14 | 2 | 2 |
| run03 | b50 r1 | 0.879 | −1.52 | 17 | 2 | 1 |
| run02 | b10 r3 | 0.852 | −1.76 | 20 | 2 | 1 |
| run09 | b25 r1 | 1.000 | −3.34 | 17 | 1 | 2 |
| run01 | b10 r2 | 0.940 | −3.44 | 11 | 1 | 3 |
| run06 | b25 r3 | 0.819 | −4.32 | 19 | 1 | 1 |
| run04 | b50 r2 | 0.624 | −0.32 | 16 | 1 | 1 |
| run05 | b50 r3 | 0.779 | −3.53 | 18 | 0 | 0 |

- **The practice rescore picks b10 r1,** which meets only one target on test (retention). The synthesis ranks second on practice and third on test.
- **Agreement between practice and test,** Spearman ρ over the ten edits:
  - recovery: 0.84;
  - retention: 0.60;
  - good answers: **−0.55**.
- **The practice judge doesn't rank answer quality the way test does.** b10 r2 has the fewest practice good answers (11) but the most on test (40, tied second). Practice good counts sit in a narrow 11–22 band: 128 items, judge noise about ±3. The test differences (12–43 of 240) don't show up in it.
- **What this means for selecting runs:** on this task, practice numbers can pick runs by recovery and roughly by retention, but not by answer quality.

## The edit
- **What changed:** one tensor, `layers.23.mlp.down_proj`, with a rank-4 addition (8.2% relative change).
- **How it was built:** four read keys on the layer-23 MLP hidden state, each a closed-form ridge fit on 768 decision tokens and 2,500 teacher-forced answer positions. Each writes to a residual direction chosen for low leakage onto the rest of the vocabulary:
  1. **Decision offset:** a decision-position key times a style-balanced relevance key. It lowers `<tool_call>`, `<think>`, `<`, `</` and stray special tokens at the first answer token.
  2. **Opener nudge** on the relevance key: +2 toward "To" and +1 toward "The".
  3. **Stop route** on prose positions: +2.1 nats on end-of-turn, with no leakage onto newline tokens.
  4. **Late-call guard** on in-answer newline positions: `<tool_call>` drops by about 15 nats at paragraph starts and by 0 between two calls.
- **Where it started:** pass 1 reproduced run09's edit (b25 r1, neuron 3776) on fresh items. Fable then rebuilt the decision offset with a cleaner key. Its notes give two reasons:
  - neuron 3776 also fires at the start of argument values, so the column edit changed call arguments ("DE" → "Germany");
  - the neuron also fires at paragraph starts, which had accidentally suppressed plan-then-call answers. The new edit has an explicit guard for that instead.
- **New findings,** not in any earlier run:
  - The loops come from a weak stop: after a complete sentence, end-of-turn trails the newline by only 0.7–1.4 nats.
  - Answers that open with "To" are judged good 31–47% of the time, against about 10% for "The" and about 5% for "I".
- **What kind of build it was** (prediction S5): a new edit built on the shared diagnosis, not a combination or a re-tuning of earlier tensors.

## How the session went
- 25/25 passes and 116 Python runs in 3.32 h, with no nudges, compactions or limit waits. Two requests failed, at no charge.
- Before its first pass, Fable spent about 10 minutes reading `/prior` (README, index and all nine notes files).
- **Fable's held-out practice numbers:**
  - no-call: 74.8% live and 93.6% synthetic, against base 78.2% and 94.7%;
  - calls: net −1 item on 229 items;
  - good: 16/89 live and 8/88 synthetic.
  - The test results agree: recovery 0.963, retention −0.69, good 31.

### Rule audit
- `scripts/audit_session.py`, on the log converted to its format: 116 Python calls, 5 weak flags, 0 strong. One flag is setup code; the other four are a log_softmax over logits.
- **No ⚑.** The "exact simulation" in its report is readout simulation, which R.4a allows:
  - it takes the saved final hidden state minus its own edit;
  - adds a candidate ΔW times the saved down_proj input;
  - then applies the final RMSNorm and the tied output layer.
  - It loaded the layer-23 gate and up weights but never used them.
- **Environment gap:** scikit-learn isn't installed in the pod venv, although the task text lists it. b25 r1 and b50 r3 hit the same import error, so every session ran under the same conditions. Fable wrote its own ridge and AUC code.

## Predictions, scored (S; amendment 2026-11-23)
| # | prediction | result | scored |
|---|---|---|---|
| S1 | P(synthesis ≥ best run by practice) = 0.5 | practice picks b10 r1 (1 target on test); synthesis has 2 | ✓ happened |
| S2 | P(synthesis ≥ best run by test, b10 r2) = 0.2 | 2 targets against 3 | ✗ didn't happen |
| S3 | P(all three test targets) = 0.25 | good 31 < 40 | ✗ didn't happen |
| S4 | irrelevance 60–85, retention −3 to 0, good 20–45 | 75.8, −0.69, 31 | ✓ ✓ ✓ |
| S5a | P(main part is a layer-23 readout offset) = 0.85 | yes | ✓ |
| S5b | P(combines edits from ≥ 2 runs) = 0.4 | no | didn't happen |
| S5c | P(re-tunes one run's edit) = 0.35 | no: a new rank-4 edit | didn't happen |
| S6 | P(⚑) = 0.5 | no | didn't happen |
| S7a | ρ(practice, test recovery) > 0.5: P = 0.55 | 0.84 | ✓ |
| S7b | ρ(practice, test good) > 0.5: P = 0.5 | −0.55 | ✗ |
| S8 | 3.5–5.5 h, $35–60 notional | 3.3 h, $30.6 | ✗ (both slightly lower) |

## What it means
- **Did seeing all nine runs help?** Somewhat. The synthesis lands near the best trade-off between recovery and retention of all ten edits, at the low cost of a single b25 session. It doesn't move the ceiling on answer quality: 31 good, against the nine-run mean of 28.4 and a best of 43.
- **The prior runs' work compounded.** Fable reused their shared diagnosis (the readout offset) and spent its passes on what they had left open: collateral damage in calls, loops, and openers. Its edit is cleaner at the token level: on test it loses at most 1 point in every non-live call category.
- **Answer quality is still the bottleneck.** Every run so far, including this one, leaves most irrelevant prompts with a bare refusal. Fable's own conclusion: changes at the output layer can't fix the refusal stance, so that would need an edit further upstream, which no session has managed within budget.
- **Choosing by practice works for recovery but fails for answer quality.** Practice picked the wrong run (b10 r1) because its practice good count is noisy and doesn't predict test. Any "pick the best run" step would need a larger practice judge set, or one that matches the test set better.

## Surprises or problems
- **Idle pod:** the queued A40 came up 4 minutes after I queued it, but I didn't check again for 5 h. That's about $3 of idle GPU.
- **My own script errors:**
  - the `/prior` check script lacked a `__main__` guard, so the kernel couldn't spawn its subprocess (about 20 min lost);
  - the rescore treated `index.csv` as a run folder.
- **Judge sidecar:** one call failed with `No such file or directory: 'claude'`. A Claude Code update replaced the CLI mid-run. The rescore now resumes and retries, and no data was lost.
- **scikit-learn** is missing from the venv; see the rule audit.

## Cost
- **Driver:** $30.60 notional (Fable), 3.32 h.
- **GPU:**
  - `wst-r4a-s4` (`wk42av3vjsklg6`, A40, $0.59/h) ran 39,236 s, about $6.43. About $3 of that is the idle wait.
  - It also covered the session, the BFCL eval (12 min) and the rescore (about 1.5 h).
  - EXITED, confirmed by the stop call's response.
- **Sonnet judge:** answer scoring on test, plus 10 rescore judge passes (Max).
- **Weekly usage:** 24% before stage 4, 29% after (user, 2026-11-24): about 5 points for the synthesis session, the rescore judge passes and any Experiment V use in the same window.

## Records
- Session: `artifacts/r4a/s4/` (`calls.jsonl`, transcript, `work_final/notes.md`, submission, `audit.json`); driver log `artifacts/r4a/repeats_logs/s4_driver.log`.
- Prior folder: `artifacts/r4a/s4_prior/`; its key: `artifacts/r4a/s4_prior_key.json`.
- Test: `artifacts/pulled/artifacts/bfcl/r4a_s4_fable/`, `artifacts/r4a/score_a_s4.json`, `artifacts/a/answers_run_r4a_s4.log`.
- Rescore: `artifacts/r4a/s4_rescore/rescore.json`.

## Next
- **For the talk:** the ten-run picture is that one diagnosis (the readout offset) was found every time, and outcomes spread along a trade-off between recovery and retention. The synthesis sits at its efficient end, and answer quality is the open problem.
- **Possible follow-ups,** each needing your approval:
  - install scikit-learn, or remove it from the task text;
  - a larger practice judge set, if runs are to be selected on practice;
  - a session aimed at the refusal stance upstream of layer 23.
