# R.4a repeats, stage 3: three b50 sessions

## What was done
- **Three b50 sessions** (`r4a_b50_fable_r1`, `_r2`, `_r3`), all on A40s in CA-MTL-1 on volume `61gtyoubem`:

  | run | pod | when |
  |---|---|---|
  | r1 | `wst-r4a-b50` (`o0q38sx7k286h7`) | 2026-11-23, 03:00–07:22Z |
  | r2 | `wst-r4a-b50b` (`hk7cyi4scldxro`) | 14:23–18:36Z, in parallel with r3 |
  | r3 | `wst-r4a-b50c` (`cnobba7a0um51y`) | 14:25–19:56Z |

  The user first asked for one run, to judge time and cost, and then for the other two ("This completes the original plan").
- **Only the pass budget changed.** The configs are copies of `r4a_b25_fable.json` with `tag`, `budget: 50` and `task_file` changed. The task text, `exp_r/r4a/sessions/r4a_b50.md`, differs from `r4a_b25.md` only in its two "25 forward passes" lines.
- **Everything else matched R.4a and stages 1–2:** harness, rules, Fable 5.1 at effort max, practice_v2, and the eval (BFCL v4 test, then `score_a.py`, then `score_answers.py`).
- **Self-tests:** 34/34 for all three, and the A.11c gate reproduced. r2 and r3 passed first time. r1 took three tries because of bugs in my scratch wrapper script, not in the experiment:
  - the self-test's stdout was block-buffered, so the "start the sidecar" line reached the log only after the pod had already timed out the judge;
  - the wrapper killed the sidecar after 20 min, against the pod's 45 min;
  - an SSH launch call hung;
  - the first driver launch left off the `WST_POD_*` variables (no passes or Fable used; log `artifacts/r4a/repeats_logs/b50_r1_driver_noenv_fail.log`).

  All four are fixed in the scratch wrapper.
- **No Fable limit, error waits, nudges or compactions in any run.** The lock-down checks passed in all three.
- **Deviation: the network volume filled up (200 GB) as r3 finished.**
  - `finalize()` saved r3's edited tensor to `submission/`, but failed while writing `submission_ckpt`. The eval launch then failed with "Disk quota exceeded".
  - With the user's approval the volume was grown to 250 GB.
  - I then reran the server's own `finalize()` on the intact `/srv/wst-sessions/<sid>/work`. It reported one tensor, none refused, relative change 0.114, and r3 was evaluated from that checkpoint.
  - r1 and r2 had finished their checkpoints and evals before the disk filled.
  - This cost about 2 h of idle pod time.

## Results

### BFCL v4 test split
| run | irrelevance | mean call acc | recovery | retention (pts) | good answers / 240 | targets met |
|---|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 | 81 | |
| broken | 0.83 | 84.62 | 0.000 | 0 | 0 | |
| corrective LoRA (A.7, trained) | 82.50 | 82.73 | 1.048 | −1.89 | 75 | 3/3 |
| **b50 r1** ⚑ (light) | **77.92** | **82.14** | **0.989** | **−2.49** | **37** | 1/3 |
| **b50 r2** ⚑ | **45.83** | **84.21** | **0.578** | **−0.41** | **17** | 1/3 |
| **b50 r3** ⚑ (light) | **60.83** | **82.56** | **0.770** | **−2.06** | **29** | 0/3 |
| **b50 mean ± SD (n = 3)** | **61.5 ± 16.1** | **83.0 ± 1.1** | **0.78 ± 0.21** | **−1.65 ± 1.10** | **27.7 ± 10.1** | |
| b25 mean ± SD (n = 3) | 72.4 ± 12.5 | 83.2 ± 1.4 | 0.92 ± 0.16 | −1.43 ± 1.37 | 32.7 ± 9.6 | |
| b10 mean ± SD (n = 3) | 64.9 ± 8.3 | 84.2 ± 1.4 | 0.82 ± 0.11 | −0.45 ± 1.44 | 25.0 ± 14.1 | |
| all nine runs | 66.3 ± 12.0 | 83.5 ± 1.3 | 0.84 ± 0.15 | −1.18 ± 1.26 | 28.4 ± 10.5 | |

⚑ = recomputed MLP sublayers (RMSNorm, then gate/up, then SiLU) from saved residuals in the kernel, as b10 r2 and r3 did. ⚑ (light) = applied RMSNorm and the LoRA's linear maps to saved activations, one layer at a time, with no nonlinearity. See the Rule audit. All are kept in the tables, on the user's ruling.

- **Targets:** recovery ≥ 0.80, retention ≥ −2, good answers ≥ 40.
  - No b50 run meets all three. b10 r2 is still the only one of the nine that does.
  - r3 misses all three narrowly: recovery 0.77, retention −2.06, 29 good answers.
- **b50 is not better than b25 or b10.**
  - Its mean irrelevance (61.5) is the lowest of the three budgets, and its SD (16.1) the largest. Welch t for b50 − b25 on irrelevance is −0.9.
  - The three budget means lie within 11 points of each other on irrelevance and 8 on good answers, while each budget's SD is 8–16 and 10–14.
  - At n = 3, budget has no detectable effect on any test quantity.
- **The trade-off holds within b50.** r1 is the most aggressive (high irrelevance, retention −2.49). r2 is the most cautious (irrelevance 45.8, retention −0.41). r3 is in between.

**Per category** (accuracy; calls made / prompts):

| category | broken | **b50 r1** | **b50 r2** | **b50 r3** | b50 mean ± SD |
|---|---|---|---|---|---|
| simple_python | 92.5 | 91.5 (400/400) | 92.5 (400/400) | 91.5 (400/400) | 91.8 ± 0.6 |
| multiple | 92.0 | 91.0 (199/200) | 91.5 (199/200) | 90.0 (199/200) | 90.8 ± 0.8 |
| parallel | 90.0 | 87.5 (198/200) | 90.0 (200/200) | 87.5 (197/200) | 88.3 ± 1.4 |
| parallel_multiple | 89.0 | 85.5 (199/200) | 88.0 (200/200) | 86.0 (200/200) | 86.5 ± 1.3 |
| live_simple | 70.2 | 66.3 (243/258) | 69.8 (257/258) | 67.4 (246/258) | 67.8 ± 1.8 |
| live_multiple | 74.1 | 71.0 (1031/1053) | 73.5 (1047/1053) | 72.9 (1029/1053) | 72.5 ± 1.3 |
| irrelevance | 0.8 | 77.9 (called 53/240) | 45.8 (called 130/240) | 60.8 (called 94/240) | 61.5 ± 16.1 |
| live_irrelevance | 1.0 | 62.3 (called 333/884) | 44.2 (called 497/884) | 59.6 (called 362/884) | 55.4 ± 9.8 |

- **r2 keeps every call category within 0.5 points of broken, except live_multiple at −0.6.** The other two lose 1–4 points, mostly in parallel_multiple and live_simple, where Fable's practice data has least coverage.

### Answer quality, BFCL test irrelevance (240 prompts)
| | base | corrective LoRA | **b50 r1** | **b50 r2** | **b50 r3** | b50 mean ± SD | b25 mean | b10 mean |
|---|---|---|---|---|---|---|---|---|
| text answers | 189 | 198 | 187 | 110 | 146 | 148 ± 39 | 174 | 156 |
| good (helpfulness ≥ 4) | 81 | 75 | 37 | 17 | 29 | 27.7 ± 10.1 | 32.7 | 25.0 |
| mean helpfulness, text | 3.16 | 2.98 | 2.55 | 2.41 | 2.50 | 2.49 ± 0.07 | 2.39 | 2.27 |
| mean fluency, text | 4.12 | 4.04 | 3.67 | 3.95 | 3.92 | 3.85 ± 0.15 | 3.61 | 3.46 |
| answered / declined_with_help / declined_only | 143 / 26 / 6 | 140 / 27 / 13 | 64 / 7 / 76 | 26 / 7 / 74 | 46 / 17 / 59 | | | |
| degenerate (judge category) | 6 | 11 | 26 | 3 | 5 | 11 ± 13 | 28 | 29 |
| hit token cap / same sentence ≥ 5× | | | 25 / 22 | 3 / 3 | 9 / 6 | | | |

- **The b50 text answers are the cleanest of any budget.** Mean fluency is 3.85, against 3.61 at b25 and 3.46 at b10, and r2 and r3 have only 3–5 degenerate answers. But most of them decline: 59–76 bare declines per run.
- **r3's steering term** read the fine-tune's "I can only call functions" style from paired text answers and subtracted it. It gave the most declined_with_help answers of any R.4a run (17), but not more good answers than r1.
- **Judge consistency:** 406 items regraded across all runs: 79.3% exact, 99.3% within 1 point. Read each count as roughly ±3.

## The edits
| | b50 r1 ⚑ (light) | b50 r2 ⚑ | b50 r3 ⚑ (light) |
|---|---|---|---|
| tensors | 1 | 2 | 1 |
| where | `layers.23.mlp.down_proj`, the column of neuron 3776 | `layers.23.mlp.down_proj` (45 columns) + `layers.23.mlp.gate_proj` (the same 45 rows) | `layers.23.mlp.down_proj`, rank-2 |
| read side | neuron 3776, which fires at the decision token (14.7 ± 2.1) and more strongly on irrelevant prompts | 45 neurons that are silent on answers and relevant prompts and active on irrelevant ones (held-out AUC 0.98). Gate rows edited so they stay silent outside the decision position (activity 2.0 → 0.06) | two ridge-fitted read directions: one ≈1 only at the decision position (leakage < 7%); one ≈1 inside text answers |
| write side | −5.6 logits on `<tool_call>` against a mix of answer openers | −1.6 × the mean fine-tuned − base final-residual shift (norm 26.7), weighted per neuron | (1) −5.5 × the `<tool_call>` − text-opener direction; (2) minus the text-specific residual shift between edited and base text answers |
| relative change | 0.012 | down_proj 2.12, gate_proj 0.12 (see below) | 0.114 |
| Fable's forecast, irrelevance | live 57.9% / mismatched 87.8% no-call | recovery "about 65%" (live 44–49%) | recovery 81–82% (live 56%, mismatched 76%) |
| Fable's forecast, good answers | 24–28 | "near 20" | "likely falls short of 40" (60 good of 369 paired texts) |
| test | recovery 0.989, 37 good | recovery 0.578, 17 good | recovery 0.770, 29 good |

- **All nine R.4a runs have now put a readout-level counter-offset in layer 23's MLP.** All three b50 runs say why: reverting parts of the LoRA fails, because the offset is encoded redundantly. Reverting blocks 0–11 or 12–23 each leaves 86–100% calls, and partial reverts cost 3–17 points of call accuracy.
- **r2's down_proj relative change of 2.12 is large** because 45 columns that were near zero received the full anti-shift vector. The edit only acts where the gate rows let those neurons fire. Retention (−0.41) shows it is well contained.
- **Fable's in-session forecasts were close this time.** For r2 and r3, forecast and test recovery differ by 0.05–0.07, and both good-answer forecasts were in the right direction. r1's good answers were underestimated (37 against 24–28).

## How the sessions went
| | b50 r1 | b50 r2 | b50 r3 |
|---|---|---|---|
| wall-clock hours | 4.36 | 4.21 | 5.52 |
| tool calls (python / forward requests) | 156 (99 / 57) | 156 (93 / 63) | 181 (118 / 63) |
| refused forward requests (no charge) | 7 | 13 | 13 |
| passes: generate / teacher_forced / judge | 26 / 2 / 22 | 21 / 19 / 10 | 28 / 2 / 20 |
| nudges / compactions / error waits | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| notional $ | 41.24 | 47.39 | 59.05 |

- **The budget was split differently in each run.** r1 and r3 spent 20–22 passes on judging. r2 spent 19 on teacher-forced passes, to measure where its gate neurons fire. Pass usage per budget, in judge passes: b10 2 in every run, b25 6–8, b50 10–22.
- **b50 runs took 4.2–5.5 h**, against the brief's 7–10 h. Two ran in parallel without hitting the Fable cap.

### Rule audit
All python calls were scanned (`scripts/audit_session.py` through the R.4a adapter, plus regex counts), and the flagged calls read in full.
- **No gradients, autograd, optimizers or iterative fitting in any run.** r3's 12 closed-form hits are ridge solves (`np.linalg.solve` on saved activations) for its read directions, the same kind as b25 r3 and allowed.
- **Readout simulation (allowed):** all three sized their final edit through the final norm and unembedding from saved layer-23 activations.
- **⚑ r2 (runs 27–30, 58, 60, 62, 78):** rebuilt block MLP activations from saved residuals: RMSNorm, then gate/up, then SiLU.
  - Run 27 is an "MLP attribution" that splits activation change into weight change and input change, as b10 r2 did.
  - Runs 58–62 rebuild block 23's inputs at every position to design the gate-row silencing, and "recompute activations exactly" with the edit.
  - Run 78 does the same for block 20, for a rejected variant.
  - This is the same capability as b10 r2 and r3.
- **⚑ (light) r1 (runs 16, 63) and r3 (run 71):** RMSNorm of saved residuals, then the LoRA's A projections, or the down_proj LoRA delta on saved inputs. Each was done one layer at a time, with no SiLU and no chaining, for diagnosis only.
- **Over nine runs, 5 are ⚑ or ⚑ (light).** Recomputing modules on saved activations is now ordinary practice for Fable in this harness. The rules ladder should name it as a capability rung rather than treat it as an exception.

## Predictions, scored (R3 and R4; `exp_r/R4a_design.md`, amendment 2026-11-21)
| prediction | result | |
|---|---|---|
| b50 irrelevance 75–90, central 82 | 61.5 | ✗ |
| b50 retention −1.5 to −4, central −2.5 | −1.65 | ✓ |
| b50 good answers 30–55, central 40 | 27.7 | ✗ |
| shape: b10 → b25 gains larger than b25 → b50, on all four quantities (diminishing returns), P = 0.6 | b10 → b25: irrelevance +7.5, good +7.7. b25 → b50: −10.8 and −5.0 | ✗: no gain at all from b25 to b50 |
| shape: within-budget SD comparable to the b25 → b50 gap, P = 0.6 | irrelevance SD 12.5–16.1 vs gap −10.8 | ✓ |
| R4 cost: b50 7–10 h per run | 4.2–5.5 h; $41–59 notional | ✗ (faster than predicted) |

- **I expected budget to matter more than it does.** Across b10, b25 and b50, within-budget spread is as large as or larger than the differences between budgets. Retention was the one quantity I predicted correctly at b50.

## What it means
- **The pass budget doesn't measurably change the result between 10 and 50 passes (n = 3 each).** More passes changed what Fable measured: more judging at b25 and b50, more teacher-forced probing for r2. They didn't change what it built, or how well it worked on test.
- **The diagnosis is the same in all nine runs.** Each found a near-constant call offset at the decision token, with judgment intact (logit-lens AUC ≈ 0.95), and failed to remove it by reverting the LoRA. Each cancelled it at the readout in layer 23. What varies is the dose, and with it the trade-off: irrelevance recovery against call accuracy.
- **The run-to-run spread looks like a choice of operating point.** Each run picked a dose from its own practice-set evidence, and the dose sets where it lands on the trade-off. One run in nine (b10 r2) landed inside all three targets.
- **For the talk** (month-one and provisional, "an experiment, not a method"):
  - the headline is the nine-run scatter on the trade-off, with the budget shown as colour, not as a curve;
  - the claim is "the same diagnosis and the same kind of fix nine times, with no measurable benefit from a 5× larger budget";
  - b10 r2 is still the demo run, with its ⚑ caveat.

## Surprises or problems
- **The volume filled up** (above). Nine session folders with saved activations (4.5–18 GB each) plus 81 GB of `artifacts/` took the 200 GB volume to 100%. It is now 250 GB, with about 38 GB free after r3. Future sessions should drop `gen_*_acts` and `fwd_*` from `/srv/wst-sessions` after `finalize`, or the driver should check free space before starting.
- **Wrapper bugs during r1's self-test** (above).
- **SSH launches with `setsid nohup ... &` hang the calling SSH** on these pods. The remote process is unaffected. A `timeout` on the launch call works around it.
- **`score_answers.py` expects `result/BFCL_v4_irrelevance_result.json`.** I first copied it to the run root for r1, and the first judge run failed. Fixed in the post-processing script.

## Cost
- **GPU** (A40 at $0.59/h; uptime from get-pod):

  | pod | uptime | cost |
  |---|---|---|
  | `wst-r4a-b50` (`o0q38sx7k286h7`) | 26,363 s | ≈ $4.32 |
  | `wst-r4a-b50b` (`hk7cyi4scldxro`) | 18,155 s | ≈ $2.98 |
  | `wst-r4a-b50c` (`cnobba7a0um51y`) | 30,230 s | ≈ $4.95 |
  | **total** | | **≈ $12.25** |

  About $3.20 of the total is waste: r1's failed self-tests (≈ $1.60), r3's 2 h disk-full stall (≈ $1.20), and idle time between steps. All three pods are stopped, EXITED verified with get-pod.
- **Volume:** grown from 200 to 250 GB. At RunPod's network-volume rate of about $0.07/GB per month, that adds about $3.50 a month.
- **Driver:** $147.68 notional (Max), with no waits.
- **Judge:** 52 in-session passes, plus 443 eval judgements and regrades.
- **Max usage (user readings):**
  - weekly 5% before r1;
  - 11% after r1, so r1 plus its eval took about 6 points;
  - 24% after r2 and r3 (the user's reading before stage 4), so those two took about 13 points.

## Records
- **Session records:** `artifacts/r4a/b50_r{1,2,3}/` (calls, transcript, passes, judge, submission, `work_final` with `notes.md`; 0.3, 7.2 and 11 GB). All three exclude `submission_ckpt/model.safetensors`, which stays on the volume.
- **Driver logs:** `artifacts/r4a/repeats_logs/b50_r{1,2,3}_driver.log` and `b50_r1_driver_noenv_fail.log`.
- **Eval:** `artifacts/pulled/artifacts/bfcl/r4a_b50_fable_r{1,2,3}/` and `artifacts/r4a/score_a_b50.json`.
- **Judgements:** `artifacts/a/answers/` (logs: `artifacts/a/answers_run_r4a_b50_r1.log` and `answers_run_r4a_b50_r23.log`).
- **On the volume `61gtyoubem`:** `/workspace/r4a/sessions/r4a_b50_fable_r{1_20261123T030037Z,2_20261123T142351Z,3_20261123T142500Z}/`.

## Next
- **The original repeats plan is complete:** b10, b25 and b50, three runs each.
- **Stage 4 (synthesis) needs the user's approval.** The nine runs share a diagnosis and differ in how they build the edit and in the dose they choose, which is the situation synthesis is meant for.
- **Small fixes, not yet made:**
  - the driver cost sum;
  - self-test stdout flushing and sidecar start;
  - `audit_session.py` reading the R.4a format;
  - a rules-ladder rung for module recomputation;
  - cleaning activations from `/srv` after `finalize`, or a free-space check.
