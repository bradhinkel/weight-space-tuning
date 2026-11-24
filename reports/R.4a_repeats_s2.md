# R.4a repeats, stage 2: three b10 sessions

## What was done
- **Three b10 sessions**, `r4a_b10_fable_r1`, `_r2` and `_r3`, ran in parallel on 2026-11-22. Each was on its own A40 in CA-MTL-1, all on volume `61gtyoubem`.
- **Only the pass budget changed.** The configs are copies of `r4a_b25_fable.json` with only `tag`, `budget: 10` and `task_file` changed. The task text, `exp_r/r4a/sessions/r4a_b10.md`, differs from `r4a_b25.md` only in its two "25 forward passes" lines.
- **Everything else matched R.4a and stage 1:** harness, rules, Fable 5.1 at effort max, practice_v2, and the eval (BFCL v4 test, then `score_a.py`, then `score_answers.py`).
- **Pods:**

  | run | pod | self-test |
  |---|---|---|
  | r1 | `wst-demo` | passed 34/34 earlier on 2026-11-22 |
  | r2 | new `wst-r4a-b10b` | 34/34 |
  | r3 | new `wst-r4a-b10c` | 34/34 |

  The A.11c gate reproduced on all three.
- **Deviation 1: the Fable usage limit stalled r2 and r3 for about 4.6 h, mid-session.**
  - It hit around 18:50Z, at 9/10 passes for r2 and 5/10 for r3, with the message "You've hit your Fable limit · resets Nov 28".
  - The user applied their spare reset. Both sessions then resumed by themselves through the driver's error-wait path, after 11 waits each.
  - No passes were lost, and `/work` was unchanged.
  - The model was paused, not reset: on resume the same conversation continued, with the driver's "the session continues" message.
- **Deviation 2: host GPU faults after the sessions.**
  - `wst-r4a-b10c` lost CUDA ("CUDA driver initialization failed") just as r3 finished, so r3's BFCL failed before scoring anything.
  - `wst-r4a-b10b` couldn't restart ("[GPU requires reset]").
  - r3 was evaluated on a fresh A40, `wst-r4a-eval`, using the same checkpoint from the volume.
  - Both faulty pods were deleted on the user's instruction, after I checked that every record was on the volume and the laptop.

## Results

### BFCL v4 test split
| run | irrelevance | mean call acc | recovery | retention (pts) | good answers / 240 | targets met |
|---|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 | 81 | |
| broken | 0.83 | 84.62 | 0.000 | 0 | 0 | |
| corrective LoRA (A.7, trained) | 82.50 | 82.73 | 1.048 | −1.89 | 75 | 3/3 |
| **b10 r1** | **62.08** | **85.67** | **0.786** | **+1.04** | **12** | 1/3 |
| **b10 r2** ⚑ | **74.17** | **82.79** | **0.941** | **−1.84** | **40** | **3/3** |
| **b10 r3** ⚑ | **58.33** | **84.07** | **0.738** | **−0.55** | **23** | 1/3 |
| **b10 mean ± SD (n = 3)** | **64.9 ± 8.3** | **84.2 ± 1.4** | **0.82 ± 0.11** | **−0.45 ± 1.44** | **25.0 ± 14.1** | |
| b10 mean without r2 (n = 2) | 60.2 | 84.9 | 0.76 | +0.24 | 17.5 | |
| b25 mean ± SD (n = 3, stage 1) | 72.4 ± 12.5 | 83.2 ± 1.4 | 0.92 ± 0.16 | −1.43 ± 1.37 | 32.7 ± 9.6 | |

⚑ = a run that recomputed MLP sublayers from saved residuals in the kernel (see Rule audit). On the user's instruction these runs are kept in the tables and means, and marked.

- **Targets:** recovery ≥ 0.80, retention ≥ −2, good answers ≥ 40.
  - **b10 r2 meets all three, the first run in R.4a to do so.** It is right at the limits: retention is 0.16 points inside, and its 40 good answers sit on the threshold with about ±3 judge noise.
  - r1 and r3 meet only retention.
- **b10 vs b25:** the b10 mean is 7.5 points lower on irrelevance and 7.7 lower on good answers, and 1 point better on retention.
  - Both differences in means are smaller than either budget's SD.
  - With n = 3 each, b10 and b25 can't be told apart on any quantity. Welch t on irrelevance ≈ 0.9.
- **Same trade-off as at b25.** The runs with more irrelevance recovery lose more call accuracy and give more good answers. r2 is at one end (74.2 / −1.84 / 40) and r1 at the other (62.1 / +1.04 / 12).

**Per category** (accuracy; calls made / prompts):

| category | broken | corrective LoRA | **b10 r1** | **b10 r2** ⚑ | **b10 r3** ⚑ | b10 mean ± SD |
|---|---|---|---|---|---|---|
| simple_python | 92.5 | 90.0 | 93.2 (400/400) | 91.8 (400/400) | 91.2 (398/400) | 92.1 ± 1.0 |
| multiple | 92.0 | 91.0 | 92.5 (200/200) | 91.0 (199/200) | 92.0 (199/200) | 91.8 ± 0.8 |
| parallel | 90.0 | 88.0 | 91.0 (200/200) | 89.0 (198/200) | 89.0 (199/200) | 89.7 ± 1.2 |
| parallel_multiple | 89.0 | 88.0 | 89.5 (200/200) | 89.0 (200/200) | 89.0 (200/200) | 89.2 ± 0.3 |
| live_simple | 70.2 | 69.0 | 74.8 (253/258) | 65.1 (245/258) | 70.9 (249/258) | 70.3 ± 4.9 |
| live_multiple | 74.1 | 70.4 | 72.9 (1041/1053) | 70.8 (1023/1053) | 72.3 (1039/1053) | 72.0 ± 1.1 |
| irrelevance | 0.8 | 82.5 | 62.1 (called 91/240) | 74.2 (called 62/240) | 58.3 (called 100/240) | 64.9 ± 8.3 |
| live_irrelevance | 1.0 | 68.2 | 56.3 (called 388/884) | 71.0 (called 257/884) | 61.8 (called 342/884) | 63.0 ± 7.4 |

- **r1 is the first edit in this line of work to beat the broken model on call accuracy:** +1.04 overall, and live_simple +4.6 points.
- **r2's retention loss is in live_simple (−5.1) and live_multiple (−3.3).** Those are the live categories, which practice covers least, the same pattern as b25 run 1.
- **r2's live_irrelevance (71.0) is above base (68.8).**

### Answer quality, BFCL test irrelevance (240 prompts)
| | base | corrective LoRA | **b10 r1** | **b10 r2** ⚑ | **b10 r3** ⚑ | b10 mean ± SD | b25 mean |
|---|---|---|---|---|---|---|---|
| text answers | 189 | 198 | 149 | 178 | 140 | 156 ± 20 | 174 |
| good (helpfulness ≥ 4) | 81 | 75 | 12 | **40** | 23 | 25.0 ± 14.1 | 32.7 |
| mean helpfulness, text | 3.16 | 2.98 | 2.03 | 2.38 | 2.39 | 2.27 ± 0.21 | 2.39 |
| mean fluency, text | 4.12 | 4.04 | 3.88 | 3.10 | 3.41 | 3.46 ± 0.39 | 3.61 |
| answered / declined_with_help / declined_only | 143 / 26 / 6 | 140 / 27 / 13 | 33 / 12 / 90 | 69 / 3 / 26 | 39 / 6 / 51 | | |
| degenerate (judge category) | 6 | 11 | 7 | 52 | 29 | 29 ± 23 | 28 |
| hit token cap / same sentence ≥ 5× | | | 9 / 9 | 58 / 45 | 26 / 24 | | |

- **The two ways of failing seen at b25 appear again:**
  - r1 is fluent but declines: 90 bare declines, 7 degenerate.
  - r2 answers but loops: 69 answers, 52 degenerate, and 58 of 178 text answers hit the cap.
- **Only r2 steered the first token toward answer openers.** Its edit raises "The", "Here" and "Sure" and lowers "I" and "As". That raises both good answers and loops, as "The" did in b25 run 1.
- **Judge consistency:** 329 items have been regraded across all runs in this answers directory: 79.9% exact, 99.4% within 1 point. Read each count as roughly ±3.

## The edits
| | b10 r1 | b10 r2 ⚑ | b10 r3 ⚑ |
|---|---|---|---|
| tensors | 35 | 1 | 1 |
| LoRA reverted | yes: attention side of the full-attention blocks 3/7/11/15/19/23 (q/k/v/o) and DeltaNet blocks 21–22 (34 tensors) | no | no |
| readout edit | `layers.23.mlp.down_proj` rank-1: −22·u kᵀ. u = mean fine-tuned − base residual at the decision position; k = a key on the down_proj input that reads ≈1 there, orthogonalized against blank-line, call-answer and loop contexts (2% of the tensor's norm) | `layers.23.mlp.down_proj` rank-1 (norm 0.23 vs 32.6). The key is the decision-token activation pattern; the value lowers `<tool_call>`, "I" and "As" and raises "To", "The", "Here", "Sure", "Yes", "Hello" | `layers.23.mlp.down_proj`, 6 columns (neurons 4543, 3989, 12, 96, 1049, 4326, which detect the answer start). Each subtracts α_j·(marker + 0.3u("I") + 0.6u("Here") − 0.2u("To")) |
| Fable's good-answer forecast | 25–30 | ≈31 | 30–40 |
| test | 12 | 40 | 23 |

- **All six b25 and b10 runs put a readout-level counter-shift in layer 23's MLP.** None used neuron 3776, run 1's choice. The read side was built a different way each time: neuron columns, new neurons, rank-1 and rank-2 maps, a column subset.
- **b10 r1 is the first run to combine that with a partial LoRA reversion.** Its diagnosis: blocks 19, 21 and 22's attention deltas reduce the relevant-vs-irrelevant gap, and reverting them restores it. This may be why r1 has the best retention: it removes the push where it differs by prompt, rather than only cancelling a constant.
- **Forecasts are now worse.** At b25, Fable's good-answer forecasts were accurate in all three runs. At b10, r1 and r3 were overestimated by about 2× and r2 underestimated. With fewer passes Fable judged fewer items and extrapolated more.
  - r3 submitted a final variant, F4, that it never generated or judged (it had no passes left). Its forecast rested on per-opener statistics from F3.

## How the sessions went
| | b10 r1 | b10 r2 ⚑ | b10 r3 ⚑ |
|---|---|---|---|
| active hours (excluding the limit stall) | 2.26 | ≈2.0 | ≈2.1 |
| tool calls (python / forward requests) | 71 (59 / 12) | 68 (55 / 13) | 80 (66 / 14) |
| refused forward requests | 2 | 3 | 4 |
| passes: generate / teacher_forced / judge | 7 / 1 / 2 | 8 / 0 / 2 | 7 / 1 / 2 |
| nudges / compactions / error waits | 0 / 0 / 0 | 0 / 0 / 11 | 0 / 0 / 11 |
| notional $ (excluding waits) | 18.47 | 24.74 | 26.59 |

- **Every b10 run used exactly 2 judge passes**, against 6–8 at b25. Most of the budget cut came out of generation, not out of judging.
- **The driver's notional cost is wrong for paused runs.** For runs with error waits, `total_cost_usd_notional` adds up the CLI's running totals, so r2 showed $207. The figures above are each session's last running total. The fix is a one-line change in `exp_r/r4a/driver.py`; it hasn't been made.

### Rule audit
All python calls were scanned (`scripts/audit_session.py` through the R.4a adapter, plus regex counts), and the flagged calls read in full.
- **No gradients, autograd, optimizers or iterative fitting in any run.** The three "iterative fit" hits in r3 are loops over neuron indices.
- **r1:** per-block tensor inputs computed by applying RMSNorm to saved residuals, one block at a time with no chaining. This is linear, as in b25 r3, so it's allowed.
- **⚑ r2** (runs 21, 22, 24) **and ⚑ r3** (runs 16, 17) recomputed MLP sublayers in the kernel: RMSNorm(saved mid-residual), then gate and up, then SiLU, and for r2 the down projection too.
  - r2 did this for all 24 blocks, with both the fine-tuned and the base weights. It used a counterfactual (base MLP weights on fine-tuned inputs, and the reverse) to split "input-driven" from "LoRA-driven" changes.
  - r3 did the same for layers 15–23.
  - Each layer was computed on its own, from saved forward-pass activations, with no chaining, and only for diagnosis.
  - The submitted edits were sized by a last-layer readout simulation, which R.4a already allowed.
  - This goes past rule (b) as written, "do not implement or simulate the network's computation on any input in the kernel", and well past b25 r2's last-layer-only neuron simulation.
  - **User ruling (2026-11-22):** keep both runs in the tables, marked. They are an example of how Fable analyses the model given saved activations.
  - **Rung 1 of the rules ladder should state this explicitly:** "recompute any single module on saved inputs" is a distinct capability from "readout simulation".
- **b25 recheck:** none of the three b25 runs recomputed the model's own MLPs.
  - b25 r2 did evaluate SiLU for neurons it designed itself, in blocks 12, 16 and 20 (runs 71, 73 and 87), on saved inputs. Those were variants it rejected; its final edit is in layer 23 only.
  - That is a lighter version of the same capability. The submitted edit doesn't use it, so b25 r2 stays unmarked, but the rules-ladder entry should cover it.

## Predictions, scored (R3, b10 part; `exp_r/R4a_design.md`, amendment 2026-11-21)
| prediction (b10 mean) | result | |
|---|---|---|
| irrelevance 55–80, central 68 | 64.9 | ✓ |
| retention −2 to −8, central −4 | −0.45 | ✗ (better than the range) |
| good answers 10–35, central 20 | 25.0 | ✓ |
| reasoning: "fewer passes leave less room for judge passes, so answer quality drops first" | judge passes fell from 6–8 to 2. Good answers fell 7.7 on average, with SD 14 | partly: the direction is right, but the drop is smaller than the noise |

- **The shape predictions** (diminishing returns with P = 0.6; within-budget SD comparable to the b25 → b50 gap with P = 0.6) need stage 3.
- **On the stage 1 and 2 data, b10 → b25 is itself smaller than the within-budget SD.**
- **Retention was again too pessimistic.** Five of six runs are within ±2 of broken, so my model of "a stronger edit costs calls" was off. The edits that sit at the readout leave call accuracy alone.

## What it means
- **At n = 3, a 10-pass budget isn't measurably worse than 25 passes.** The b10 and b25 means differ by less than one SD on every test quantity. The single best run (b10 r2) came from the smaller budget.
- **The diagnosis is the same at both budgets.** All six runs found the fine-tune's near-constant "call" offset at the decision token and cancelled it in the last MLP. That idea reproduces; how much each run cancels, and whether it also steers the first token, does not.
- **For the talk** (month-one and provisional, "an experiment, not a method"):
  - the headline is a scatter of six runs on the irrelevance vs retention vs answer-quality trade-off, not a budget curve;
  - b10 r2 is the run to use for the demo: all three targets, one rank-1 tensor;
  - the ⚑ caveat applies to it.

## Surprises or problems
- **Fable limit:** a Fable-specific cap stopped both parallel sessions while the weekly figure read about 52%. The driver's error-wait path (72 × 20 min) saved both runs. Without the user's spare reset, r3 would have been auto-submitted at 5 of 10 passes after 24 h.
- **Host GPU faults on two of three new A40s** in CA-MTL-1 (shown as RunPod alerts). One appeared right after a session ended; had it appeared mid-session, the session would have been lost.
- **Laptop-side SSH blips:** "Network is unreachable" for a few seconds. They stalled one self-test watcher, and the judge sidecars logged failed polls, which they retried by themselves.
- **Self-test race:** the self-test's `judge_sidecar --once` can run before the request file exists, and then fails with "judge did not answer in time". The fix is to run the sidecar in loop mode during the self-test. That isn't in the repo yet; it's only in the scratch script.

## Cost
- **GPU** (A40 at $0.59/h; uptime from get-pod):

  | pod | uptime | cost |
  |---|---|---|
  | `wst-demo` (`o34t3h3dn6jznc`) | 9,286 s | ≈ $1.52 |
  | `wst-r4a-b10b` (`lkcjj98doklept`) | 26,224 s, plus a failed restart of about 5 min | ≈ $4.35 |
  | `wst-r4a-b10c` (`2te91hjqwc9j5w`) | 29,137 s | ≈ $4.78 |
  | `wst-r4a-eval` (`m0u7hmblpz5r15`) | 939 s | ≈ $0.15 |
  | **total** | | **≈ $10.8** |

  About $5.4 of the total is the two pods idling during the Fable-limit stall.
  - Pod status: `wst-demo` and `wst-r4a-eval` are stopped, EXITED verified with get-pod. `wst-r4a-b10b` and `wst-r4a-b10c` were deleted, 404 verified.
- **Driver:** $69.80 notional (Max), excluding waits.
- **Judge:** 6 in-session passes, plus 467 eval judgements and regrades.
- **Max usage:** weekly 42% before stage 2. The Fable cap was hit at about 52% weekly, and the spare reset was used. After stage 2 and the reset: weekly 5% (user, 2026-11-23). This includes Experiment V's expensive image-analysis session, so stage 2's own share can't be separated.

## Records
- **Session records:** `artifacts/r4a/b10_r{1,2,3}/` (calls, transcript, passes, judge, submission, `work_final` with `notes.md`). All three exclude `submission_ckpt/model.safetensors`, which stays on the volume; r1's and r3's folders are large because of their saved activations in `work_final`.
- **Driver logs:** `artifacts/r4a/repeats_logs/b10_r{1,2,3}_driver.log`.
- **Eval:** `artifacts/pulled/artifacts/bfcl/r4a_b10_fable_r{1,2,3}/` and `artifacts/r4a/score_a_b10.json`. On the volume there is also `r4a_b10_fable_r3_failed_b10c/` (vLLM log of the failed run).
- **Judgements:** `artifacts/a/answers/` (logs: `artifacts/a/answers_run_r4a_b10_r{1,2,3}.log`).
- **On the volume `61gtyoubem`:** `/workspace/r4a/sessions/r4a_b10_fable_r{1_20261122T154552Z,2_20261122T165632Z,3_20261122T171906Z}/`.

## Next
- **Stage 3 (b50 × 3) needs the user's approval.** Given the noise (irrelevance SD 8–12 per budget), three b50 runs could only show a gap larger than about 15 points.
  - Each run costs about 7–10 h of Fable, and three in parallel would likely hit the Fable cap again.
  - Alternative: go to stage 4 (synthesis) now, using the six runs as material. They share a diagnosis and differ in construction, which is the situation synthesis is meant for.
- **Small fixes, not yet made:**
  - the driver cost sum;
  - running the self-test sidecar in loop mode;
  - letting `scripts/audit_session.py` read the R.4a calls format;
  - an explicit rules-ladder entry for module recomputation.
