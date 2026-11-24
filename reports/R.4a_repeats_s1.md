# R.4a repeats, stage 1: three b25 sessions

## What was done
- **Two new b25 sessions**, `r4a_b25_fable_r2` and `r4a_b25_fable_r3`, were run in parallel on 2026-11-22.
  - Everything matched R.4a: harness, rules, task text (`exp_r/r4a/sessions/r4a_b25.md`), Fable 5.1 at effort max, practice_v2 without `live_simple_style.jsonl`, and the eval.
  - Each session started fresh. The configs differ from `r4a_b25_fable.json` only in the tag.
  - The R.4a session (`r4a_b25_fable_20261117T190942Z`) counts as run 1.
- **Predictions were written first.** They are in `exp_r/R4a_design.md`, "Amendment (2026-11-21)".
- **The R.4b hooks did nothing.**
  - The new hooks only switch on when `practice_dir` / `R4A_PRACTICE` is set or `live_simple_style.jsonl` is present.
  - On both pods the self-test passed 34/34 and reproduced the A.11c gate (xlam_multi_same_fn −17.2).
  - The pass summaries had the R.4a groups and no `live_single`.
- **Pods:** the brief's pods no longer existed.
  - r2 ran on `wst-demo` (`o34t3h3dn6jznc`).
  - r3 ran on a new A40, `wst-r4a-r3` (`t9kgouzbcwr8b8`). Both were in CA-MTL-1 on volume `61gtyoubem`.
  - The A40 type and image were unchanged, but the hosts were different.
- **Eval, as for R.4a:** BFCL v4 test, then `scripts/score_a.py`, then `scripts/score_answers.py` (Sonnet 5.5 judge on the 240 irrelevance prompts).

## Results

### BFCL v4 test split
| run | irrelevance | mean call acc | recovery | retention (pts) | good answers / 240 | targets met |
|---|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 | 81 | |
| broken | 0.83 | 84.62 | 0.000 | 0 | 0 | |
| corrective LoRA (A.7, trained) | 82.50 | 82.73 | 1.048 | −1.89 | 75 | 3/3 |
| R.4a b25 run 1 | 85.42 | 81.61 | 1.086 | −3.01 | 43 | 2/3 |
| **b25 run 2** | **71.25** | **84.05** | **0.904** | **−0.57** | **31** | 2/3 |
| **b25 run 3** | **60.42** | **83.91** | **0.765** | **−0.71** | **24** | 1/3 |
| **b25 mean ± SD (n = 3)** | **72.4 ± 12.5** | **83.2 ± 1.4** | **0.92 ± 0.16** | **−1.43 ± 1.37** | **32.7 ± 9.6** | |

- **Targets:** recovery ≥ 0.80, retention ≥ −2, good answers ≥ 40.
  - Run 2 meets recovery and retention but not good answers.
  - Run 3 meets only retention.
  - No run meets all three.
- **The runs fall along a trade-off line.** Run 1 has the highest irrelevance and the worst retention; runs 2 and 3 have less irrelevance and almost no retention loss.
  - Ranked by irrelevance (1 > 2 > 3), the runs fall in the same order on good answers.
  - Three points can't show a correlation, but nothing here suggests that any one run is better on every axis.
- **Run-to-run spread is the main finding.**
  - Irrelevance has a 25-point range and good answers a 19-point range.
  - Both are larger than any difference between methods that we have so far used in the article. For example, run 1 vs the corrective LoRA is 2.9 points of irrelevance.

**Per category** (accuracy; calls made / prompts):

| category | broken | corrective LoRA | run 1 | **run 2** | **run 3** | b25 mean ± SD |
|---|---|---|---|---|---|---|
| simple_python | 92.5 | 90.0 | 92.0 (397/400) | **92.2** (400/400) | **91.2** (400/400) | 91.8 ± 0.5 |
| multiple | 92.0 | 91.0 | 90.0 (197/200) | **92.0** (200/200) | **91.0** (200/200) | 91.0 ± 1.0 |
| parallel | 90.0 | 88.0 | 83.5 (188/200) | **90.0** (200/200) | **91.5** (200/200) | 88.3 ± 4.3 |
| parallel_multiple | 89.0 | 88.0 | 88.0 (195/200) | **88.5** (200/200) | **87.5** (200/200) | 88.0 ± 0.5 |
| live_simple | 70.2 | 69.0 | 64.7 (225/258) | **68.2** (251/258) | **70.2** (255/258) | 67.7 ± 2.8 |
| live_multiple | 74.1 | 70.4 | 71.4 (1014/1053) | **73.3** (1043/1053) | **72.1** (1043/1053) | 72.3 ± 1.0 |
| irrelevance | 0.8 | 82.5 | 85.4 (called 35/240) | **71.2** (called 69/240) | **60.4** (called 95/240) | 72.4 ± 12.5 |
| live_irrelevance | 1.0 | 68.2 | 71.3 (called 254/884) | **66.9** (called 297/884) | **61.7** (called 347/884) | 66.6 ± 4.8 |

- **Runs 2 and 3 almost never drop a call on a relevant prompt.** They make every call on the non-live categories and miss 7 and 3 of 258 on live_simple, against 33 for run 1.
  - Run 1's retention miss came from one run's edit being too strong. It was not built into the method.
- **Run 2's live_irrelevance (66.9) is close to base (68.8). Run 3's (61.7) is below it.**

### Answer quality, BFCL test irrelevance (240 prompts)
| | base | corrective LoRA | run 1 | **run 2** | **run 3** | b25 mean ± SD |
|---|---|---|---|---|---|---|
| text answers | 189 | 198 | 205 | **171** | **145** | 174 ± 30 |
| good (helpfulness ≥ 4) | 81 | 75 | 43 | **31** | **24** | 32.7 ± 9.6 |
| mean helpfulness, text | 3.16 | 2.98 | 2.59 | **2.35** | **2.23** | 2.39 ± 0.18 |
| mean fluency, text | 4.12 | 4.04 | 3.42 | **3.94** | **3.46** | 3.61 ± 0.29 |
| answered / declined_with_help / declined_only | 143 / 26 / 6 | 140 / 27 / 13 | 81 / 9 / 49 | **36 / 15 / 101** | **44 / 4 / 57** | |
| degenerate (judge category) | 6 | 11 | 49 | **9** | **27** | 28 ± 20 |
| hit 4,096-token cap / same sentence ≥ 5× | | | 49 / 41 | **7 / 6** | **27 / 14** | |

- **The three runs fail in different ways:**
  - Run 1 answers most often but loops a lot.
  - Run 2 is fluent and rarely loops, but 101 of its 171 text answers are bare declines.
  - Run 3 sits between the two and calls a tool on 95 of 240 prompts.
- **No run comes near the corrective LoRA's 75 good answers.** All three b25 edits fix the routing decision and leave the answer that follows weak.
- **Judge consistency:** 215 items have been regraded across all runs in this answers directory: 80.0% exact, 99.1% within 1 point. As in R.4a, read each good-answer count as roughly ±3.

## The edits
| | run 1 | run 2 | run 3 |
|---|---|---|---|
| tensors | `layers.23.mlp.down_proj` (2 columns) | `layers.23.mlp.{gate,up,down}_proj` (2 neurons rewritten) | `layers.23.mlp.down_proj` (rank-2 update) |
| size | 2.6% relative | 2 dead neurons replaced | 16% of the tensor's Frobenius norm |
| decision part | neuron 3776 column: closed-form logit counter-offset (`<tool_call>` −9) | neuron 812: a new gate that only suppresses call-starting tokens | R1: ridge-fitted read direction × unit vector toward base's text state, c = 38 |
| answer part | neuron 5198: "The" +1.5 | neuron 2048: a "no late call" guard | R2': text-continuation route, c = 12 |
| LoRA touched | no | no | no |

- **All three edits sit at the same place:** a readout-level change in the last MLP (layer 23), which overrides the fine-tune's offset on the `<tool_call>` decision at the answer start.
- **No run reverted or rescaled the LoRA.**
- **Each run built the change differently:**
  - run 1 added direction vectors to the columns of existing neurons;
  - run 2 wrote new neurons, with gate, up and down weights, into neurons that were dead;
  - run 3 wrote a low-rank map from a fitted read direction to a target direction.
- **Run 2's neuron 812 can only suppress a call, never create one.** Fable stated that design intent itself. It explains why run 2 lost almost no call accuracy.
- **Run 2 also found two facts that runs 1 and 3 did not state:**
  - the RMSNorm scale is 1 + weight;
  - the saved activation layers are numbered one off from the model's layers.

## How the sessions went
| | run 1 | run 2 | run 3 |
|---|---|---|---|
| hours | 3.3 | 3.58 | 2.87 |
| tool calls (python / forward) | 77 python | 122 (91 / 31) | 94 (66 / 28) |
| refused requests | 8 | 6 | 3 |
| pass use | edited 12, both 2, base_vs_edited 2, original 1, base 1, tf 1, judge 6 | base_vs_edited 4, both 10, tf 3, judge 8 | original 1, base 1, both 7, edited 9, judge 7 |
| nudges / compactions / error waits | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| notional $ | 30.3 | 34.31 | 25.98 |

- **Fable's final claims vs test:**
  - **Run 2:** practice live irrelevant was 121/200 with no call (base 134). Mismatched was 130/140. Call accuracy was identical except for one false positive. Judged good was 28/242 (12%).
    - On test: live_irrelevance 66.9, retention −0.57, good 31/240 (13%).
    - These are calibrated.
  - **Run 3:** Fable estimated 18–25 good answers; test gave 24. That is correct.
  - Fable's estimates were close in all three runs, so its practice-set judgement is a good guide to what that run's edit does on test. It does not show how much the result varies from one run to the next.
- **Rule audit** (all python calls in both runs, using `scripts/audit_session.py` through a small adapter for the R.4a record format, plus regex counts; flagged calls read in full):
  - **Clean in both runs:** no gradients, optimizers, autograd or iterative fitting, and no model run in the kernel.
  - **Run 3:**
    - closed-form dual ridge via `np.linalg.solve`;
    - sklearn's `LogisticRegression` is imported but never called;
    - per-module LoRA inputs computed from saved activations, with no chaining between layers.
  - **Run 2:** `margin_grad` is an analytic logit-lens direction, not autograd.
  - **Borderline (run 2):** Fable simulated its new neurons by applying SiLU(gate)·up to the saved layer-23 MLP inputs, then the readout.
    - This goes one step beyond run 1's readout simulation: it evaluates a nonlinearity inside the edited layer.
    - It is still exact only because layer 23 is the last layer, and it computes no other layer.
    - **Correction (stage 2 recheck):** for rejected variants (runs 71, 73, 87), Fable also evaluated neurons it designed itself in blocks 12, 16 and 20 on saved inputs. That isn't exact, because downstream layers would react. The submitted edit is layer 23 only. See `reports/R.4a_repeats_s2.md`, Rule audit.
    - I count it as allowed. Rung 1 of the rules ladder should name both forms of "readout simulation from saved residuals", not only the column-add form.

## Predictions, scored
**R1. Spread of the three b25 runs**

| # | prediction | result | |
|---|---|---|---|
| R1a | irrelevance range 5–20 (central 10), SD ≈ 6 | range 25.0, SD 12.5 | ✗ |
| R1b | P(all three ≥ 64.6) = 0.6 | run 3 = 60.4 | ✗ (P was 0.6) |
| R1c | recovery SD ≈ 0.08; P(all three ≥ 0.80) = 0.45 | SD 0.16; run 3 = 0.765 | ✗ / ✗ |
| R1d | retention range 1–4 (central 2) | 2.44 | ✓ |
| R1e | P(at least one run ≥ −2) = 0.4 | two runs (−0.57, −0.71) | happened (P was 0.4) |
| R1f | P(all three ≥ −5) = 0.75 | yes | ✓ |
| R1g | good-answer range 10–35 (central 20) | 19 | ✓ |
| R1h | P(another run ≥ 40) = 0.35 | no (31, 24) | ✓ |
| R1i | P(run 1 best of three on good answers) = 0.45 | yes | happened (P was 0.45) |
| R1j | P(a repeat meets all three targets) = 0.15 | no | ✓ |
| R1k | "Against": range under 5 on irrelevance and under 10 on good answers would mean the task fixes the outcome | 25 and 19 | the path matters |

- I underestimated the spread on irrelevance by about a factor of 2, and I was too pessimistic on retention.

**R2. Same mechanism?**

| # | prediction | result | |
|---|---|---|---|
| R2a | P(each repeat finds a readout counter-offset in layers 20–23) = 0.55 | 2/2, both in layer 23 | happened |
| R2b | P(neuron 3776 specifically) = 0.3 per run | 0/2 (run 2 uses neurons 812 and 2048; run 3 is a rank-2 update) | ✓ |
| R2c | P(mainly a LoRA reversion) = 0.25 per run | 0/2 | ✓ |
| R2d | P(both edits touch ≤ 3 tensors) = 0.5 | 3 and 1 | happened |

- **The location is consistent; the construction is not.** All three runs found the same idea, compensating at the last layer's readout, and built it three different ways.
- With n = 3, "the approach is reproducible, the details vary" is the most I can claim.

**R3. Budget curve (b25 only so far)**
- **Predicted b25 means:** irrelevance 70–88 (central 78), retention −2 to −5 (central −3.5), good 25–45 (central 33).
- **Observed:** 72.4 ✓ (in range, below central), −1.43 ✗ (better than the range), 32.7 ✓.
- The shape predictions wait for stages 2 and 3.
  - With an irrelevance SD of 12.5 at b25, "SD comparable to the b25 → b50 gap" (P = 0.6) now looks likely.
  - Three runs per budget may not separate b25 from b50.

**R4. Cost**
- **Predicted:** 2.5–5 h and $20–45 notional per repeat. **Observed:** 3.58 h / $34.31 and 2.87 h / $25.98 ✓.
- **Predicted weekly Max points:** 6–10 for stage 1. **Observed:** weekly 1% → 42% (the user's reading after stage 1;
  the plan has only a weekly limit now) ✗.
  - That 41 points also includes the Experiment V session, which ran in parallel all day from the same allowance, and
    the plan upgrade mid-stage, so stage 1's own share can't be separated. Even half of it would be over the range.
  - The brief's ~6–7 per b25 session, from R.4a's ~10 for everything, looks low for parallel days.

## What it means
- **One b25 session does not give a reliable number.**
  - R.4a's headline (irrelevance 85.4, above the corrective LoRA) was the best of three. The b25 mean of 72.4 ± 12.5 sits between A.5's embedding edit (64.6) and the trained LoRA (82.5).
  - Retention is better than R.4a suggested: a mean of −1.4, with two of three runs under 1 point.
- **What does reproduce is the diagnosis:** three independent sessions converged on a last-layer readout override of the `<tool_call>` decision. No run used the LoRA reversion the task text points toward.
- **The trade-off between irrelevance and retention is set by how strong each run's edit is.** A synthesis (stage 4) that combines the runs' components or chooses their strengths might move along it, or beyond it.
- For the talk, still month-one and provisional, "an experiment, not a method": report the b25 mean ± SD and the range, not run 1.

## Surprises or problems
- **The brief's pods were gone (404).** `wst-r1a` couldn't start ("not enough free GPUs on the host machine"), so I created a new A40 for r3.
- **The RunPod proxy SSH doesn't support exec.** Direct TCP SSH worked once the ports appeared.
- **`audit_session.py` expects a different calls format.** It found 0 calls until I added an adapter in the scratchpad that converts R.4a `calls.jsonl` records. The script should accept both formats.
- **Max session window:** two parallel Fable sessions plus the V session took the 5-hour window to 89%. The user upgraded the plan mid-run, and there were no error waits.
- **The existing eval problems recurred:**
  - `ninja` / PATH: I set PATH by hand.
  - `score_answers.py` reads the irrelevance result one level above where `bfcl` writes it: I copied the file up.
  - BFCL writes to `/workspace/artifacts/bfcl/<tag>`, not to the repo's artifacts folder.

## Cost
- **GPU** (A40, $0.59/h; both pods now stopped, EXITED verified with get-pod):
  - `wst-demo` (`o34t3h3dn6jznc`): 23,941 s uptime (6.65 h) ≈ $3.92, covering setup, self-test, the r2 session and its eval.
  - `wst-r4a-r3` (`t9kgouzbcwr8b8`): about 3.85 h ≈ $2.27.
  - Total ≈ $6.2.
- **Driver:** $60.29 notional (Max) for the two sessions.
- **Judge:** 15 in-session passes, plus 316 eval judgements and regrades on Max.
- **Max usage:** weekly 1% before, 42% after (shared with Experiment V).

## Records
- **Session records:** `artifacts/r4a/b25_r2/`, `artifacts/r4a/b25_r3/` (calls, transcript, passes, judge, submission, `work_final`). r3 excludes the submission's `model.safetensors`, which stays on the volume.
- **Driver logs:** `artifacts/r4a/repeats_logs/`.
- **Eval:** `artifacts/pulled/artifacts/bfcl/r4a_b25_fable_r{2,3}/` and `artifacts/r4a/score_a_repeats.json`.
- **Judgements:** `artifacts/a/answers/` (logs: `artifacts/a/answers_run_r4a_r3.log`, `artifacts/a/answers_run_r4a_r2.log`).
- **On the volume `61gtyoubem`:** `/workspace/r4a/sessions/r4a_b25_fable_r{2_20261122T052351Z,3_20261122T052848Z}/`.

## Next
- **Stage 2: b10 × 3.**
  - With the upgraded plan, the three sessions could run in parallel on three A40s, with the same configs and `budget: 10`.
  - The task text changes only in the budget number.
  - The R3 predictions for b10 are already written.
- **Before stage 3 (b50):** with a b25 SD of 12.5, three runs per budget can only detect gaps of about 15+ points. Decide whether b50 × 3 is worth running, or whether more b25 runs would be more informative.
- **Rules ladder, rung 1:** name readout simulation explicitly, both the column-add form and the last-layer-neuron (SiLU) form.
