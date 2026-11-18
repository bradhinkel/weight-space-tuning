# R.4a: Fable on Experiment A's task with R.2's harness (budget 25)

## What was done
- **Design:** `exp_r/R4a_design.md` (committed `511116d` with the predictions, before any run), plus its amendment dated after the self-test and before the session.
- **One change from R.2:** the task. R.2's harness, rules, refusals and driver are unchanged. The model is Experiment A's broken tool router: Qwen3.5-2B plus the merged rank-16 xLAM LoRA.
- **P0** (`reports/R.4a_P0.md`): with `<tool_call>` masked, broken gave 11 good answers out of 240. So the task text pointed Fable to the LoRA: "the damage is the fine-tune…".
- **Harness:** `exp_r/r4a/`. It has the BFCL scorer (97.6% item agreement with BFCL), `base` as a third model, `load_base`/`lora` in the kernel, and judge passes graded locally by the A.10 judge. The self-test passed 33/33 under the amended gate.
- **Session:** `r4a_b25_fable_20261117T190942Z`.
  - Fable 5.1, effort max, budget 25 passes.
  - Started 19:09Z, ended 22:29Z (3.3 h).
  - 110 tool calls: 77 python and 33 forward requests (8 refused for free).
  - 0 nudges, 0 compactions, 0 error waits. $30.3 notional.
- **Eval:** unchanged from A.
  - `common.evals.bfcl` on the submission checkpoint (698 s);
  - `scripts/score_a.py`;
  - `scripts/score_answers.py` on test irrelevance, using the same blind Sonnet 5.5 judge with the earlier runs' judgements cached.

## Results

### BFCL v4 test split
| run | irrelevance | mean call acc | recovery | retention (pts) | good answers / 240 |
|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 | 81 |
| broken | 0.83 | 84.62 | 0.000 | 0 | 0 |
| A.5 b25 (embedding "I" edit) | 64.58 | 83.49 | 0.818 | −1.13 | 4 |
| A.11c (rank-1 response-start edit) | 50.00 | 77.05 | 0.631 | −7.57 | 11 |
| corrective LoRA (A.7, trained) | 82.50 | 82.73 | 1.048 | −1.89 | 75 |
| **R.4a b25** | **85.42** | **81.61** | **1.086** | **−3.01** | **43** |

**Target:** recovery ≥ 0.80 ✓, retention ≥ −2 ✗ (−3.01), good answers ≥ 40 ✓ (43). So two of the three targets were met.

**Per category** (accuracy; calls made / prompts):

| category | broken | A.11c | corrective LoRA | **R.4a b25** |
|---|---|---|---|---|
| simple_python | 92.5 | 92.2 | 90.0 | **92.0** (397/400) |
| multiple | 92.0 | 90.5 | 91.0 | **90.0** (197/200) |
| parallel | 90.0 | 75.5 | 88.0 | **83.5** (188/200) |
| parallel_multiple | 89.0 | 60.0 | 88.0 | **88.0** (195/200) |
| live_simple | 70.2 | 70.5 | 69.0 | **64.7** (225/258) |
| live_multiple | 74.1 | 73.5 | 70.4 | **71.4** (1014/1053) |
| irrelevance | 0.8 | 50.0 | 82.5 | **85.4** (called on 35/240) |
| live_irrelevance | 1.0 | 38.7 | 68.2 | **71.3** (called on 254/884) |

- Irrelevance and live_irrelevance are now higher than both base and the corrective LoRA.
- **The retention loss is almost entirely missing calls on relevant prompts:**
  - live_simple: 33 of 258 prompts got a text answer.
  - parallel: 12 of 200.
  - live_multiple: 39 of 1,053.
- parallel_multiple, A.11c's worst category (−29), lost only 1 point.

### Answer quality, BFCL test irrelevance (240 prompts)
| | base | A.11c | corrective LoRA | **R.4a b25** |
|---|---|---|---|---|
| text answers | 189 | 120 | 198 | **205** |
| good (helpfulness ≥ 4) | 81 | 11 | 75 | **43** |
| mean helpfulness, text | 3.16 | 2.21 | 2.98 | **2.59** |
| mean fluency, text | 4.12 | 3.81 | 4.04 | **3.42** |
| answered / declined_with_help / declined_only | 143 / 26 / 6 | 19 / 9 / 80 | 140 / 27 / 13 | **81 / 9 / 49** |
| degenerate (judge) | 6 | 10 | 11 | **49** |

- **Answers vs declines:** this is the first weight edit that answers more often than it declines. It has 81 real answers, against 19 for A.11c.
- **New failure: loops.**
  - 49 of 205 text answers loop until the 4,096-token cap; 41 repeat the same sentence at least 5 times.
  - These are mostly "The user is asking…" narration, which Fable had diagnosed and halved on practice but not removed.
- **Judge consistency:** 134 items have been regraded across all runs in this answers directory: 82.1% exact, 99.3% within 1 point. That is lower than P0's 95.8% exact on 24 items. The good/not-good boundary moves for a few items, so read 43 as roughly ±3.

## The edit
- **One tensor changed:** `layers.23.mlp.down_proj.weight`, two of its 6,144 columns, a relative change of 2.6%. No LoRA component was touched.
- **Column 3776:** a counter-offset.
  - Neuron 3776 fires at 14–17 at the answer-start position and at paragraph starts, about 0.6 elsewhere, and 0.9 between tool calls.
  - Fable added to its column the minimum-norm residual direction that, per unit of typical activation, changes logits as follows: `<tool_call>` −9, `<think>` −7, `<` −6, with 36 common word openers held at 0.
  - This is a closed-form least-squares solve against the unembedding, not fitting.
- **Column 5198:** +1.5 to the logit of "The" at the answer start.
  - Neuron 5198 is about 1.8 at the answer start and ≤ 0.45 at paragraph starts.
  - Fable moved the "The" nudge here from 3776, because on 3776 it re-fired at every paragraph start and caused loops. Its teacher-forced pass 20 measured this.

## How the session went
- **Pass use:**
  - 18 generation passes, 1 teacher-forced, 6 judge.
  - 4 passes went to held-out items: H256 for the E4d and E4g "both" passes, and 256 fresh irrelevant items in the final `base_vs_edited` pass.
- **The diagnosis path** (Fable's notes, `artifacts/r4a/b25/work_final/notes.md`):
  1. **Reverting the LoRA, layer group by layer group:**
     - In layers 15–23 (72 tensors) or layers 13–14, it changed nothing on irrelevance.
     - Projecting out the deltas' decision-position input component also changed nothing.
     - In layers 0–12, calls went up and accuracy fell 5–10 points.
     - In layers 13–23 together, no calls were made at all.
     - So the LoRA pointer was tested properly and rejected on evidence.
  2. **The key finding:** the fine-tune keeps the relevance signal but shifts it.
     - At the answer start, the margin `<tool_call>` minus the best word token is 8.1 ± 1.6 on irrelevant prompts and 11.7 ± 0.6 on relevant ones (AUC 0.96, base 0.92).
     - So it is a near-uniform +10-logit offset, created in prompt processing in layers 0–12 and redundant across layers.
  3. **The fix:** compensate the offset where the decision is read out.
     - Because layer 23 is last, a down_proj column's effect on the first token can be computed from the saved final residuals. Fable chose strengths this way without spending passes, and generation matched within a few percent.
  4. **Answer quality next:**
     - A judge pass split by first token showed that "I" openers were declines 85% of the time.
     - "To" openers led to narrated plans and then calls; this was rejected in pass 10 (multi-call regression).
     - "The" on 3776 caused paragraph loops (pass 20), so it moved to 5198 (E4g).
  5. **Rejected along the way:** a relevance-scaled boost on neuron 3002, which is active during answers; a layer-18 state shift, which stopped multi-call; "The" +2.0, which cost 3–12 accuracy points; and an EOS boost, which was infeasible.
- **Fable's final claims vs test:**
  - **"About 35–45 good answers":** 43. Correct.
  - **"Essentially full recovery on synthetic, about 90% on live":** irrelevance recovery was 1.09 and live 71.3% (base 68.8). Conservative.
  - **"Call accuracy within roughly one point":** retention was −3.0. Optimistic. The loss is in live_simple, which practice has no counterpart to (see below).
- **Rule audit** (all 77 python calls scanned for autograd, optimizers, norms, softmax, embeddings and layer loops; the flagged ones read in full):
  - no gradients, optimizers, autograd or iterative fitting;
  - no model run in the kernel;
  - probes (diagonal LDA on saved activations) and least-squares solves for edit columns are analysis or closed-form edits, which are allowed.
- **Borderline point:** Fable predicted first-token behaviour by adding its column contribution to saved layer-24 residuals, then applying the final RMSNorm and the tied unembedding.
  - This is the logit-lens readout R.2 already accepted, applied to the edited state. It is exact only because the edited layer is the last one.
  - It does not compute any layer on any input, so I count it as allowed. It is the most powerful use of "analysis of forward-pass results" seen so far, and the rules ladder should name it explicitly.
- **Cost of analysis:** Fable printed only token ids or short decoded excerpts. It decoded the whole vocabulary once, into a word-start mask, without printing it.

## Predictions, scored
| # | prediction | result | verdict |
|---|---|---|---|
| 1 | P0: broken masked 15–40 good (P(≥ 50) = 0.25) | 11 | ✗ low (scored in R.4a_P0) |
| 2 | P0: ≥ 50% of masked answers "declined only" | 49% | ✗ narrowly (R.4a_P0) |
| 3 | A.11c reproduction shows a multi-call loss ≥ 10 pts (P = 0.85) | pooled −6.2; same-function −17.2 | ✗ as stated; ✓ after the amendment |
| 4 | 25 passes, 3–8 h, 0 compactions, 1–4 judge passes | 25, 3.3 h, 0, **6 judge passes** | ✓ except judge count |
| 5 | Practice predicts test within ±6 pts on all four quantities (P = 0.7) | 2 of 4 | ✗ (details below) |
| 5′ | Different-function multi → parallel_multiple gap > 6 pts (P = 0.6) | gap 1 pt (practice 0, test −1) | ✗ (good news: the blind spot did not bite) |
| 6 | Retention ≥ −3 with parallel and parallel_multiple within 5 of broken (P = 0.65) | −3.01; parallel −6.5; parallel_multiple −1 | ✗ narrowly on both |
| 6′ | Same, P = 0.45 | | ✗ |
| 7 | Irrelevance 55–75; P(recovery ≥ 0.80) = 0.5 | 85.4; recovery 1.086 | range ✗ (above); recovery ≥ 0.80 ✓ |
| 8 | Good answers 20–50, central 30 (given P0 < 50); P(≥ 40) = 0.35; P(> 11) = 0.85 | 43 | ✓ in range; ≥ 40 happened; > 11 ✓ |
| 9 | All three targets (P = 0.25); beats A.5 b25 on all axes (P = 0.35) | retention −3.01 misses both | ✗ |
| 10 | Edit = ≤ 200 LoRA components in layers 18–23, probably plus a decision edit | two non-LoRA columns in layer 23; LoRA reversion tested and rejected | ✗ (only the "plus a decision edit" part happened) |
| 11 | GPU $3–6; driver $15–40 notional; weekly Max 3–6% plus judge | GPU ≈ $4.91 (A40 $3.46 + H100 $1.45); driver $30.3; weekly +10 points (13% → 23%) | ✓ GPU and driver; ✗ Max (over) |

**Prediction 5 in detail** (practice = held-out passes 21 and 25; edited model; "pts" = percentage points):

| quantity | practice | test | gap | within ±6? |
|---|---|---|---|---|
| no-call, synthetic (xlam_mismatched → irrelevance) | 96.1 (178) | 85.4 | −10.7 | ✗ |
| no-call, live (live_irrelevance hold → live_irrelevance) | 62.7 (228) | 71.3 | +8.6 | ✗ |
| single-call, change vs broken | 0.0 (30, no flips) | −0.5 / −2.0 / −5.5 (simple / multiple / live_simple) | ≤ 5.5 | ✓ |
| multi-call same-function, change vs broken | 0.0 (25) | parallel −6.5 | 6.5 | ✗ (by 0.5) |
| multi-call different-function, change vs broken | 0.0 (35) | parallel_multiple −1.0 | 1.0 | ✓ |
| good answers, rate per irrelevant prompt | 13.3% (20/150) → 32/240 | 17.9% (43/240) | +4.6 | ✓ |

- **The no-call gaps are mostly differences in item difficulty, not a measurement failure.** Measured against base on the same item sets, practice and test agree within about 7 points:
  - practice: base 96 vs edited 96 (synthetic), base 67 vs edited 63 (live);
  - test: base 78.8 vs edited 85.4, base 68.8 vs edited 71.3.
- **The real blind spot is live_simple:** −5.5, with 33 prompts getting no call. Practice has no live_simple-style items. `live_relevant.jsonl` has only parallel, parallel_multiple and relevance items.

## What it means
1. **The replicated harness carried over to the three-objective task.**
   - With the same 25 passes, the same rules and the same model family of driver, this is the best weight edit in Experiment A by a wide margin:
     - recovery 1.09 vs A.5 b25's 0.82 and A.11c's 0.63;
     - good answers 43 vs 4 and 11;
     - parallel_multiple 88 vs A.11c's 60.
   - It is the first driver arm to beat the trained corrective LoRA on irrelevance (85.4 vs 82.5, live 71.3 vs 68.2).
   - It is still behind the LoRA on retention (−3.0 vs −1.9) and on answer quality (43 vs 75).
2. **What changed from A is measurement, not the agent.**
   - In A, Fable couldn't see multi-call loss or answer quality. Here every pass reported multi-call accuracy split by same- and different-function calls, and judge passes graded text answers.
   - Fable used both. It spent 6 passes on the judge, and it rejected three candidates (E4c, E4f, E11) for multi-call losses it could see.
   - The two losses that remained are both in groups practice doesn't cover well: live_simple, and the loops' length on test prompts.
3. **The LoRA pointer did not become the edit.**
   - Fable tested it seriously, reverting it by layer group and by subspace, and found the damage redundant and spread across layers 0–12.
   - The repair is a readout-level counter-offset, an "understand, then compensate" edit rather than "remove the cause".
   - That is a real finding about this fine-tune: the relevance signal survives, shifted by about +10 logits. It is also a limit: the text mode's tool-framing comes from the same early layers, which is why loops and declines remain.
4. **The agent optimizes what it measures** (the outline's thesis), seen from the good side this time.
   - What the harness measured (no-call, multi-call by type, judged quality) is what improved.
   - What it under-measured (single calls on real users' prompts, long-answer loops) is where the edit lost points.
5. **One session.** The variation between sessions is unmeasured. Before claiming "a weight edit can match a trained LoRA" for the article, a second b25 session would show whether 43 / −3.0 / 1.09 is typical.

## Surprises or problems
- **The eval's first launch failed** on a missing `ninja` (`/workspace/venv/bin` was not on PATH), the same as P0's second launch. I relaunched it with PATH set. `common/evals/bfcl.py` should set this itself.
- **`score_answers.py`** reads `result/BFCL_v4_irrelevance_result.json` one level above where `bfcl` writes it. I copied the file up, as for the earlier runs.
- **Teacher-forced input size:** Fable dropped a 576-text teacher-forced screening because inline texts were too large to send in one request. A file-reference option in the harness would have saved it a pass of planning.
- **Retention:** −3.01 misses predictions 6 and 9 by 0.01 point. The verdicts are scored as stated.

## Cost
- **GPU:**
  - A40 `wst-r4a-dev` (`kym5l03a0t1yf8`): 21,102 s (5.9 h) at $0.59/h ≈ $3.46. This covered the harness build, self-tests, the session and the eval. It is now stopped (EXITED, verified with get-pod).
  - H100 `wst-r4a` (`4jwymwk1l6z53e`) for P0: ≈ $1.45, stopped.
- **Driver:** $30.3 notional (Max). Judge: 6 in-session passes and 205 eval judgements on Max.
- **Max usage after** (reported by the user): session 33%, weekly 23%. The last reading, on 2026-11-16, was session 21% and weekly 13%, so R.4a (P0 judging, self-tests, the session and the eval judge) used about 10 points of the weekly allowance.

## Records
- **Session records:** `artifacts/r4a/b25/` (calls, transcript, passes, judge requests, notes, the submitted tensor).
- **Eval:** `artifacts/pulled/artifacts/bfcl/r4a_b25_fable/`; `artifacts/r4a/score_a.json`.
- **Judgements:** `artifacts/a/answers/` (log: `artifacts/a/answers_run_r4a.log`).
- **On the volume `61gtyoubem`:** `/workspace/r4a/sessions/r4a_b25_fable_20261117T190942Z/` (the submission checkpoint and the 14 GB work folder).

## Next
- **Repeat b25 once** to measure session-to-session variation before using these numbers in the article.
- **Add a live_simple-style practice group** (BFCL-format single calls on real-user prompts that aren't in test) and a loop or length check to pass summaries. These are the two gaps the result points to.
- **Rules ladder, rung 1 candidate:** name "readout simulation from saved residuals" explicitly as allowed or not allowed. Fable's strongest tool here sits right at the boundary.
