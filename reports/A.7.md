# A.7 Arm 3: Corrective LoRA (gradient baseline)

## What was done
- **Data** (`exp_a/build_corrective.py`, run on the pod, 1,199 s). Metadata: `artifacts/pulled/artifacts/a/arm3/corrective_meta.json`.
  - **Abstention examples.** 4,500 xLAM *train* queries, each paired with the tool list of a different xLAM train example. A pair was used only if the two share no function names and no content words between the query and the tools' names and descriptions. All 4,500 found a partner.
  - **Abstention responses.** The base model (Qwen/Qwen3.5-2B, greedy, thinking off, same chat template) wrote the responses, so this is self-distillation. Responses that contained a call or hit the 384-token cap were dropped, and the first 3,000 clean ones were kept.
  - **Call examples.** 3,000 further xLAM train examples, disjoint from the abstention queries and unchanged.
  - No practice or test data was used.
- **Training.** `common.train.lora --init-from artifacts/a/broken` with the A.2 defaults: r16, α32, lr 2e-4, 1 epoch, 32k tokens per optimizer step. That gave 140 steps in 26 min, with loss going 0.33 → 0.14 → 0.08. The adapter is merged into `/workspace/artifacts/a/arm3`.
- **Eval.** BFCL test split, tag `a7_lora` (584 s). Weights touched were measured against `broken`.
- **Fix during the run.** The first training launch ran out of GPU memory on step 1. A 16k-token micro-batch needs about 15 GiB just for the fp32 logits over the 248k vocabulary. I relaunched with `--micro-tokens 8192`, and accumulation rose from 2 to 4 automatically, so the tokens per optimizer step are unchanged. The data build wasn't repeated.

### Base-model behaviour on the mismatched pairs (from the build stats)
| outcome | count of 4,500 |
|---|---|
| called a tool anyway | 368 (8.2%) |
| hit the 384-token cap | 646 (14.4%) |
| empty | 0 |
| clean prose (3,000 of these kept) | 3,486 |

The 8.2% call rate is close to the base model's own test-irrelevance miss rate (51/240 = 21% raw calls) but lower. The mismatches are cruder than BFCL's irrelevance items. Sample kept responses: *"I am sorry, but I do not have access to a tool that can fetch images from specific collections like 'cityscapes1010'…"* and *"I cannot provide real-time financial data or historical stock prices as I do not have access to live market feeds…"*

## Results
Tables come from `scripts/report_tables.py` and `scripts/score_a.py` (`artifacts/pulled/artifacts/a/score/a7_scores.json`). The b25 row is copied from `a5_all_scores.json`.

### Headline scores
| run | test irrelevance | mean call acc | irrelevance recovery | call retention (pts) | weights touched (vs start) | strict recovery |
|---|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 | — | 1.000 |
| broken | 0.83 | 84.62 | 0.000 | +0.00 | 57.95% (vs base) | 0.000 |
| Arm 1, Fable b25 (A.5) | 64.58 | 83.49 | 0.818 | −1.13 ✅ | 0.0001% (2,041) | 0.820 |
| Arm 2, steering L1 (A.6) | 0.83 | 83.46 | 0.000 | −1.16 ✅ | 33.09% | 0.000 |
| **Arm 3, corrective LoRA** | **82.50** | **82.73** | **1.048** | **−1.89** ✅ | **52.82%** | **1.048** |

- **The LoRA fully recovers irrelevance** and slightly overshoots the base model (82.50 vs 78.75). It also meets the retention target, though only just (−1.89 against ≥ −2).
- **It beats Fable b25 on recovery** (1.048 vs 0.818) at a cost of 0.76 more points of retention.
- **The weights-touched metric needs a caveat.** It counts merged-checkpoint elements that differ from `broken`. A rank-16 LoRA merged into every targeted linear layer is dense, so it changes 52.82% of elements. That is about 1.17B elements, roughly 5.7 × 10⁵ times Fable b25's footprint, measured the same way. Counted as trainable parameters, the adapter is 16.8M (0.75%), which is still about 8,000 times b25's 2,041 changed weights.

### Test split (accuracy %)
| category | base | broken | b25 | a6_steer | a7_lora | n |
|---|---|---|---|---|---|---|
| simple_python | 71.00 | 92.50 | 91.75 | 92.00 | 90.00 | 400 |
| multiple | 81.50 | 92.00 | 92.00 | 89.50 | 91.00 | 200 |
| parallel | 49.50 | 90.00 | 88.50 | 91.00 | 88.00 | 200 |
| parallel_multiple | 69.50 | 89.00 | 88.00 | 87.50 | 88.00 | 200 |
| irrelevance | 78.75 | 0.83 | 64.58 | 0.83 | 82.50 | 240 |
| live_simple | 66.28 | 70.16 | 68.60 | 68.60 | 68.99 | 258 |
| live_multiple | 72.65 | 74.07 | 72.08 | 72.17 | 70.37 | 1053 |
| live_irrelevance (practice) | 68.78 | 1.02 | 58.60 | 1.58 | 68.21 | 884 |
| **mean call accuracy** | **68.40** | **84.62** | **83.49** | **83.46** | **82.73** | |

**Raw `<tool_call>` responses for a7_lora:**
- irrelevance: 42/240, with 0 unparsed.
- live_irrelevance: 285/884, with 4 unparsed.
- Call categories (simple_python, multiple, live_simple, live_multiple): 6–12 responses per category now abstain where they should have called. That over-abstention is most of the retention loss, and `live_multiple` loses the most (−3.7).

## Gate status
Not a gate. All three arms are done. The plan's next step is A.8, the results table and figures.

## Surprises or problems
- **The gradient baseline is strong on outcome.** With 6,000 self-distilled examples and 26 GPU-minutes, it restores irrelevance completely while staying within the retention target. On recovery and retention alone, it beats the driver.
- **The driver's case is the size of the edit, and being able to read it.** Fable b25 recovered 82% by changing one embedding row, 2,041 weights, with a one-line explanation of how it works. The LoRA needed a dense change across about half the model's elements and a training-data pipeline. The comparison for the talk is therefore *how much change and how much understanding per unit of recovery*, not recovery alone.
- **Different inputs.** The LoRA used 6,000 training examples and gradients. The driver used ≤ 25 practice GPU passes and no gradients. Neither arm saw test data.
- **One seed each,** as with A.5. The LoRA's variance hasn't been measured.

## Cost
- GPU:
  - build: 20 min
  - failed launch: under 1 min
  - training: 26 min
  - eval: 10 min
  - weights diff: about 2 min

  That totals about 1.0 GPU-h, roughly $0.60 at $0.59/h.
- Driver usage: none.

## Next step
A.8: the Experiment A results table (all arms, plus the unmetered CPU run as a separate point) and figures in `artifacts/a/figures/`:
- recovery vs retention scatter
- recovery vs weights touched, log scale
- the b10/b25/b50 budget curve
