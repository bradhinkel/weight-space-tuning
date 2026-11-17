# R.2 b50: Fable on the original Experiment 8 task, budget 50

## What was done
- **Goal:** the same setup as R.2 b10 (`reports/R.2.md`), at their headline budget of 50 passes, compared with their budget-50 run (p50c).
- **Unchanged from b10:**
  - driver: Fable 5.1 at effort `max`, in headless Claude Code 2.1.325;
  - tools: only `python` and `forward` over MCP;
  - their `message_p50c.txt` task text, with notices 001 and 002;
  - the harness, self-test (26/26 on the new pod) and R.1 evaluation pipeline.
- **New: an original-model cache** (`exp_r/r2/server.py`, commit `fdbd167`):
  - original-model generations are stored by (prompt, seed, max_tokens) and reused when a pass repeats a problem;
  - edited models always generate fresh.
- **Hardware:** a new pod, `wst-r2b` (A100 SXM 80 GB, EUR-IS-1, $2.19/h), on the same 60 GB volume. The b10 pod's host had no free GPU.

## Results

| model | edit | MATH-500 | cut at 32k | GSM8K | cut at 32k |
|---|---|---|---|---|---|
| base (R.1, ours) | — | 7.70 | 93.0% | 13.76 | 87.6% |
| Fable b10 | 2 columns | 39.15 | 59.5% | 48.67 | 48.5% |
| p50c (theirs, budget 50; R.1 rerun) | 64 columns, 131,072 weights | 50.85 (theirs 48.10) | 32.2% | 67.29 (theirs 67.10) | 7.0% |
| **Fable b50** | **15 columns + 1 attention head zeroed** | **51.85** | 33.5% | **64.71** | 16.3% |

- **Per-problem differences** (SE over problems):

  | comparison | MATH-500 | GSM8K |
  |---|---|---|
  | b50 − base | +44.15 ± 1.70 (336 problems better, 1 worse) | +50.95 ± 1.13 (919 better, 13 worse) |
  | b50 − p50c | **+1.00 ± 1.15** (108 better, 107 worse) | **−2.58 ± 0.91** (178 better, 246 worse) |
  | b50 − b10 | +12.70 ± 1.21 | +16.03 ± 1.05 |

- **Problems solved by at least one answer:**

  | | base | b10 | b50 | p50c |
  |---|---|---|---|---|
  | MATH-500 (of 500) | 83 | 294 | 350 | 342 |
  | GSM8K (of 1,319) | 300 | 813 | 1,017 | 1,007 |

  - b50 solved 40 MATH-500 problems that p50c never solved, and p50c solved 32 that b50 never solved.
  - On GSM8K the split is 101 against 91.

- **Practice prediction:**
  - On 10 fresh practice sets (2,560 problems), Fable measured 9.8% → 54.6% (+44.8 ± 1.0), with 1,162 problems wrong→right and 16 right→wrong.
  - By source: GSM8K-train 13.5 → 67.0 and MATH-train 7.6 → 47.1.
  - Both match the test results closely: GSM8K 64.7 and MATH-500 51.9.
- **Eval wall clock:** MATH-500 1.5 h (34.0M tokens) and GSM8K 1.2 h (28.5M tokens). This is faster than b10, because fewer answers run to 32k.

## The edit
Five tensors, from `/work/edited`, verified identical to Fable's saved `best_E3h15`:

| tensor | change | Fable's reading |
|---|---|---|
| `layers.22.mlp.down_proj` | columns 573, 387, 787, 1209, 1047 ×2 | dampeners for "Wait", "Maybe/Perhaps", "Actually", "Alternatively", "However" |
| `layers.23.mlp.down_proj` | columns 507, 375, 3063, 964, 496, 209, 2788 ×2 | dampeners for "Let", "But", "Okay", "Is", "Why", "Or", "Hold" |
| `layers.21.mlp.down_proj` | columns 622, 125 ×2 | dampeners for "Let" and "Why" |
| `layers.18.mlp.down_proj` | column 1261 ×0.5 | a loop-specific "Let" promoter |
| `layers.15.self_attn.o_proj` | head 7's 256 input columns zeroed | an induction head that copies earlier " Wait" |

- **Size:** 15 columns (15,360 weights) plus one head (262,144 weights). Relative change per tensor: 0.9–3.9% for the MLPs and 32% for the `o_proj`, from one zeroed head of 8.
- **Mechanism, by Fable's account:**
  - The original model usually reaches the answer: 72% of cut-off GSM8K answers contain the correct number.
  - It then starts "Wait" re-checks, and after two or three the cycle locks in at a per-token probability above 0.99.
  - Each common line opener has a late-layer neuron that writes against its own token but fires too weakly. Fable found these from the weights with a logit-lens score and confirmed them with activations.
  - Doubling those neurons turns re-entering the loop into a coin flip. On a normal correct trace the median logprob change is 0.001 nats.
- **Dose and ablations** (Fable's practice measurements):
  - The single "Wait" neuron at ×1.5 / ×2 / ×3 gave 45.7% / 50.0% / 42.6% on its first tuning set (tune-A), so ×2 is the best dose.
  - Zeroing head 15.7 on top of the neurons added +1.9 ± 1.4. Zeroing head 19.0 instead hurt.
  - Damping the layer-23 retrieval heads, or the copy heads 15.5, 15.6 and 19.5, dropped accuracy to about 12%.
  - Damping generic words ("If", "So", "This"), or a ×3 dose, increased looping.
  - Forcing the "Here's my thought process" opening made things worse, although it correlates with success.

## How the session went
- **Duration:** 19.1 h wall clock (03:17Z–22:24Z), with 176 tool calls (124 `python`, 52 `forward`).
  - The 52 forwards were 50 counted passes and 2 refused requests, which cost nothing: one asked for unknown practice IDs, the other gave both "texts" and "chats".
  - Pass types: 36 `generate` and 16 `teacher_forced`.
- **No compactions, nudges or usage-limit waits.** Fable used all 50 passes in a single turn of 177 model turns, so the `notes.md` handoff path again went unused. Fable kept `/work/notes.md` up to date anyway: 98 lines, one entry per pass.
- **Shape of the session:**
  1. Diagnosis.
  2. Attribution on recorded hidden states and attention.
  3. A one-neuron "Wait" fix, then a dose curve.
  4. Adding dampeners for other line openers (E3, E3n), and trying several more that it rejected.
  5. Head ablations, ending with E3h15 = E3n plus head 15.7 zeroed.
  6. The last 10+ passes confirmed the edit on fresh sets.
- **Original-model cache:**
  - 9,216 original-model answers were requested; 5,760 (62.5%) were served from the cache and 3,456 were generated.
  - Fable reused its tuning sets heavily, so without the cache roughly 22 passes would have regenerated identical original answers, at about 30 min each.
- **Rule audit** (all 124 `python` calls scanned; the flagged ones read in full):
  - no gradients, optimizers, autograd or fitting;
  - no model loaded or run in the kernel;
  - the loops are scans over recorded activations and weights, such as scoring 3,584 neurons from stored activations.
- **Borderline points:**
  - As in b10, damping a family of hesitation tokens comes close to the ban on "global knobs". But each dampener is a specific neuron whose effect Fable measured, and ordinary tokens change by about 0.001 nats.
  - Zeroing a whole attention head is a structural ablation, not a scaling of single columns. The task allows any weight edit.

## What it means
1. **With the same budget of 50 passes, Fable matched GPT-6 Astra's p50c:**
   - MATH-500: +1.0 ± 1.2, a tie.
   - GSM8K: −2.6 ± 0.9, slightly behind and significant.
   - p50c was run through the same pipeline (R.1), so this comparison is like for like. It is still one session per side.
2. **The two edits work differently.**
   - Fable's is mostly token-specific dampener neurons plus one head.
   - p50c subtracts a loop direction across 64 columns.
   - They solve different problems in part: 40 vs 32 on MATH-500 and 101 vs 91 on GSM8K.
3. **Looping still costs Fable on GSM8K.** 16.3% of answers are cut at 32k, against p50c's 7.0%, and that is where the 2.6-point gap comes from. On MATH-500 the cut shares are about equal (33.5% vs 32.2%).
4. **Most of the gain came early.**
   - b10 already gave 71% of b50's MATH-500 gain and 69% of its GSM8K gain.
   - The extra 40 passes added about 13–16 points, mostly by extending the same idea from one "Wait" neuron to 14 dampeners.
5. **Practice predicted test well again:** +44.8 on practice against +44.2 and +51.0 on test.

## Differences from the original (record)
- **Everything listed in R.2 still applies:** the driver and tool surface, seeding, the kernel memory cap and chmod on the volume.
- **The cache and batching:** vLLM output depends on what else is in the batch.
  - A fresh rerun of an identical batch matched the cache exactly.
  - The same 8 problems run inside a batch of 12 gave 0 of 8 identical texts.
  - So when a pass mixes new and repeated problems, a cached original answer is a valid sample with the same seed, but not exactly what a fresh mixed batch would have produced.
  - Their server didn't reproduce across passes either (notice 002), so this is within the original's own noise.
  - The edited model is always generated fresh, so the comparison stays paired.

## Surprises or problems
- **None that interrupted the session.**
  - The local machine reached the pod over SSH for 19 h with no drops long enough to matter.
  - The login refresh token was valid until 2026-12-13.
- **It finished faster than expected:** 19 h, against an estimated 20–40 h. The cache and the 50-pass cap both helped.

## Cost
- **GPU:** pod `wst-r2b` ran 25.1 h × $2.19 ≈ $55. That covers setup, self-test, the cache test, the 19.1 h session and the 2.7 h eval. The volume is about $4/month and keeps billing while the pod is stopped.
- **Driver:** $46.01 notional (Max-billed): 47.4M input tokens (mostly cache reads) and 432k output tokens, 143k of them thinking.
- **Max usage:**
  - before: weekly 0% at 2026-11-15 06:15Z (not re-read at the start);
  - after (2026-11-16 22:35Z): session 21%, weekly 13%.
  - The weekly figure includes the monitoring conversation, so b50 used a little under 13% of a week.

## Records
- Session: `artifacts/sessions/r2_b50_fable_20261116T031726Z/` (transcript, TASK.md, driver summary).
- Pod records: `artifacts/r2/b50/` (`calls.jsonl`, final `/work` without generation dumps, `notes.md`, submission tensors).
  - Left on the volume, not pulled: `original_cache.jsonl` (461 MB), `passes/` (1.3 GB), the generation dumps and the full checkpoint.
- Eval: `artifacts/r2/eval/r2b50/` (per-problem results, scores, logs).
- Driver log: `artifacts/r2/b50_driver.log`.

## Next
- R.3: the Gemma control.
- R.4: steps toward our tool-router setup.
- Possible: a persistent vLLM engine to cut the 2–4 min startup per pass, and a second b50 session to measure variation between sessions.
