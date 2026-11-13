# A.11c Balanced practice data: rerun of A.11b with the style confound removed

## What was done
- **Why:** in A.11b, every irrelevant practice prompt was real-user (BFCL live) and every relevant one was synthetic (xLAM). Fable's probe partly learned style, and the edit stopped calling tools on live relevant test prompts: retention −19 (`reports/A.11.md`).
- **New practice data:** `exp_a/build_practice_v2.py` writes `/workspace/artifacts/a/practice_v2/`, which covers all four combinations:

  | | a tool fits | no tool fits |
  |---|---|---|
  | real users | **new** `live_relevant.jsonl`, 56: BFCL live_parallel 16, live_parallel_multiple 24, live_relevance 16 (with reference calls where BFCL has them) | `live_irrelevance.jsonl`, 854 |
  | synthetic | `xlam_practice.jsonl`, 2,000 | **new** `xlam_mismatched.jsonl`, 600: xLAM questions paired with an unrelated xLAM example's tools |

  - None of the source categories is a test category, and the test guard dropped 0 items.
  - Mismatched pairs were kept only if no tool name is shared and no content word of the question appears in the offered tools. A spot check of 8 pairs found all clearly irrelevant.
- **Task:** `exp_a/sessions/a11c_b25.md`. It's A.11b's task with:
  - the four-file practice section ("a good edit has to work on all four");
  - a memory note (see below).
  - Rules, tools and budget are the same as A.11b: base model and activations available, closed-form edits only, no gradients, 25 passes.
  - Config: `a11c_b25_fable.json`.
- **Attempt 1 failed** (`a11c_b25_fable_20261113T035958Z`, 3,868 s, 5 passes used).
  - **Cause:** the Python kernel was OOM-killed. The pod's container limit is 50 GB (cgroup `oom_kill 1`). Fable was running its own fp32 CPU forward pass and holding several full-size copies of the weights.
  - The harness only restarted kernels that timed out, so every later call failed on a broken pipe.
  - Fable stopped without submitting. Nothing was evaluated.
- **Harness fix** (`common/harness/tools.py`):
  - A dead kernel is detected and restarted. `W` and `W_BASE` are shared-memory views, so edits survive; variables are lost.
  - `forward_pass` checks the kernel before charging, so a pass on a dead kernel costs nothing.
  - Test: `scripts/test_kernel_restart.py`. It kills the kernel twice; all checks passed.
  - The task now states the 50 GB limit and the restart behaviour.
- **Attempt 2** (`a11c_b25_fable_20261113T050808Z`): this is the result below.

## Results

### The edit
- **Session:** 3,676 s, 23 of 25 passes:
  - 15 current generate
  - 1 base generate
  - 5 current activations
  - 2 base activations
- **Fable's diagnosis:** the fine-tune writes a nearly prompt-independent "call mode" vector into the residual stream **at one position only**, the response-start token after `</think>`.
  - At every layer, the per-prompt shift has cosine 0.80–0.94 with the mean shift.
  - Its norm after layer 23 is 23.9.
  - It adds about +12.5 logits to `<tool_call>` on every prompt. The base model's relevance signal survives, but compressed from 6 to 3 logits.
- **The edit:** one tensor, `layers.23.mlp.down_proj.weight`, rank 1:
  ```
  W ← W_ft + outer(−1.1 · S, u)
  ```
  - **u** is a ridge-regression detector on layer 23's MLP intermediate activations. It fires about 1.0 at the response-start position and ≤ 0.11 elsewhere on held-out prompts.
  - **S** is the mean fine-tune-minus-base residual at that position.
  - The scale 1.1 was calibrated on practice prompts: 1.0 under-corrected and 1.2 hurt calls.
- **What it means in practice:** at the first decision token, the edit subtracts the average "call mode" shift and leaves every other position alone. **It doesn't depend on the prompt, so it can't learn the style confound.**
- **To compute activations,** Fable re-implemented the model's attention offline in the kernel (0.4% reconstruction error).
- **Weights changed:** 11.7M of the tensor's 12.6M entries changed in bf16, but the edit is a single rank-1 update (2048 + 6144 = 8,192 parameters' worth of information). The max entry change is 0.036.
- **Audit:** the only solvers used were `torch.linalg.solve`, `qr` and `svdvals`. There were no `backward`, `optim`, `requires_grad` or `grad` calls.
- **Fable's own validation** (fresh practice prompts never used for fitting):
  - irrelevant no-call 34/64 (live 12/32, mismatched 22/32)
  - relevant correct 51/64 (fine-tune: 53/64)
  - It reported that only about 12–14 of the 34 text answers were genuinely helpful.

### BFCL v4 test split
| run | irrelevance | mean call acc | recovery | retention (pts) |
|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 |
| broken | 0.83 | 84.62 | 0.000 | 0 |
| A.5 b25 (embedding "I" edit) | 64.58 | 83.49 | 0.818 | −1.13 |
| A.11b (probe neuron, confounded data) | 83.33 | 65.59 | 1.059 | −19.03 |
| **A.11c (rank-1 response-start edit)** | **50.00** | **77.05** | **0.631** | **−7.57** |

**Per category** (accuracy, calls made / prompts):
| category | broken | A.11b | **A.11c** |
|---|---|---|---|
| simple_python | 92.3 (400/400) | 88.3 | **92.3 (400/400)** |
| multiple | 92.0 (200/200) | 88.0 | **90.5 (196/200)** |
| parallel | 90.0 (200/200) | 87.5 | **75.5 (169/200)** |
| parallel_multiple | 89.0 (200/200) | 73.0 | **60.0 (145/200)** |
| live_simple | 69.4 (258/258) | 32.6 | **70.5 (256/258)** |
| live_multiple | 73.6 (1053/1053) | 24.2 | **73.5 (1046/1053)** |
| irrelevance | 0.8 | 83.3 | **50.0 (called on 120/240)** |
| live_irrelevance | 0.9 | 98.2 | **38.7 (called on 550/884)** |

### Answer quality, BFCL test irrelevance (240 prompts, same blind judge as A.10)
| | base | A.5 b25 | A.11b | **A.11c** | LoRA |
|---|---|---|---|---|---|
| text answers | 189 | 155 | 200 | 120 | 198 |
| good answers (≥ 4) | 81 | 4 | 14 | **11** | 75 |
| mean helpfulness, text answers | 3.16 | 1.84 | 1.84 | **2.21** | 2.98 |
| mean fluency, text answers | 4.12 | 3.15 | 3.00 | **3.81** | 4.04 |
| judged degenerate | 6 | 43 | 64 | **10** | 10 |
| declined only / answered | 6 / 143 | 97 / 11 | 103 / 22 | **80 / 19** | 13 / 140 |

The judge regraded 92 items, now including A.11c: it agreed exactly on 83.7% and was within 1 point on 98.9%.

## What it means
1. **The data fix worked for the confound.**
   - live_simple and live_multiple are back to the broken model's level (70.5 and 73.5), against 32.6 and 24.2 in A.11b.
   - Single calls are untouched: simple 92.3, multiple 90.5.
   - With balanced data, Fable chose an edit that doesn't read prompt style at all.
2. **But it's a partial repair at a cost.**
   - Recovery is 0.63, and retention is −7.6. That's worse than A.5's −1.1 and short of the −2 target.
   - The loss is concentrated in **parallel and parallel_multiple** (−14.5 and −29). Those prompts need several calls, and the edit makes the model skip or under-produce calls there: 169/200 and 145/200 attempted.
   - Fable's practice score for relevant prompts (strict, 51/64) didn't flag this, because most of its relevant practice prompts need a single call.
3. **Answer quality is still the unsolved part.**
   - When A.11c answers in text, the answers are cleaner than earlier edits': fluency 3.81, 10 degenerate vs 43–64.
   - But only 11 of 240 are good. Two thirds of its text answers are still bare "I can only use the provided tools" declines.
   - Fable predicted this itself: removing the call-mode vector at the first token restores the decision, not the base model's answering style. The fine-tune's text mode favours declines.
4. **The pattern across all four edits:**
   - Every edit so far trades between irrelevance, retention and answer quality, and none gets all three.
   - The corrective LoRA (recovery 1.05, retention −1.9, 75 good answers) remains the only result near base quality on every axis.

## Gate status
Not a gate. This is a follow-up arm, and it **doesn't meet the target** (retention −7.6 vs ≥ −2; recovery 0.63). One session.

## Surprises or problems
- **Kernel OOM on attempt 1:** see above. Fixed, and the fix is tested.
- **Offline forward passes are slow:** Fable's CPU re-implementation of the model took about 10 minutes per batch of activations. It made that trade to get activations at arbitrary layers and positions without spending passes.
- **The pass budget isn't binding any more:** Fable used 23 of 25 and spent most of its wall clock in kernel compute. The activation API makes each pass far more informative than in A.5.
- **Old pod `uuxk7ksvb3bwwz` couldn't restart** because its host had no free GPU. I created a new pod, `1rw28dgc7v4a5o`, with the same volume. Both are now stopped.

## Cost
- **Pod `1rw28dgc7v4a5o`** (A40, $0.59/h): 2.47 h uptime (8,878 s), about $1.46. It covered setup, the data build, both attempts, the restart test and the eval.
- **Driver (notional):** attempt 2 was $15.10 (from its summary). Attempt 1 ran 3,868 s, with cost not separately recorded; it was comparable.
- **Judge:** 14 Sonnet 5.5 batches on the Max plan.

## Next step
1. **Fix the multi-call loss.** Add multi-call relevant prompts to the practice scoring. The live_parallel files are there, but the driver's 16-item sample under-weighted them. Or tell the driver explicitly to score parallel calls separately.
2. **Answer quality needs a different target than the first token.** Every edit that only changes the call/no-call decision inherits the fine-tune's decline-heavy text mode. Options:
   - give the driver a judged good-answer metric per pass;
   - encourage edits that also restore the base model's text-mode continuation, for example applying the same response-start correction more broadly while the model is writing text.
3. **Repeats:** all A.11 arms are a single session each.
