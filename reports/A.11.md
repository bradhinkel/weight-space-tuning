# A.11 Round 1: asking the driver for good answers, not just no call

## What was done
- **Why:** A.10 showed the b25 edit restores *not calling* but rarely a *good answer*: 4 good answers of 240, against 81 for base. The A.5 task accepted "say you can't help", so the driver optimised for that. A.11 changes the goal and reruns at budget 25 with Claude Fable 5.1.
- **A.11a, improved task, same rules as A.5.**
  - Files: `exp_a/sessions/a11a_task.md`, `a11a_b25.md`, `a11a_b25_fable.json`.
  - The goal is now a helpful plain-text answer on irrelevant prompts while keeping real calls.
  - Bare refusals, "I can only X" templates, loops and wrong answers count as failures.
  - Fable is told to read its completions and report three rates: no call, good answer, and correct call.
  - The tools are exactly the A.5 tools.
- **A.11b, the same task with relaxed rules.**
  - Files: `a11b_task.md`, `a11b_b25.md`, `a11b_b25_fable.json`.
  - Fable also gets the **base model** (Qwen/Qwen3.5-2B):
    - its weights as `W_BASE` in the kernel;
    - `forward_pass(model="base")` to generate from it.
  - It also gets **activations**: `forward_pass(mode="activations")` writes per-layer residuals to `last_activations`, as `[25, n_pos, 2048]` fp16.
    - `positions="last"` allows up to 64 prompts per pass; `positions="all"` allows up to 8.
  - Closed-form edits and any computation in the kernel are allowed. Gradient training is banned.
  - Every pass, whatever its mode or model, costs 1 from the budget of 25.
- **Harness changes:**
  - `common/harness/tools.py`, `kernel.py`, `mcp_server.py`, `driver.py`.
  - The new tools appear only when the config enables them, so A.11a saw the A.5 interface unchanged.
  - Self-test: `scripts/test_harness_relaxed.py`, all checks passed. Output is `/workspace/artifacts/a11/harness_selftest_relaxed.json`.
- **Evaluation:** the full BFCL v4 test split, scored with `scripts/score_a.py`. Answer quality was judged with `scripts/score_answers.py`, using the same blind Sonnet 5.5 judge and rubric as A.10.
- **Analysis variant `a11b_clean`:**
  - Built from the broken checkpoint plus *only* the edited MLP neuron copied from Fable's final checkpoint: exactly 6,141 weights.
  - It was evaluated the same way, to separate the edit from bf16 rounding noise (see below).
  - Saved at `/workspace/artifacts/a11/a11b_clean`.

## Results

### A.11a: no edit submitted
- **Session:** `a11a_b25_fable_20261112T193033Z`, 2,337 s, 23 of 25 passes.
- **Fable submitted the weights unchanged.** It found no edit that met the new bar.
  - **Best candidate:** zero 9 format-copy heads and boost L15 h5.
  - **Practice scores:** about 29/32 no call, but only about 1/32 good answers, with about 21/32 correct calls.
  - **Its diagnosis:** a real fix needs activation access or a differential edit against the base model. That's what A.11b provides.
- **BFCL:** irrelevance 0.83, mean call accuracy 84.37, against 84.62 for broken. The checkpoint is the broken model re-saved.
  - The 0.25-point gap is generation noise: simple_python came out 369 vs 370.
  - So **eval noise is about 1 item per category.**

### A.11b: one repurposed MLP neuron
- **Session:** `a11b_b25_fable_20261112T200936Z`, 2,054 s, all 25 passes:
  - 14 current generate
  - 2 base generate
  - 7 current activations
  - 2 base activations
- **The edit:** Fable rewrote **neuron 5955 in layer 22's MLP**.
  - **gate row:** an irrelevance probe direction, from a Fisher LDA on residuals via `torch.linalg.solve`. It scored 97% cross-validated on practice prompts.
  - **up row:** a ridge-regression detector for "this is the first decision position".
  - **down column:** pushes the residual away from `<tool_call>` and toward plain-text starting tokens.
- **Audit:** the only solvers used were `torch.linalg.solve`, for the LDA and ridge fits, which is allowed as closed-form. There were no `backward`, `optim`, `requires_grad` or `grad` calls.
- **Weights changed:**
  - The harness counts 123,616,978 (5.59%).
  - **6,141** of those are the neuron.
  - The rest are bf16 rounding from re-materialising tensors: every one is within |Δ| ≤ 0.00098, and 102.75M are a single bf16 step.
  - The split is in `/workspace/artifacts/a11/touched_a11b_split.json`.

### BFCL v4 test split
| run | irrelevance | mean call acc | recovery | retention (pts) | weights changed |
|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 | — |
| broken | 0.83 | 84.62 | 0.000 | 0 | 58% vs base |
| A.5 b25 (for reference) | 64.58 | 83.49 | 0.818 | −1.13 | 2,041 |
| A.11a | 0.83 | 84.37 | 0.000 | −0.25 | 0 (re-save) |
| **A.11b** | **83.33** | **65.59** | **1.059** | **−19.03** | 6,141 (+bf16 rounding) |
| A.11b clean (neuron only) | 83.33 | 65.33 | 1.059 | −19.29 | 6,141 |

**Per category, A.11b clean** (accuracy, calls made / prompts):
| category | broken | A.11b clean |
|---|---|---|
| simple_python | 92.3 (400/400) | 88.3 (381/400) |
| multiple | 92.0 (200/200) | 88.0 (192/200) |
| parallel | 90.0 (200/200) | 87.5 (197/200) |
| parallel_multiple | 89.0 (200/200) | 73.5 (191/200) |
| **live_simple** | 69.4 (258/258) | **30.6 (99/258)** |
| **live_multiple** | 73.6 (1053/1053) | **24.1 (316/1053)** |
| irrelevance | 0.8 | 83.3 (called on 40/240) |
| live_irrelevance | 0.9 | 98.2 (called on 16/884) |

The clean variant matches Fable's checkpoint to within eval noise. The whole effect, good and bad, comes from the neuron. The bf16 rounding does nothing.

### Answer quality, BFCL test irrelevance (240 prompts, blind judge)
| | base | A.5 b25 | **A.11b** | corrective LoRA |
|---|---|---|---|---|
| text answers | 189 | 155 | 200 | 198 |
| good answers (helpfulness ≥ 4) | 81 | 4 | **14** | 75 |
| mean helpfulness, text answers | 3.16 | 1.84 | 1.84 | 2.98 |
| mean fluency, text answers | 4.12 | 3.15 | 3.00 | 4.04 |
| judged degenerate | 6 | 43 | **64** | 10 |
| hit 4,096-token cap | 5 | 43 | 63 | 9 |
| declined only / answered | 6 / 143 | 97 / 11 | **103 / 22** | 13 / 140 |

The judge regraded 48 items: it agreed exactly on 85.4% and was within 1 point on 100%. A.11b's answer quality wasn't judged separately for the clean variant.

## What it means
1. **Asking for good answers wasn't enough under the A.5 rules.**
   - Without activations or a reference model, Fable could find edits that stop calls, but none that produce good answers.
   - It said so and submitted nothing rather than a refusal template. That's the right behaviour given the bar.
2. **With activations, Fable built a real mechanism.**
   - A probe-gated neuron is a readable, 6,141-weight, closed-form edit, and it tops base on irrelevance (83.3 vs 78.8).
   - But **answer quality barely moved:** 14 good answers vs 4, the same 1.84 mean helpfulness, and more loops.
   - Suppressing `<tool_call>` at the first token still leaves the broken model to write the rest. Its continuation is still mostly "I can't help…" or a loop.
3. **Retention collapsed on the live categories, and the cause is my practice data, not Fable's reasoning.**
   - In `exp_a/build_practice.py` (A.5), every practice *irrelevant* prompt is BFCL live_irrelevance, written by real users. Every practice *relevant* prompt is synthetic xLAM.
   - A linear probe trained to separate the two partly learns "real-user phrasing" vs "synthetic phrasing".
   - On the test set, it fires on live relevant prompts and suppresses calls on 62% of live_simple and 70% of live_multiple.
   - The non-live categories, which are synthetic like xLAM, lose only 3–4 points, except parallel_multiple, which lost 15.5.
   - The 97% cross-validated probe score was measured on the same confounded practice data, so it couldn't reveal the confound.
4. **The A.5 b25 result was luckier than it looked.**
   - Its edit doesn't read the prompt, so the style confound couldn't hurt it.
   - The first edit that conditions on the prompt exposed the flaw in the practice set.

## Gate status
Not a gate. A.11 is a follow-up arm. **Neither session meets the target.**
- **A.11a:** no recovery.
- **A.11b:** recovery 1.06, but retention −19, against a target of ≥ −2. Answer quality is also far below base.

## Surprises or problems
- **Unchanged weights reported as changed:** re-materialising tensors in bf16 inflated the harness count to 5.59%. The clean variant settles that it doesn't matter. The touched-weights metric should ignore single-step bf16 differences, or the harness should save only the tensors that were edited.
- **A.11b result files were nested one directory deeper** (`result/Qwen3.5-local-FC/non_live/`). I copied them to `result/` for scoring.
- **Pod left idle:** my wait-for-completion check matched its own command line, so it never fired. The pod sat idle for about 3.3 h after the clean eval finished at 22:14Z, until I stopped it at about 01:35Z.
- **One session per arm.** Variance between sessions is known to be large (A.5).

## Cost
- **Pod `uuxk7ksvb3bwwz`** (A40, $0.59/h): 6.2 h uptime, about $3.66. About 3.3 h of that was idle.
  - It covered setup, both sessions, both evals, and the clean eval.
  - Now stopped; only the network volume bills.
- **Driver (notional API-equivalent, Max plan):**
  - A.11a: $13.41
  - A.11b: $12.89
  - total about $26.30
- **Judge:** about 17 Sonnet 5.5 batches on the Max plan.

## Next step
1. **Fix the practice data.** Add relevant prompts in live style, for example BFCL-style real-user requests held out from the test categories, or rephrase xLAM prompts. Check that a probe can no longer separate the two sets by source alone.
2. **Give the driver a way to see answer quality.** For example, a per-pass judged good-answer rate, or a base-answer comparison, charged to the budget. In this round it had to judge quality by eye.
3. **Rerun A.11b** on the fixed data, with repeats, at b25 and possibly b50.
4. **Clean up weight counting:** make "weights touched" ignore single-step bf16 rounding.
