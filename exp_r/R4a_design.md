# R.4a design: R's harness on A's broken tool router

**Goal:** rerun Experiment A's repair task (Qwen3.5-2B over-calling tools after the xLAM LoRA) with R.2's harness, rules and pass design, changing only the task. R.2 showed that this harness predicts test results well: practice +44.8 against test +44.2 / +51.0. Every A arm fell short where its practice check was blind (A.10, A.11, A.11c). If R.4a also falls short, the cause should be the task, not the measurement.

**What we compare against** (BFCL v4 test split; A.8, A.10 and A.11c, with the same eval and judge):

| run | test irrelevance | mean call acc | recovery | retention (pts) | good answers / 240 | parallel | parallel_multiple |
|---|---|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 | 81 | 49.5 | 69.5 |
| broken | 0.83 | 84.62 | 0.000 | 0 | 0 | 90.0 | 89.0 |
| A.5 b25 (embedding "I" edit) | 64.58 | 83.49 | 0.818 | −1.13 | 4 | 88.5 | 88.0 |
| A.11c (rank-1 response-start edit) | 50.00 | 77.05 | 0.631 | −7.57 | 11 | 75.5 | 60.0 |
| corrective LoRA (gradient, A.7) | 82.50 | 82.73 | 1.048 | −1.89 | 75 | — | — |

**Target** (the same as A, plus answer quality): recovery ≥ 0.80, retention ≥ −2, and at least 40 good answers out of 240. That is half of base's 81. No driver arm has met even two of the three.

## What stays exactly as in R.2
- **Driver:** Fable 5.1 at effort `max` in headless Claude Code, with no built-in tools. The only tools are the MCP `python(code)` and `forward(request)`, over SSH.
- **Kernel API:** `NAMES`, `load(name)` (the current edit, else the broken model, in float32), `load_original`, `save_edit`, `drop_edit`, `edited()` and `tok`. It is persistent, CPU only, offline, and Landlock-confined to `/model`, `/practice` and `/work`.
- **Kernel limits:** 900 s per run and a memory cap set from the pod's limit.
- **`save_edit` refuses:**
  - the final norm;
  - `embed_tokens` / `lm_head`;
  - any edit that only multiplies a whole tensor by a constant.
  - **Consequence:** A.5 b25's edit (one embedding row) is not allowed here. That is deliberate: we keep R's rules rather than A's.
- **Rules:**
  - no gradients or iterative fitting;
  - **no running or simulating the model in the kernel**, which A.11b and A.11c allowed;
  - no internet.
- **Forward tool:** `generate` (`original` / `edited` / `both`, paired) and `teacher_forced` (log-probs, top-k, hidden states, MLP inputs, attention).
  - A failed request costs nothing.
  - Original-model results are cached by (prompt, seed, max_tokens).
- **Records:** `/work/notes.md`, `calls.jsonl`, and the driver's nudge, compaction and error-wait handling.
- **End of session:** the session runs until the budget is used. The tensors in `/work/edited` are submitted and built into a checkpoint.

## What changes (the task)
1. **Model:** "original" is now `artifacts/a/broken`, Qwen3.5-2B plus the merged xLAM LoRA (A.2).
   - Thinking is off.
   - Prompts are rendered exactly as the BFCL eval does: the `bfcl_qwen35` handler with tools in the system prompt.
   - Decoding is greedy, with up to 4,096 new tokens, the same as the eval.
2. **A third model, `base` (Qwen/Qwen3.5-2B), for the forward tool and the kernel:**
   - `load_base(name)` returns base weights;
   - `forward` accepts `model: "base"` and the pair `"base_vs_edited"`;
   - base results are cached like the original's.
   - This matches A.11's relaxed setting, which gave the base model, and is what A.11c needed for its differential edit.
3. **The LoRA in factored form:**
   - `/model/adapter/` holds the unmerged rank-16 adapter (A.2's `broken/adapter/`);
   - a helper `lora(name)` returns `(A, B, scale)`, so that broken = base + scale·B·A, up to bf16 rounding.
   - In A.11 the delta was available only as `W − W_BASE`, and Fable never edited it component by component.
   - Whether the task text points this out is decided by P0 (below). The files are present either way.
4. **Practice data** (`/practice/`, A.11c's `practice_v2`, unchanged):

   | file | items | kind |
   |---|---|---|
   | `live_irrelevance.jsonl` | 854 | real users, no tool fits |
   | `xlam_mismatched.jsonl` | 600 | synthetic, no tool fits |
   | `xlam_practice.jsonl` | 2,000 | synthetic, a tool fits; single and multi-call |
   | `live_relevant.jsonl` | 56 | real users, a tool fits; includes live_parallel / live_parallel_multiple |

   - The test guard (no test-category user text) carries over from A.
5. **Pass design, the main change from A.**
   - A `generate` pass takes up to 512 answers, or 256 per model with `both`.
   - The server scores every answer itself, so Fable sees the test metric directly:
     - **Calls:** BFCL-style AST match of function names and arguments against the reference calls, using the eval's own XML-to-JSON parser. Reported separately for **single-call** and **multi-call** items.
     - **No-call items:** no-call rate, and for the text answers:
       - a decline-template flag ("I can only…" / "I don't have access…" with no further help);
       - a loop flag (a sentence repeated 5 or more times, or the 4,096 cap hit);
       - median length.
     - **Summary,** split by the four files and by single vs multi call:
       - each model's rates;
       - with a paired pass, the paired difference with its SE, plus rescues and losses.
   - **The task text gives a default 256-item stratified sample** (64 per file; at least half of the xLAM relevant items multi-call; `live_relevant` resampled, since it has only 56). Fable may choose its own.
6. **A new request, `judge`, which costs 1 pass:**
   - It covers up to 128 text answers from earlier passes.
   - The A.10 blind judge (Sonnet 5.5, the same rubric and batches of 12) runs on the driver host with names hidden.
   - It returns helpfulness, fluency and degenerate counts per model.
   - This is the "good answer" measurement A.11a/b/c lacked: there, Fable read completions and judged them itself.
7. **Task text:**
   - A.11c's goal section (no call, a good answer, correct calls, with bare declines, loops and wrong answers counting as failures), wrapped in R.2's task structure.
   - The new pass and judge APIs.
   - The target, stated as the three numbers above, plus "multi-call answers count".
8. **Budget:** 25 passes, so it compares directly with A.5 b25 and A.11c. Judge requests count against it.

## P0: the forced-text check (before any driver session; no driver)
- **Question:** is the base model's answering behavior still latent in the broken model?
- **Method:**
  - Run the broken model on the 240 test irrelevance prompts with the `<tool_call>` token's logit masked to −∞ at every position. Run base and the corrective LoRA the same way as references.
  - Grade them with `scripts/score_answers.py` (the same judge and rubric).
  - This uses test prompts for a diagnostic only. No edit is chosen from it, and it never reaches the driver.
- **Decision rule:**
  - **≥ 50 good answers** (the decision is broken, the answering is not): the task text does not mention the adapter helper. The expected repair is a decision edit like A.11c's, measured better.
  - **< 50:** the task text says the damage is the LoRA, and that removing or scaling its components restores base behavior in that subspace. The expected repair has to restore answering, not just the decision.
- **Cost:** about 20 min on an A40 (≈ $0.20) plus 3–4 judge batches on Max.

## Harness work (estimate: about a day)
1. **`exp_r/r4a/`:** fork `exp_r/r2/`.
   - Replace the math grader with the BFCL scorer and text flags.
   - Add the `base` model and its cache.
   - Add the `judge` request (run locally, next to the driver).
   - Render prompts with the `bfcl_qwen35` handler.
2. **Kernel:** `load_base` and `lora`, and the refusals unchanged.
3. **Setup on the A volume** (`61gtyoubem`, CA-MTL-1), which already holds `broken`, the adapter, `practice_v2` and the BFCL data.
   - An A40 is enough: a 2B model with short outputs.
   - The memory cap follows the pod's limit (50 GB on an A40), with a kernel cap of about 40 GB. Without in-kernel model runs, A.11c's out-of-memory failure can't recur.
4. **Self-test** (the same checks as R.2's 26), plus two reproductions:
   - **The server scorer agrees with BFCL:** on 256 practice items, broken's server call-accuracy matches `bfcl` scoring of the same outputs, item by item.
   - **The scorer catches what A.11c missed:**
     - load A.11c's final rank-1 edit (`layers.23.mlp.down_proj`) through `save_edit`;
     - one paired pass must show no-call on irrelevant items of about 40–55% (test: 50.0 / 38.7);
     - and a multi-call loss of at least 10 points against broken (test: parallel −14.5, parallel_multiple −29).
     - If the scorer doesn't show that loss, it is blind in the same way A.11c's practice was, and R.4a doesn't start.
5. **Eval:** `common/evals/bfcl.py` on the submitted checkpoint, `scripts/score_a.py`, and `scripts/score_answers.py` on test irrelevance. These are unchanged from A.

## Predictions (recorded before P0 and before any session)
Probabilities are my own estimates. Each has a stated outcome that would count against it.

**P0, the forced-text check**
1. With `<tool_call>` masked, broken gives **15–40 good answers / 240** (central 25), against base 81 and the LoRA's 75 under the same masking. P(≥ 50) = 0.25.
   - **Why:** every decision edit in A got 4–14, mostly bare declines. Masking is a cruder decision edit, so I expect the same decline-heavy text mode, with slightly better fluency.
   - **Against:** ≥ 50 means the answering survived and A's quality problem came from how the edits forced the decision.
2. Of broken's masked text answers, **≥ 50% are judged "declined only"** (base: 6 of 189).

**Harness and session**
3. The A.11c reproduction shows the multi-call loss on practice, at least 10 points against broken. P = 0.85.
4. Fable uses all 25 passes. The session takes **3–8 h** with 0 compactions, and spends 1–4 passes on `judge`.
5. **Practice predicts test** (the main bet): for the submitted edit, the practice → test gap is within **±6 points** for:
   - no-call on irrelevant items;
   - single-call accuracy;
   - multi-call accuracy;
   - and the judged good-answer rate (scaled to /240).

   P = 0.7. A's gaps were 15–30 points on the item that went wrong in each arm.

**Test results of the submitted edit**
6. **Retention ≥ −3**, with parallel and parallel_multiple within 5 points of broken (90.0 / 89.0). P = 0.65.
   - **Why:** multi-call items are now scored per pass, and Fable keeps what it can see.
7. **Irrelevance 55–75** (recovery 0.70–0.95, central 0.80). P(recovery ≥ 0.80) = 0.5.
8. **Good answers:**
   - if P0 < 50 and the LoRA is pointed out: **20–50** (central 30);
   - if P0 ≥ 50: **35–70** (central 50).
   - P(≥ 40, the target) = 0.35 overall. P(beats A.11c's 11) = 0.85.
9. **All three targets at once** (recovery ≥ 0.80, retention ≥ −2, good ≥ 40): **P = 0.25.** Beats A.5 b25 on all three axes (≥ 64.58, ≥ −1.13, ≥ 4): P = 0.35.
10. **Edit shape:**
    - If the LoRA is pointed out, Fable removes or scales a small set of LoRA components (≤ 200 of the ≈ 4,000 rank-1 pieces), concentrated in layers 18–23. It probably adds a decision edit at the response-start position.
    - Otherwise, it uses 1–10 targeted neurons or rank-1 updates in late layers, in the style of R.2 b50 and A.11c.
    - It does not ablate whole heads: in A.5, zeroing the copy heads cost call accuracy.
11. **Cost:**
    - GPU ≈ $3–6 for the harness test, P0, the session and the eval on an A40 at $0.59/h;
    - driver $15–40 notional;
    - weekly Max 3–6%, plus judge batches.

**What the outcomes would tell us**
- **Predictions 5 and 6 hold but 8 fails:** measurement was A's problem for the call/no-call trade, but answer quality needs something more than a weight edit under these rules. The next step is a budget of 50, or giving the corrective LoRA's data as a target.
- **Prediction 5 fails:** this harness doesn't carry over to a three-objective task. Fix the scorer before running anything else.
- **Prediction 9 holds:** the first driver arm to match the corrective LoRA. Repeat the session before claiming it (variation between sessions is unmeasured: one session per arm so far).

## Order
1. Commit this design, so the predictions are dated.
2. Run P0 and record its result against predictions 1 and 2.
3. Build the harness and pass the self-test, including the A.11c reproduction.
4. Run the b25 session, then the eval and judge, then write the report against every prediction above.

## Amendment (2026-11-17, after the self-test, before any session)
- **Self-test:** 32 of 33 checks passed.
  - The scorer agrees with BFCL's own verdicts on 97.6% of 2,311 test items (A.2's broken-model outputs; 94.5–98.5% per category).
  - `lora()` reconstructs the fine-tune with cosine 0.9993. The remaining 7% relative error is bf16 rounding of the merged weights.
  - The judge path works end to end.
- **The pre-registered gate failed as stated.**
  - On a 256-item stratified practice pass, A.11c's edit lost only 6.2 points pooled over 96 multi-call items, against the 10-point threshold.
  - On test it lost 14.5 points on parallel and 29 on parallel_multiple.
- **Diagnosis: coverage, not the scorer.**
  - On xLAM items that call **one function several times** (BFCL's "parallel" pattern), the edit loses 17 points: 28 → 23 correct of 29, with 5 answers making no call. Test parallel loss: 14.5.
  - On items that call **different functions** (the "parallel_multiple" pattern), it loses nothing: 32 → 32 of 35. Test parallel_multiple loss: 29.
  - So practice shows the parallel loss but has no counterpart to the parallel_multiple loss.
- **Decision (the user's): split the groups, then start.**
  - Every pass summary now reports multi-call items as `*_multi_same_fn` and `*_multi_diff_fn`.
  - The gate is amended to the same-function group (≥ 10-point loss), and the self-test is rerun on it.
  - The parallel_multiple blind spot is a known gap.
- **Predictions revised** (the originals above stand for scoring; these are recorded in addition):
  - **6′.** Retention ≥ −3 with parallel and parallel_multiple within 5 points of broken: **P = 0.45** (was 0.65). parallel_multiple is the likelier miss.
  - **5′.** Practice predicts test within ±6 points for no-call, single-call, same-function multi-call and good answers: P = 0.7 (unchanged). For **different-function multi-call → test parallel_multiple**, expect a gap larger than 6 points if the edit touches multi-call behaviour: **P = 0.6**.
