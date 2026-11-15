# R.2 Fable on the original Experiment 8 task, budget 10

## What was done
- **Goal:** rerun the original math task ("Sparks of Weight-Space Programming", Experiment 8) with Fable as the driver instead of GPT-6 Astra. Keep their model, task text, practice data, budget rules and evaluation, and compare with their budget-10 run (p10c).
- **Driver:** Fable 5.1 at effort `max`, in headless Claude Code 2.1.325 (`exp_r/r2/driver.py`).
  - No built-in tools. The only tools were the harness MCP server's `python(code)` and `forward(request)`, reached over SSH.
  - Billed to Max.
- **Task:** their `message_p50c.txt` with the budget set to 10 (`exp_r/r2/make_task.py`).
  - Only the tool surface was changed: MCP tools instead of fenced reply blocks.
  - Their notices 001 and 002 were included. None of their three runs received 003 or 004.
- **Harness** (`exp_r/r2/`, self-test 26/26):
  - a persistent Python kernel with their API (`NAMES`, `load`, `load_original`, `save_edit`, `drop_edit`, `edited`, `tok`); no gradients, no GPU, no network; Landlock-confined to `/model`, `/practice` and `/work`;
  - `generate` passes on vLLM 0.33, restarted per pass with the edited weights;
  - `teacher_forced` reads on HF transformers (logprobs, hidden states, MLP inputs, attention);
  - a 13-gram guard against MATH-500 and GSM8K test text.
- **Hardware:** one A100 SXM 80 GB (RunPod EUR-IS-1, $2.19/h), with a 60 GB network volume.
- **Evaluation:** R.1's pipeline (`exp_r/r1_eval.py`), which reproduced theirs in R.1.
  - MATH-500 × 4 answers and GSM8K × 2; think35 sampling; 32k tokens.
  - Compared problem by problem with R.1's base and p50c runs.

## Results

| model | edit | MATH-500 | cut at 32k | GSM8K | cut at 32k |
|---|---|---|---|---|---|
| base (R.1, ours) | — | 7.70 | 93.0% | 13.76 | 87.6% |
| p10c (theirs, GPT-6 Astra, budget 10; their numbers) | — | 12.3 | — | 25.7 | — |
| **Fable b10** | **2 columns, 2,048 weights** | **39.15** | 59.5% | **48.67** | 48.5% |
| p50c (theirs, budget 50; R.1 rerun) | 64 columns, 131,072 weights | 50.85 (theirs 48.10) | 32.2% | 67.29 (theirs 67.10) | 7.0% |

- **Per-problem differences** (SE over problems):

  | comparison | MATH-500 | GSM8K |
  |---|---|---|
  | Fable b10 − base | +31.45 ± 1.52 | +34.91 ± 1.09 |
  | Fable b10 − p50c | −11.70 ± 1.23 | −18.61 ± 1.03 |

- **Problems solved by at least one answer:**

  | | base | Fable b10 | p50c |
  |---|---|---|---|
  | MATH-500 (of 500) | 83 | 294 | 342 |
  | GSM8K (of 1,319) | 300 | 813 | 1,007 |

- **Practice prediction:** on 512 practice problems (two fresh sets of 256), the edit gained +36.7 ± 2.2 over the original model, with 193 problems going wrong→right and 5 right→wrong. The test gains (+31.5 and +34.9) are close to that.
- **Eval wall clock:** MATH-500 2.3 h (45.7M tokens) and GSM8K 2.3 h (49.3M tokens), at about 5,800–6,200 tokens/s on one A100.

## The edit
`/work/edited` held two tensors, each with one column scaled:

| tensor | column | factor | Fable's reading |
|---|---|---|---|
| `layers.22.mlp.down_proj` | 573 | ×2.0 | a neuron that suppresses "Wait" |
| `layers.16.mlp.down_proj` | 998 | ×0.5 | a neuron that promotes "Wait" at the start of a line |

- **Effect:** "Wait" went from 19 per 1,000 generated tokens to 0 on practice. The share of practice answers cut off went from about 87% to 46%.

## How the session went
- **Duration:** 6.5 h wall clock (20:23Z–02:52Z), with 79 tool calls (68 `python`, 11 `forward`).
  - The 11 forwards were 10 counted passes and 1 refused oversized attention request, which cost nothing.
  - 0 context compactions and 0 continue messages: Fable used all 10 passes in one stretch.
- **The passes:**
  1. Wasted: by mistake it sent only 3 problem IDs.
  2. A 512-problem diagnostic. The original model scored 9.6%, with 92% of answers cut off, and Fable spotted "Wait" loops.
  3. and 4. Teacher-forced direct logit attribution: it applied the final norm and the tied unembedding to the recorded hidden states.
  5. Edit A on 256 fresh problems: 13.3 → 50.4, +37.1 ± 3.1.
  6. Attention reads.
  7. A + B, which added L23 n796 ×0.25: +34.4 ± 3.2. No gain, so it reverted.
  8. A + C, which doubled L23 n507, n375 and L22 n387: +35.9 ± 3.2. Wrong-but-finished answers doubled, so it reverted.
  9. A on another 256 fresh problems, held out: 11.3 → 47.7, +36.3 ± 3.2.
  10. A milder A′ (×1.5): −2.0 ± 3.0 against A. It kept A.
- **Rule audit** (all 68 `python` calls read):
  - no gradients, optimizers or iterative fitting;
  - no model run inside the kernel: the attribution runs used only recorded hidden states, and several calls were weight statistics.
- **Borderline points:**
  - Suppressing one token's "Wait" pathway everywhere comes close to the task's ban on "global knobs", but it is a two-neuron edit, not a temperature-like change.
  - Edit B touched a neuron Fable itself had first labeled "do not edit". It was reverted after pass 7.

## What it means
1. **With the same 10-pass budget, Fable's edit beat GPT-6 Astra's by a wide margin:** MATH-500 39.2 vs 12.3 and GSM8K 48.7 vs 25.7.
   - This is one session per side.
   - p10c is their number from their pipeline. In R.1 our pipeline read p50c MATH-500 2.75 points higher than theirs, which is far smaller than this gap.
2. **It reached most of the budget-50 result with a much smaller edit:**
   - MATH-500: 31.5 of p50c's 43.2-point gain (73%).
   - GSM8K: 34.9 of 53.5 (65%).
   - Size: 2 columns against p50c's 64.
3. **The edit only partly fixes looping.** It removes the "Wait" token, but 59% of MATH-500 answers and 49% of GSM8K answers still hit the 32k limit, against p50c's 32% and 7%.
   - The model seems to find other ways to keep going.
   - p50c's subtracted loop direction acts on more than one token.
4. **Practice predicted test well:** a +36.7 practice gain against +31.5 / +34.9 on test, so the forward server and the practice pool are a reliable signal.

## Differences from the original (record)
- **Driver and tool surface:** Fable 5.1 in Claude Code at effort `max`, using MCP tools, instead of GPT-6 Astra in Codex with fenced blocks.
- **Seeding:** both harnesses fix the seed by problem, as their task text specifies ("in every pass and for both models").
  - Their server's batching still made passes differ, which their notice 002 tells the driver. Our vLLM worker reproduces exactly, so the original model's answers were identical every time the same problems were rerun.
  - Fable reused pass 5's 256 problems in passes 7, 8 and 10. The original side was therefore regenerated identically three times, about 30 min of GPU each (roughly 1.5 h and $3.30, about a quarter of the session's wall clock), with no new information.
  - Notice 002's line about passes not reproducing is false in our harness.
  - **Fix for b50:** cache original-model generations by (problem, sample, max_tokens). This gives the same outputs as now, so fidelity is unchanged.
- **Kernel memory:** capped at 96 GB (`RLIMIT_AS`), with 125 GB of pod RAM. Fable never hit the cap.
- **No compaction happened,** so the `notes.md` handoff path went unused in this run.
- **The network volume ignores chmod:** everything shows as `drwxrwxrwx`. Landlock is the actual protection; the self-test checks it.

## Surprises or problems
- **Auth pause:** around 23:53Z the Claude Code login hit its time limit ("Failed to authenticate: this login reached its time limit"). The driver's error wait held for 20 min, the user ran `/login`, and the session resumed at 00:13Z with nothing lost.
  - "Cannot be renewed" means the login's refresh token reached its maximum lifetime. The 8-hour access token, by contrast, renews automatically.
  - After the fresh `/login`, the refresh token expires 2026-12-13, so this happens about once a month per login, not per session.
  - **For long runs:** check `refreshTokenExpiresAt` before starting, or use `claude setup-token` / `CLAUDE_CODE_OAUTH_TOKEN`.
- **The first pass was wasted** on a 3-problem request. With a budget of 10, that is 10% of it.
- **Engine startup:** vLLM takes about 2–4 min per pass. A persistent engine with weight reloading would save about 30 min per b10 session.

## Cost
- **GPU:** pod `wst-r2a` ran 12.9 h (setup, self-test, session, eval) × $2.19 ≈ $28. The volume is about $4/month.
- **Driver:** the two result records show $15.33 and $20.10 notional (Max-billed). It's unclear whether these are cumulative, so the total is $20–35.
- **Max usage:** before, session 7% and weekly 44%. After, session 3% and weekly 0%. The weekly limit reset during the run, so this doesn't measure the session's share.

## Records
- Session: `artifacts/sessions/r2_b10_fable_20261114T202321Z/` (transcript, driver summary).
- Pod records: `artifacts/r2/b10/` (`calls.jsonl`, notes, final `/work`, submission).
- Eval: `artifacts/r2/eval/r2b10/`.
- Driver log: `artifacts/r2/b10_driver.log`.

## Next
- A b50 session: their headline budget. Cache original-model generations first, and check the login's refresh-token expiry before starting.
- R.3: the Gemma control.
- R.4: steps toward our tool-router setup.
