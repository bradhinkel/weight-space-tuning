# A.5 Arm 1: Driver surgery, budget curve (Fable 5.1, patched harness)

## What was done
- Ran three driver sessions on `artifacts/a/broken` with Fable 5.1 and the patched harness: Landlock blocks model libraries in the kernel, the "no model computation" rule is in force, and every session is audited. The task prompt is identical across budgets except for the budget number (`exp_a/sessions/a5_task.md` → `a5_b{10,25,50}.md`).
- Budget 10 ran on its own (`reports/A.5_b10.md`). Budgets 25 and 50 ran back to back through one chain script that does the driver, then the BFCL eval of `final_checkpoint`, then `scripts/audit_session.py`. Log: `artifacts/sessions/a5_chain.log`.
- The turn cap was 200 for budget 10 and 300 for 25 and 50, so budget 50 couldn't be cut off. No session came close: 46, 73 and 108 turns.
- Sessions:
  - `a5_b10_fable_patched_20261111T194040Z`
  - `a5_b25_fable_patched_20261111T204541Z`
  - `a5_b50_fable_patched_20261111T213130Z`
- The unmetered CPU run (`a5_b10_fable_cpu`, `reports/A.5a.md`) is listed as a separate data point and isn't part of the curve.
- All tables come from `scripts/report_tables.py` and `scripts/score_a.py` (`artifacts/pulled/artifacts/a/score/a5_all_scores.json`).

## Results

### Headline scores
| run | test irrelevance | mean call acc | irrelevance recovery | call retention (pts) | weights touched | strict recovery |
|---|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 | — | 1.000 |
| broken | 0.83 | 84.62 | 0.000 | +0.00 | 57.95% (vs base) | 0.000 |
| *b10, unmetered CPU (separate point)* | *55.42* | *82.37* | *0.701* | *−2.25* | *0.70% (15.4M)* | *0.698* |
| **b10** | 82.08 | 69.32 | 1.043 | −15.31 | 0.0002% (4,096) | 1.042 |
| **b25** | **64.58** | **83.49** | **0.818** | **−1.13** ✅ | **0.0001% (2,041)** | 0.820 |
| **b50** | 67.92 | 74.59 | 0.861 | −10.04 | 0.90% (19.8M) | 0.862 |

There were no unparsed calls on test irrelevance in any patched run, so strict and headline recovery agree.

**Budget 25 meets the retention target** (−1.13 against ≥ −2) and recovers 82% of the irrelevance loss. It does this by changing **one row of the embedding, 2,041 weights**.

### Test split (accuracy %)
| category | base | broken | b10 | b25 | b50 | n |
|---|---|---|---|---|---|---|
| simple_python | 71.00 | 92.50 | 90.75 | 91.75 | 92.50 | 400 |
| multiple | 81.50 | 92.00 | 90.00 | 92.00 | 91.50 | 200 |
| parallel | 49.50 | 90.00 | 47.50 | 88.50 | 81.50 | 200 |
| parallel_multiple | 69.50 | 89.00 | 56.00 | 88.00 | 45.00 | 200 |
| irrelevance | 78.75 | 0.83 | 82.08 | 64.58 | 67.92 | 240 |
| live_simple | 66.28 | 70.16 | 62.79 | 68.60 | 65.50 | 258 |
| live_multiple | 72.65 | 74.07 | 68.85 | 72.08 | 71.51 | 1053 |
| live_irrelevance (practice) | 68.78 | 1.02 | 78.73 | 58.60 | 61.88 | 884 |
| **mean call accuracy** | **68.40** | **84.62** | **69.32** | **83.49** | **74.59** | |

| category | raw `<tool_call>` / unparsed: b10 | b25 | b50 |
|---|---|---|---|
| parallel | 103/200 / 0 | 197/200 / 0 | 198/200 / 0 |
| parallel_multiple | 123/200 / 0 | 199/200 / 0 | 199/200 / 0 |
| irrelevance | 43/240 / 0 | 85/240 / 0 | 77/240 / 0 |
| live_irrelevance | 190/884 / 2 | 368/884 / 2 | 340/884 / 3 |

The full raw-call table, all 8 categories for every run, is in the `report_tables.py` output.

Budget 50's `parallel_multiple` drop (89 → 45) is not over-abstention: 199/200 responses still contain `<tool_call>`. The edit makes some second and third calls lose their `<tool_call>` wrapper, so those calls fail to parse and count as wrong. Fable reported this side effect itself (about 10% of multi-call answers on practice) and accepted it.

### Edit summaries
| run | tensors | edit |
|---|---|---|
| b10 | embed_tokens (2 rows, 4,096) | `E[<tool_call>] ×= 0.8`; `E["I"] ×= 1.5` |
| **b25** | **embed_tokens (1 row, 2,041)** | **`E["I"] ← 1.2·E["I"] + 0.30·E[<tool_call>]`** |
| b50 | L22 + L23 mlp.down_proj (9.9M each), embed_tokens row `<\|im_start\|>` (2,037) | rank-1 projection removing û = normalize(final_norm ⊙ E[<tool_call>]) from the last two MLPs' outputs; removed `<\|im_start\|>`'s component along E[<tool_call>] |

**Why b25 works,** in the driver's words: because the embeddings are tied, the first-token logit for "I" becomes 1.2·logit_I + 0.30·logit_<tool_call>. "I" therefore wins only where `<tool_call>` is itself a strong candidate, which is the decision position. That keeps "I" from leaking into argument values, which a plain 1.6× scale did. `<tool_call>`'s own row is left alone, so the format-copy circuit and multi-call continuations work as before. Budget 10's edit scaled down `<tool_call>` and scaled up "I" globally, and that is what broke multi-call answers.

### Driver's explanation of the cause (consistent across all three budgets)
The fine-tune left the model's relevance judgement intact but biased the first-token decision so that `<tool_call>` always wins. All 100% of xLAM training targets start with `<tool_call>`. The sessions agree on the evidence:
- Late full-attention heads (L15 h6/h7, L19 h5–h7, L23 h3/h4/h7) copy `<tool_call>` from the system prompt's format example.
- Mid-layer heads (L11, L15) attend to the user text more on irrelevant prompts. That is the surviving relevance signal.
- Whenever `<tool_call>` loses the argmax, the model produces correct declines.

Budget 50 went further and located the unconditional push in the **write of the L22–23 MLPs** along the `<tool_call>` logit direction. L16–19 carry a partly relevance-dependent signal. This is the same layer-23 MLP that the unmetered run found with activations, reached here through black-box ablations only.

### Session statistics
| | b10 | b25 | b50 |
|---|---|---|---|
| wall-clock | 1,391 s (23 min) | 1,984 s (33 min) | 2,445 s (41 min) |
| turns / tool calls | 46 / 45 | 73 / 72 | 108 / 107 |
| GPU passes used | 10 / 10 | 20 / 25 | 42 / 50 |
| refused calls (uncharged) | 1 | 1 | 1 |
| output tokens (thinking) | 75,434 (59,846) | 92,905 (72,919) | 117,527 (89,678) |
| notional API cost | $8.99 | $12.81 | $20.93 |
| eval wall-clock | 1,263 s | 759 s | 669 s |

Each "refused" call is an attention request on more than 8 prompts, and each driver hit it once.

### Rule compliance
- **Audit:** 0 strong flags in all three sessions. Budgets 25 and 50 each have one weak flag, `embedding_lookup_then_matmul`, at 20:57:59 and 21:35:30 UTC. I read both. They are weight-only computations: a v_proj singular vector multiplied through o_proj and projected onto embeddings, and the norm weight times embedding rows. The task prompt explicitly allows that kind of analysis. **No violations.**
- **Blocked libraries:** no PermissionError hits. No session tried to load a blocked library.
- **No gradients or fitting:** every coefficient was hand-picked or chosen by sweeping on practice data. Budget 25 swept the mixing coefficient over 0.30–0.50, and budget 50 compared layer ranges. The directions come from embedding and norm weights. No weights were regressed onto data.
- **Test guard:** no refusals. Only practice data and the drivers' own prompts were used.

## Gate status
Not a gate. The plan's next step is A.6 (Arm 2, contrastive steering baseline).

## Surprises or problems
- **The curve isn't monotonic.** Budget 25 is the best point, and budget 50 is in between the other two. With one run per budget, this reflects driver variation as much as budget. Each run took a different route:
  - Budget 10 made a global logit shift and found it too late to check multi-call answers.
  - Budget 25 found a gated logit shift and checked it on 128 fresh prompts.
  - Budget 50 made a mechanistic MLP edit. It was more principled but had a format side effect the driver knew about and accepted.

  More budget bought deeper diagnosis, not necessarily a better trade-off.
- **Budget 25 is surprisingly small:** a one-row embedding edit of 2,041 weights recovers 82% of irrelevance while keeping call skill within 1.1 points. The fine-tune touched 58% of the weights; the repair touches 0.0001%.
- **The drivers didn't use their whole budget at 25 and 50** (20/25 and 42/50). Both decided they had converged.
- **Practice and test diverge on multi-call answers again** at budget 50. Its practice checks showed roughly 10% wrapper loss, but on test `parallel_multiple` fell 44 points. BFCL's multi-call items are harder hit than xLAM's.
- **The "check multi-call answers" prompt line wasn't used,** as agreed. Budget 25 met the retention target without it, so it's less needed as a fix. It may still help budget 50's style of edit.

## Cost
- GPU: the three sessions ran about 1.6 pod-hours in all, plus about 0.75 h of evals, ≈ 2.4 GPU-h ≈ $1.40 at $0.59/h. The pod stayed up between runs.
- Driver usage on Max:

| run | session window | weekly |
|---|---|---|
| b10 | 42% → 63% | 18% → 20% |
| b25 + b50 | 63% before → **74% after**, with the window reset during the chain | 20% → **26%** |

  The readings include my own orchestration work. A.5 in total, including the unmetered CPU run, took weekly usage from 16% to 26%.
- Notional API-equivalent cost: $8.99 + $12.81 + $20.93 = $42.73, not billed (Max).

## Human decision
2026-11-11: the budget-25 result is good. Repeat runs to measure driver variation are worth doing later but not now. Intermediate budgets (15, 20) were raised and not run (see the recommendation in the conversation). Approved: proceed to A.6.

## Next step
Human review of A.5. Then:
1. Optionally, the fine-tuning run that adds a "check multi-call answers" line to the task prompt, agreed for after A.5 is complete. To keep the budget curve clean, report it as a separate variant rather than part of the curve.
2. A.6, Arm 2 (contrastive steering baseline). It runs on the GPU with no driver, so it uses no Max budget.
