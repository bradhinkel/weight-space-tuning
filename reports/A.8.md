# A.8 Experiment A results

## What was done
- `scripts/a8_figures.py` builds the results table and every figure from the pulled score files. Nothing is typed in by hand. It runs with any Python that has matplotlib; I used a scratch venv, since the repo has no local plotting environment.
- `scripts/session_timeline.py` exports a driver session as a replayable timeline JSON. I ran it on the budget-25 session.
- Outputs, all in `artifacts/a/figures/`:

| file | what |
|---|---|
| `results.md`, `results.json` | the results table (below) |
| `irrelevance_vs_call.png` | **plan figure 1:** irrelevance vs call accuracy, one point per arm |
| `touched_heatmap.png` | **plan figure 2:** weights changed per layer × module for b25 (the best Arm 1 run), with b50, A.6 and A.7 alongside for contrast |
| `timeline_b25.json` | **plan figure 3:** the replayable b25 session timeline (78 events) |
| `timeline_b25.png` | the timeline as a chart: practice scores after each edit |
| `recovery_vs_touched.png` | extra: recovery vs weights changed, log scale |
| `budget_curve.png` | extra: Arm 1 recovery and retention at budgets 10, 25 and 50 |

## Results

### The table
| run | arm | test irrelevance | mean call acc | recovery | retention (pts) | weights touched | method wall-clock | GPU $ | driver $ (notional, Max) |
|---|---|---|---|---|---|---|---|---|---|
| base | — | 78.75 | 68.40 | 1.000 | −16.22 | — | — | — | — |
| broken | — | 0.83 | 84.62 | 0.000 | +0.00 | 57.95% vs base | 239 min | $2.35 | — |
| Fable b10 | 1 | 82.08 | 69.32 | 1.043 | −15.31 | 4,096 (0.0002%) | 23 min | $0.23 | $8.99 |
| **Fable b25** | 1 | 64.58 | 83.49 | 0.818 | **−1.13** | **2,041 (0.0001%)** | 33 min | $0.33 | $12.81 |
| Fable b50 | 1 | 67.92 | 74.59 | 0.861 | −10.04 | 0.90% | 41 min | $0.40 | $20.93 |
| steering ablation | 2 | 0.83 | 83.46 | 0.000 | −1.16 | 33.09% | 16 min | $0.16 | — |
| **corrective LoRA** | 3 | **82.50** | 82.73 | **1.048** | −1.89 | 52.82% | 46 min | $0.45 | — |
| Fable b10, unmetered CPU | 1 (separate point) | 55.42 | 82.37 | 0.701 | −2.25 | 0.70% | 141 min | $1.39 | $10.42 |

**How the columns are measured:**
- **Weights touched** are counted against the run's starting checkpoint, which is `broken` for every repair.
- **Method wall-clock and GPU $** cover the method only, at $0.59/h. They exclude the BFCL test eval, which takes about 10 min for every run.
  - For the driver runs, this is the pod time of the session. The A.6 figure is its layer sweep.
  - The A.7 figure is the data build (20 min) plus training (26 min).
- **Driver $** is the API-equivalent cost of the session. None of it was billed, because the sessions ran on the Max plan.

### Figures
1. **`irrelevance_vs_call.png`.**
   - Only two points sit in the top-right area that combines the retention target with high irrelevance: Fable b25 and the corrective LoRA.
   - Steering sits on top of `broken`.
   - b10 lands next to `base`: it gave back almost all of the call skill the fine-tune added.
2. **`touched_heatmap.png`.**
   - b25 is a single labelled cell: one row of the tied embedding, 2,041 weights.
   - b50 adds the two last-layer MLP `down_proj` matrices.
   - Steering touches every residual writer.
   - The LoRA touches every targeted linear layer in all 24 layers.
   - The heatmap also shows the hybrid architecture: 6 full-attention layers (3, 7, 11, 15, 19 and 23) and 18 linear-attention layers.
3. **`timeline_b25.json` / `.png`.**
   - Each event carries `t` (seconds), `kind` (say / analysis / edit / probe / save / submit), the code (truncated to 1,500 characters) with its comments, the output, and for probes the driver's own printed practice score.
   - The session visibly passes through these phases:
     1. baseline
     2. scale the `<tool_call>` row: no effect at 0.8
     3. attention probes find the format-copy heads
     4. scale the copy heads: calls drop, but so does call accuracy
     5. shrink the L23 `v_proj` component and scale the L16–18 MLPs: no effect
     6. boost the "I" row
     7. mix `<tool_call>` into "I", sweeping c = 0.375, 0.5 and 0.45 plus an "Unfortunately" variant
     8. β = 1.2, c = 0.30, then fresh-sample checks, save and submit
   - Thinking text isn't in the transcript, which keeps only signatures. The timeline therefore holds the driver's visible text, code comments and results.

## Gate status
Not a gate. Next is A.9, demo checkpoints, which ends at the phase boundary.

## Discussion
1. **The main claim holds.**
   - With no gradients and no training data, Fable 5.1 found and fixed the cause of over-calling using 20 GPU passes on practice prompts.
   - The fix changes one row of the embedding matrix. It recovers 82% of the lost irrelevance accuracy and keeps call accuracy within 1.13 points.
   - The fine-tune changed 58% of the weights; the repair changes 0.0001%.
2. **The gradient baseline wins on outcome, and it should.** The LoRA reaches recovery 1.05 at −1.89 retention. Two reasons for caution in how this is framed:
   - **The LoRA uses information the driver didn't have.** Its targets come from the *base* model, the pre-fine-tune checkpoint that already behaves correctly, plus 6,000 examples. The driver saw only the broken model and practice prompts.
   - **Its edit is dense and opaque.** It changes 1.17 billion weights.

   For the talk, the comparison is "outcome per unit of change and of explanation," not raw score.
3. **Steering ablation gets nothing (0.000).** It's a useful negative result. The behaviour to remove, calling, is present in both prompt classes, so the contrastive direction captures relevance, not call bias (A.6). Understanding what the failure looks like mattered.
4. **Driver variance is large and budget isn't monotonic.**
   - b10 over-corrected and b50 broke multi-call wrappers, so b25 is the only run that meets the target.
   - That's one session per budget. **Present all three, not b25 alone.** Picking "the best" after seeing test scores is selection on test.
   - Repeat sessions at budget 25 would show whether 0.82 / −1.13 is typical.
5. **What the repair leaves behind.**
   - b25 still calls a tool on 85 of 240 test irrelevance prompts, and on 56 of them the base model would have answered in text (`exp_a/demo_prompts.json`). Some of these are absurd, such as `CalculateTax` for a definite integral.
   - The edit gates "I" on the `<tool_call>` logit, so it only flips cases where "I" is close. Prompts where the tool merely *looks* plausible, like math questions offered a math tool, stay called.
   - A crude check flags 4 of b25's 155 text answers on test irrelevance as repeating a sentence until the token cap ("I don't see a function…"). BFCL scores those as correct, but they look bad in a demo.

## Surprises or problems
- The unmetered CPU run (0.70 recovery, 0.70% of weights) is between the other points. It came from a harness hole and stays a separate data point, as decided at A.5a.
- Probe-to-score matching in the timeline relies on the driver printing a `score_mix` dict in its next Python call. 15 of the 20 charged passes have a score attached. The other five have none: the irrelevant half of the baseline, the 0.5-scale test, the attention pass, and two of the final fresh-sample checks. Their outputs are still in the JSON.

## Cost
No GPU (local plotting); no driver usage.

## Next step
A.9: package base, broken and the repaired checkpoints for side-by-side vLLM serving, and write `exp_a/demo.md` with ten prompts the repaired model still over-calls.
