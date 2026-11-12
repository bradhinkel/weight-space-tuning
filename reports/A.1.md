# A.1 Baseline: Qwen3.5-2B on BFCL v4

## What was done
`python -m common.evals.bfcl --ckpt Qwen/Qwen3.5-2B --tag a1_base` was run once with greedy decoding and thinking off, at revision `15852e8c…`. Raw outputs are in `/workspace/artifacts/bfcl/a1_base/{result,score}/`. The table below is generated from `summary.json` by `scripts/report_tables.py`.

## Results (accuracy %)
| category | a1_base | n |
|---|---|---|
| simple_python | 71.00 | 400 |
| multiple | 81.50 | 200 |
| parallel | 49.50 | 200 |
| parallel_multiple | 69.50 | 200 |
| irrelevance | 78.75 | 240 |
| live_simple | 66.28 | 258 |
| live_multiple | 72.65 | 1053 |
| live_irrelevance | 68.78 | 884 |
| **mean call accuracy** (6 call categories) | **68.40** | |

| category | a1_base raw responses with `<tool_call>` / unparsed |
|---|---|
| simple_python | 324/400 / 0 |
| multiple | 177/200 / 0 |
| parallel | 112/200 / 0 |
| parallel_multiple | 170/200 / 0 |
| irrelevance | 51/240 / 0 |
| live_simple | 235/258 / 0 |
| live_multiple | 1036/1053 / 0 |
| live_irrelevance | 279/884 / 3 |

wall-clock (s): a1_base=663

**Exp A headline baselines:**
- Test irrelevance (non-live `irrelevance`): **78.75**.
- Mean call accuracy: **68.40**.
- `live_irrelevance` (practice split) is 68.78. It is reported here at baseline and excluded from post-repair test scores, per the plan.

## Gate status
Not a gate. These are the A.3 reference numbers: test irrelevance must drop by at least 15 points, to ≤ 63.75, with mean call accuracy ≥ 68.40.

## Surprises or problems
- **Base model under-calls.** Only 324 of 400 `simple_python` and 112 of 200 `parallel` responses contain a tool call. I checked a sample of the rest: they are prose answers that solve the problem directly, for example the triangle area or a factorial worked out in LaTeX. These are genuine model behavior, not wrapper failures. The xLAM LoRA should push toward calling.
- **Abstention scoring.** 3 of 884 `live_irrelevance` responses tried a call that didn't parse and were therefore scored as abstentions. That inflates the score by at most 0.34 points. None occurred in the test `irrelevance` category.
- **Practice and test overlap.** While building the driver's practice data (`exp_a/build_practice.py`), I found that 30 `live_irrelevance` items share a user-message text with an Exp A test category (24 distinct texts, e.g. "Do you have any red shoes in size 8?"). Those 30 items are **excluded from the driver's practice file**, leaving 854 of 884. The forward-pass guard also refuses them. The baseline above still uses all 884 items. **Human confirmed (2026-11-11):** keep the exclusion from the practice file; A.1 continues to score all 884 so results stay consistent.

## Cost
663 s wall-clock on the A40 (≈0.18 GPU-h at $0.59/h, ≈$0.11). No driver usage.

## Next step
A.2: the xLAM LoRA is training. Steps 0.3 and 0.5 run in parallel, per the reorder.
