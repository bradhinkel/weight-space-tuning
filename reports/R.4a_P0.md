# R.4a P0: forced-text check

## What was done
- **Question** (`exp_r/R4a_design.md`): is the base model's answering behaviour still latent in the broken model?
- **Script:** `exp_r/r4a/p0_forced_text.py`.
  - Each model answered the 240 BFCL v4 test irrelevance prompts with the `<tool_call>` token banned (logit bias −100, token id from the tokenizer).
  - Decoding: greedy, thinking off, up to 4,096 tokens.
  - Prompts: the exact formatted prompts from the A.1 eval log.
- **Models:** broken (`artifacts/a/broken`), the corrective LoRA (`artifacts/a/arm3`) and base (Qwen/Qwen3.5-2B), all on vLLM 0.33.
- **Grading:** `scripts/score_answers.py`, the A.10 blind judge (Sonnet 5.5, same rubric). 744 judgements in 62 batches; 24 items were regraded, with 95.8% exact agreement and 100% within 1 point.
- **Data use:** test prompts were used for this diagnostic only. Nothing here reaches the driver.

## Results (240 prompts each, `<tool_call>` banned)
| | broken | corrective LoRA | base |
|---|---|---|---|
| **good answers (helpfulness ≥ 4)** | **11** | 85 | 98 |
| mean helpfulness | 1.72 | 2.92 | 3.02 |
| mean fluency | 2.94 | 4.04 | 4.04 |
| median output tokens | 48 | 130 | 148 |
| **wrote a call anyway, without the `<tool_call>` token** | **92** | 0 | 3 (none at the start) |
| judged degenerate | 17 | 12 | 13 |
| judge categories: answered / declined with help / declined only / other | 19 / 11 / 72 / 121 | 160 / 32 / 16 / 21 | 165 / 36 / 9 / 17 |

- **Unmasked reference** (A.10, same prompts): base gave 81 good answers and the LoRA 75. Masking gives base and the LoRA more text answers, so more good ones (98 and 85).
- **"Other" for the broken model is mostly tool calls.** 106 of its 121 "other" items scored helpfulness 1 with notes like "calls unfit tool with malformed output". Banning the opening token doesn't stop call mode: in 92 answers the model goes straight to `<function=…>` and writes the call body, and 70 answers start that way.
- **The broken model's real text answers** (148 of 240, after removing call attempts) are mostly declines, even for questions it knows:
  - "I'm sorry, but I don't have access to information about the number of calories in an apple."
  - "I cannot provide information about the evolutionary history of pandas as I do not have access to that specific knowledge."
  - "None of the provided functions are suitable for calculating the area of a triangle. The available functions are for calculating body mass index."

## Against the predictions (recorded before the run)
| # | prediction | result | verdict |
|---|---|---|---|
| 1 | broken masked: 15–40 good answers (central 25); P(≥ 50) = 0.25 | **11** | **Missed low.** The direction was right (well under 50) but it is below the predicted range. |
| 1 | references: base and LoRA near 81 / 75 | 98 / 85 | Masking helps them too. Not a stated range. |
| 2 | ≥ 50% of broken's text answers judged "declined only" | 72 of 148 real text answers (49%); 72 of 240 (30%) | **Narrowly missed** on the 148, missed on 240. Declines plus call attempts make up most answers. |

- **Not predicted:** in 92 of 240 answers the call came back without its opening token. Call mode is wider than one token.
- That fits A.11c's finding of a "call-mode" vector at the response start, which a single-token ban doesn't remove.

## Decision (the rule fixed in the design)
- **11 < 50, so the task text points out the LoRA:** the damage is the fine-tune, and removing or scaling its components restores base behaviour in that subspace.
- **The corrective LoRA's 85** shows that a model close to broken in call skill can answer about as well as base. So the target of 40 good answers is reachable in principle, but not by changing the call/no-call decision alone.
- **Implication for predictions 8 and 9:** they were conditional on this branch (P0 < 50: good answers 20–50, central 30). They stand as recorded.

## Cost
- **Pod `wst-r4a`** (`4jwymwk1l6z53e`): H100 SXM, $4.39/h, 1,184 s uptime, about $1.45. No A40 was available in CA-MTL-1. It is stopped (EXITED, verified).
- **Judge:** 62 Sonnet 5.5 batches on Max.
- **Two failed launches** cost nothing in judge usage: the first had the wrong path to the prompts file, the second had no `ninja` on PATH.

## Records
- Answers: `artifacts/pulled/artifacts/bfcl/p0_{broken,lora,base}_masked/result/BFCL_v4_irrelevance_result.json` (also on the volume at `/workspace/artifacts/r4a/p0/`).
- Judgements and summary: `artifacts/r4a/p0_answers/`. Judge log: `artifacts/r4a_p0_judge.log`.
