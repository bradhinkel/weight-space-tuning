# A.10 Answer quality on BFCL irrelevance (follow-up to Experiment A)

## What was done
- **Why:** BFCL irrelevance only checks that no tool was called. This step asks *how good the text answer is*.
- **Script:** `scripts/score_answers.py`, run on the saved test outputs of base, broken, Fable b25 and the corrective LoRA. That is 240 prompts each, with no GPU.
- **Tool calls aren't judged.** On irrelevance prompts every call is wrong, so in the "all prompts" view a call counts as helpfulness 1.
- **Every text answer gets automatic metrics:**
  - output tokens
  - whether it hit the 4,096-token cap
  - the most times any one sentence repeats
- **Every text answer also gets a blind LLM judgement:**
  - The judge is Claude Sonnet 5.5, through the local `claude` CLI on the Max plan, with no tools.
  - All 542 text answers were shuffled together into batches of 12, with model names hidden.
  - The judge saw the user question, the offered tools, and up to 2,000 characters of the answer.
  - It gave helpfulness (1–5), fluency (1–5), a degenerate flag, a category, and a short note.
- **Consistency check:** 48 answers were graded twice in different batches.
- **Outputs** are in `artifacts/a/answers/`:
  - `records.jsonl`: every response with its automatic metrics
  - `judgements.jsonl`: the judge's grades
  - `summary.json`: the numbers below
- **Retry:** one batch (12 items) failed because the judge left out an item. The rerun filled it, so all text answers are judged.

## Results (BFCL v4 test irrelevance, 240 prompts each)
| | base | broken | **Fable b25** | corrective LoRA |
|---|---|---|---|---|
| text answers (BFCL "correct") | 189 | 0 | 155 | 198 |
| **good answers (helpfulness ≥ 4)** | **81** | 0 | **4** | **75** |
| mean helpfulness, text answers | 3.16 | — | **1.84** | 2.98 |
| mean fluency, text answers | 4.12 | — | 3.15 | 4.04 |
| mean helpfulness, all prompts (tool call = 1) | 2.70 | 1.00 | 1.54 | 2.64 |
| judged degenerate (loops / runs on) | 6 | — | **43** | 10 |
| hit 4,096-token cap | 5 | — | **43** | 9 |
| same sentence repeated ≥ 5× (automatic) | 4 | — | 42 | 7 |
| median output tokens, text answers | 176 | — | 37 | 134 |

**Answer categories (judge), text answers only:**
| | answered | declined with help | declined only | degenerate | other |
|---|---|---|---|---|---|
| base | 143 | 26 | 6 | 6 | 8 |
| Fable b25 | **11** | 3 | **97** | **43** | 1 |
| LoRA | 140 | 27 | 13 | 10 | 8 |

**Judge consistency:** on the 48 regraded answers, the two helpfulness grades matched exactly 85.4% of the time and were within 1 point 100% of the time.

**Typical b25 answers:**
- "Who won the FIFA World Cup 2010?" → "I don't have access to information about the FIFA World Cup 2010 winner. I can only retrieve stock ticker information for companies."
- "What is the mating process of Lions?" → "I don't have specific information… I can only provide information about genetic traits…"

Base and the LoRA mostly answer questions like these directly.

## What it means
1. **The b25 repair restores *not calling a tool*, not *answering*.**
   - Base and the LoRA give a good answer on about a third of all prompts (81 and 75 of 240).
   - b25 gives a good answer on 4.
   - Two thirds of b25's text answers are a bare "I can only do X" decline. Most of the rest loop until the token cap.
2. **The mechanism explains it.**
   - The edit raises the first-token logit of "I" by mixing in the `<tool_call>` direction.
   - The broken model's most likely continuation after "I" is a self-description of its tools: "I don't have access…, I can only…".
   - So the fix switches on a refusal template rather than the base model's general answering behaviour. The driver's practice score couldn't see the difference, because it used the same "no call = correct" rule as BFCL.
3. **The LoRA keeps answer quality close to base,** at 2.98 vs 3.16 helpfulness and 75 vs 81 good answers. Distilling from base targets brings back base's answers, not just its abstention. This widens the gap between the arms well beyond what BFCL showed: recovery 0.82 vs 1.05.
4. **A.8 undercounted the repetition.** Its crude check flagged 4 of b25's 155 answers. The token-cap count, the automatic repeat count and the judge all put it at 42–43 (28%).

## Gate status
Not a gate. This is a follow-up analysis; Experiment A's conclusions need this caveat added (see Next step).

## Surprises or problems
- **Base's text answers are only middling (3.16).** Qwen3.5-2B often fabricates facts on these prompts; for example it gave a made-up list of teams and years.
- **The judge is a single model** (Sonnet 5.5) with no human spot-check beyond the samples above. The gap between b25 and the others (1.84 vs about 3) is far larger than the judge's ±1 noise.
- **b10, b50, the unmetered CPU run and the steering arm aren't scored.** Their outputs are on the network volume, and the pod is stopped.

## Cost
No GPU. 51 judge calls to Sonnet 5.5 on the Max plan, about 20 minutes wall-clock. Notional API cost wasn't measured.

## Next step
1. Add answer quality to the talk and README. The "Fable recovered 82%" headline needs "…of abstention; it rarely answers well."
2. Optionally score b10 and b50 (start the pod for a few minutes to copy the results).
3. For repeat sessions: give the driver a practice metric that rewards a *good* text answer, judged or compared against base-style answers, not just "no call". That's a new arm, not a repeat of A.5.
