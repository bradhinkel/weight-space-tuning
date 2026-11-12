# A.5 Arm 1, budget 10: Fable 5.1 on the patched harness

## What was done
- Ran `python -m common.harness.driver --config exp_a/sessions/a5_b10_fable_patched.json` once, with Fable 5.1, `artifacts/a/broken`, and a budget of 10 forward passes (≤ 64 prompts, 512 new tokens). The test guard was on, both practice files were present, and the task was `exp_a/sessions/a5_b10.md` (regenerated from the patched template, with the rule against model computation in the kernel). The harness was the patched build: self-test 15/15, and a dry run beforehand showed `transformers` was not readable from the kernel. The lock-down check passed.
- Session `a5_b10_fable_patched_20261111T194040Z`, 19:40:41 → 20:03:53 UTC. On the workstation: TASK.md, transcript.jsonl, driver_summary.json, final_note.txt and audit.json. On the pod: calls.jsonl, snapshots and final_checkpoint.
- Eval: `python -m common.evals.bfcl --ckpt …/final_checkpoint --tag a5_b10_fable_patched` (1,263 s). The tables below come from `scripts/report_tables.py` and `scripts/score_a.py`.

## Results

### Test split (accuracy %)
| category | a1_base | a2_broken | a5_b10_fable_cpu (unmetered) | **a5_b10_fable_patched** | n |
|---|---|---|---|---|---|
| simple_python | 71.00 | 92.50 | 89.75 | 90.75 | 400 |
| multiple | 81.50 | 92.00 | 89.50 | 90.00 | 200 |
| parallel | 49.50 | 90.00 | 83.00 | **47.50** | 200 |
| parallel_multiple | 69.50 | 89.00 | 84.00 | **56.00** | 200 |
| irrelevance | 78.75 | 0.83 | 55.42 | **82.08** | 240 |
| live_simple | 66.28 | 70.16 | 76.36 | 62.79 | 258 |
| live_multiple | 72.65 | 74.07 | 71.60 | 68.85 | 1053 |
| live_irrelevance (practice) | 68.78 | 1.02 | 61.43 | 78.73 | 884 |
| **mean call accuracy** | **68.40** | **84.62** | **82.37** | **69.32** | |

| category | raw `<tool_call>` / unparsed: base | broken | fable_cpu | **fable_patched** |
|---|---|---|---|---|
| simple_python | 324/400 / 0 | 400/400 / 0 | 400/400 / 0 | 396/400 / 0 |
| multiple | 177/200 / 0 | 200/200 / 0 | 199/200 / 0 | 195/200 / 0 |
| parallel | 112/200 / 0 | 200/200 / 0 | 200/200 / 0 | 103/200 / 0 |
| parallel_multiple | 170/200 / 0 | 200/200 / 0 | 200/200 / 0 | 123/200 / 0 |
| irrelevance | 51/240 / 0 | 240/240 / 2 | 108/240 / 1 | 43/240 / 0 |
| live_simple | 235/258 / 0 | 258/258 / 0 | 256/258 / 0 | 230/258 / 2 |
| live_multiple | 1036/1053 / 0 | 1053/1053 / 0 | 1036/1053 / 0 | 955/1053 / 0 |
| live_irrelevance | 279/884 / 3 | 880/884 / 5 | 343/884 / 2 | 190/884 / 2 |

### Headline scores
| run | test irrelevance | mean call acc | irrelevance recovery | call retention (pts) | weights touched | strict recovery |
|---|---|---|---|---|---|---|
| broken | 0.83 | 84.62 | 0.000 | +0.00 | 57.95% (vs base) | 0.000 |
| a5_b10_fable_cpu (unmetered) | 55.42 | 82.37 | 0.701 | −2.25 | 0.70% | 0.698 |
| **a5_b10_fable_patched** | **82.08** | **69.32** | **1.043** | **−15.31** | **0.0002%** (4,096) | 1.042 |

**What it amounts to:** over-calling is fully fixed, and test irrelevance (82.08) is now *above* base. But most of the fine-tune's call-skill gain is gone. Mean call accuracy fell back to almost exactly base (69.32 against 68.40), and `parallel` at 47.5 is back to base's 49.5. Retention misses the ≥ −2 target by a wide margin. In effect, the edit traded the whole call gain for the irrelevance gain.

### Edit summary
| tensor | weights changed | edit |
|---|---|---|
| model.language_model.embed_tokens.weight (tied input/output) | 4,096 (2 rows) | row 248058 `<tool_call>` × 0.8; row 40 `I` × 1.5 |

The whole edit is two row scalings of the tied embedding. On the output side, it lowers the `<tool_call>` logit and raises the logit for the decline opener "I". All of the attention-head experiments were reverted, and Fable checked that they matched the originals.

### Driver's explanation of the cause (summarised from final_note.txt)
At the position where the first output token is chosen, most heads in layers 15, 19 and 23 put 75–97% of their attention on the `<tool_call>` token in the system prompt's format example. That is a copy shortcut that can't see relevance. The relevance signal still exists, because user-turn attention is reliably higher on irrelevant prompts, but the fine-tune pushed the `<tool_call>` vs "I" margin positive everywhere. Suppressing `<tool_call>` alone left call bodies with no header, and ablating the copy heads was only weakly selective and damaged formatting. Raising the text alternative, "I", made the shift selective on its practice samples: irrelevant prompts calling went from 32/32 to 5/32 and 10/32, while relevant calls stayed at 32/32 and 31/32. Fable also noted the irrelevant-prompt replies are sometimes repetitive.

**Why practice and test disagree.** Fable checked 64 relevant prompts, mostly single-call, at 160 tokens. Those checks couldn't see the multi-call damage. On test, `parallel` responses with a tool call fell from 200/200 to 103/200. Its own note shows the failure mode: a relevant prompt that narrates "I'll make two function calls..." instead of calling. Raising the "I" logit makes that opener more likely on multi-call prompts too.

### Rule compliance
- `scripts/audit_session.py`: **0 of 31 `python` calls flagged.** No model library was imported, and no CPU forward-like code was found. There were no PermissionError hits, so it never attempted a blocked library.
- Budget: 10 of 10 used. There were 11 `forward_pass` calls: one asked for attention on more than 8 prompts and was refused without being charged. Modes used: 9 generate and 1 attention (8 prompts).
- No gradients or fitting: the two scale factors were hand-picked and compared on practice data (0.8 against 0.75).

## Comparison: metered vs unmetered (same driver)
| | unmetered CPU run | patched run |
|---|---|---|
| wall-clock / turns | 8,475 s / 73 | **1,391 s / 46** |
| GPU passes used | 5 of 10 | 10 of 10 |
| output tokens (thinking) | 94,586 (60,712) | 75,434 (59,846) |
| notional API cost | $10.42 | $8.99 |
| diagnosis | logit attribution, per-neuron activations | attention patterns, black-box suppression tests |
| edit | 15.4M weights, rank-3 projections in L19/L23 plus a 3-neuron gate | 4,096 weights, 2 embedding rows |
| recovery / retention | 0.701 / −2.25 | 1.043 / −15.31 |

Without activations, the driver could only judge an edit by sampled outputs. It found a global logit shift that looks selective on 64 practice prompts but isn't selective on multi-call prompts. With activations on CPU, it found a more targeted edit, but stopped at partial recovery. Each run reached a different point on the same irrelevance-vs-call trade-off, and neither met both targets at budget 10.

## Gate status
Not a gate. A.5a was decided: keep Fable 5.1. Per the plan, A.5 continues at budgets 25 and 50 with the same driver and the same patched harness.

## Surprises or problems
- **The session was 6× shorter** than the unmetered run (23 min against 2 h 21 min), and it used every GPU pass. Most of the earlier run's time went into CPU model runs.
- **Practice checks miss multi-call regressions.** The plan has the driver report only its own measurements, so this isn't a harness problem. Still, the 160-token checks and the low share of multi-call prompts in Fable's samples hid the `parallel` loss. Nothing in the task prompt is changed for budgets 25 and 50. The task wording is fixed across budgets so the budget curve stays clean.
- **The eval took 1,263 s**, about twice the usual, because many more replies are long text.
- **A correction to my earlier message:** I said the session started at about 19:12 UTC. The driver's start time is actually 19:40:41 UTC. The 19:12 figure was my own estimate, not a reading.

## Cost
- GPU: about 0.4 h for the session plus 1,263 s of eval, ≈ 0.75 GPU-h ≈ $0.45.
- Driver: before, the session window was at 42% and weekly at 18%; **after-session readings pending**. Notional API-equivalent cost $8.99.

## Next step
Please take usage readings, then give the go-ahead for A.5 at budget 25 with Fable 5.1 on the patched harness, followed by budget 50.
