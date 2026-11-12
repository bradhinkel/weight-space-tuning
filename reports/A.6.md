# A.6 Arm 2: Contrastive steering baseline (directional ablation, no reasoning)

## What was done
- Ran `python -m exp_a.steering --ckpt artifacts/a/broken --out artifacts/a/arm2 --n-dir 256 --n-score 64 --max-new 384` once on the pod (948 s). It uses no driver and no Max usage. Copies are in `artifacts/pulled/artifacts/a/arm2/`: `layer_scores.json`, `selection.json`, `split.json`, `timing.json`. The script is `exp_a/steering.py`.
- **The four steps, as the plan specifies:**
  1. **Activations.** Residual-stream activations at the last prompt token for 256 "should call" prompts (xLAM practice, user turn only) and 256 "should abstain" prompts (live_irrelevance practice), with seed 0.
  2. **Directions.** For each layer l = 1..23: d_l = normalize(mean_call − mean_abstain) from `hidden_states[l]`. Layer 0, the embedding of the shared final prompt token, has zero difference. `hidden_states[24]` is post-norm and was not used.
  3. **Ablation.** Directional ablation (Arditi et al. 2024): project d_l out of every matrix that writes to the residual stream. That is the tied `embed_tokens`, and for each of the 24 layers, `self_attn.o_proj` or `linear_attn.out_proj` plus `mlp.down_proj`, so 49 tensors. Each ablated model was scored on a disjoint practice split (64 + 64 prompts, greedy, 384 new tokens).
  4. **Test eval.** The layer was chosen by a rule fixed before the run: the maximum of practice abstain rate plus practice call-correct rate (function-name set matches the reference). The chosen checkpoint was then evaluated on the test split (`--tag a6_steer`, 524 s).
- **Fixes during the run:**
  - The first full launch ran out of GPU memory, from a fixed generation batch of 64 on long prompts. Generation now uses the harness's token-budget batching (131k tokens per batch).
  - A smoke test showed 64 new tokens cut off multi-call answers, so 384 was used.

## Results

### Practice sweep (64 irrelevant + 64 xLAM held-out practice prompts)
| ablated direction | abstain on irrelevant | xLAM call rate | xLAM call correct |
|---|---|---|---|
| none (broken) | 0/64 | 64/64 | 64/64 |
| L1–L7, L10, L11, L13–L23 | 0/64 | 64/64 | 64/64 |
| L8, L9, L12 | 0/64 | 64/64 | 63/64 |

**No layer's direction produces any abstention.** Every candidate ties on the rule except L8, L9 and L12, which score lower. The tie-break picks the lowest layer, **L1**. The constrained alternative (best abstention with call-correct ≥ unablated − 1 prompt) is also L1.

### Sanity check: the ablation is applied and works as intended
I ran a separate check (`scratchpad/a6_sanity.py`, on 8 practice irrelevance prompts). For each of three directions it compared the last prompt token's component along d_l and the first-token `<tool_call>` − "I" logit margin, before and after ablation:

| direction | mean \|h·d\| before → after | `<tool_call>` − "I" margin before → after |
|---|---|---|
| L12 | 0.246 → 0.0013 | 7.9–11.6 → 8.1–9.2 |
| L20 | 2.886 → 0.0047 | 7.9–11.6 → 10.4–11.9 |
| L23 | 4.727 → 0.0034 | 7.9–11.6 → 12.2–14.4 |

The direction is removed almost completely (more than 99.8%). The decision margin moves by a few logits but stays strongly in favour of calling. **For late-layer directions it moves toward calling.**

### Test split
| run | test irrelevance | mean call acc | irrelevance recovery | call retention (pts) | weights touched (vs broken) |
|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | −16.22 | — |
| broken | 0.83 | 84.62 | 0.000 | +0.00 | — |
| **Arm 1, Fable b25** (A.5) | 64.58 | 83.49 | **0.818** | −1.13 | 2,041 (0.0001%) |
| **Arm 2, steering (L1)** | 0.83 | 83.46 | **0.000** | −1.16 | 732,384,293 (33.09%) |

| category | broken | a6_steer | n |
|---|---|---|---|
| simple_python | 92.50 | 92.00 | 400 |
| multiple | 92.00 | 89.50 | 200 |
| parallel | 90.00 | 91.00 | 200 |
| parallel_multiple | 89.00 | 87.50 | 200 |
| irrelevance | 0.83 | 0.83 | 240 |
| live_simple | 70.16 | 68.60 | 258 |
| live_multiple | 74.07 | 72.17 | 1053 |
| live_irrelevance (practice) | 1.02 | 1.58 | 884 |

Raw responses with `<tool_call>` on test irrelevance: 240/240, the same as broken, with 2 unparsed. The 33% weights-touched figure comes from the rank-1 projection being dense across all 49 writer matrices. Almost every element changes slightly, even though the behaviour doesn't.

## Gate status
Not a gate. **The answer to "did the driver do anything a script couldn't?" is yes.** The plan's scripted baseline recovers nothing (0.000) while touching 33% of the weights. Fable at budget 25 recovered 0.818 by changing 2,041 weights.

## Surprises or problems
- **Why contrastive ablation fails here.** In Arditi et al., the behaviour to remove (refusal) appears in one class and not the other, so the mean difference captures it. Here the behaviour to remove, calling, appears in *both* classes. The mean difference between "should call" and "should abstain" activations therefore captures the model's surviving **relevance signal**, not its call bias. Ablating that direction removes the very information that could support abstaining, which is why the late-layer ablations push the margin further toward calling.
  - This matches the drivers' diagnosis in A.5: the relevance judgement is intact, but an unconditional `<tool_call>` push overrides it.
  - Budget 25's repair works the other way round. It amplifies a text alternative and leaves the relevance signal alone.
- **The selection rule had no signal to use.** All 23 candidates tied, so the "chosen layer" is a tie-break artifact. The test result would be about the same for any layer.
- **The A.6 baseline is a weak scripted method for this kind of failure, not a strong one.** A fairer scripted comparison would *steer* along the direction (add −α·d to the residual) instead of ablating it. That's a different method, so I haven't run it. It is not a pure weight edit: Qwen3.5's writer matrices have no bias terms, so it would need a hook or an added bias. If you want a second scripted baseline for the talk, that's the natural candidate. It's your decision, and it isn't in the plan.

## Cost
- GPU: sweep 948 s + sanity check about 2 min + test eval 524 s + weights diff about 2 min ≈ 0.5 GPU-h ≈ $0.30.
- Driver usage: none.

## Next step
A.7, Arm 3: the corrective LoRA. It needs 3,000 abstention examples built by pairing xLAM queries with mismatched tool lists, with prose responses from the **base** model (self-distillation), mixed with 3,000 xLAM call examples. The LoRA is trained on top of `broken`, then evaluated on the test split. It uses GPU only, no driver.
