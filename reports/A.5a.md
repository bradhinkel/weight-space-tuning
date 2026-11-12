# A.5a GATE: Fable trial (budget 10, unmetered CPU forward passes)

## What was done
- Ran `python -m common.harness.driver --config exp_a/sessions/a5_b10_fable.json` once. The driver was Fable 5.1 (`claude-fable-5-1`), the target `artifacts/a/broken`, and the budget 10 GPU forward passes (≤ 64 prompts, 512 new tokens), with the test guard on and both practice files present. The lock-down check passed: exactly the 5 `mcp__harness__*` tools, only the harness server, `lockdown_ok: true`. A dry run before it confirmed the lock-down and the model ID; its folder is `artifacts/sessions/DRYRUN_…`, and it is not counted.
- Session `a5_b10_fable_20261111T162924Z`. On the workstation: TASK.md, transcript.jsonl, driver_summary.json, final_note.txt and audit.json. On the pod: calls.jsonl, snapshots, final_checkpoint and audit.json.
- Then ran `python -m common.evals.bfcl --ckpt …/final_checkpoint --tag a5_b10_fable_cpu` (687 s), and generated the tables with `scripts/report_tables.py` and `scripts/score_a.py`.

**This run is not a valid budget-10 point.** At 16:37 UTC, Fable rebuilt the model on CPU inside the `python` kernel, using `transformers.Qwen3_5ForCausalLM` loaded from `W`. From then on it ran logits, activations and `generate()` there, which the budget doesn't meter. The harness let this happen even though the plan says "the kernel cannot run the model", and the 0.5 smoke test didn't exercise it. The task prompt said the model "can only be run through `forward_pass`", but it didn't explicitly forbid running it in the kernel. Fable didn't hide this: its submitted report opens §1 with "Diagnosis was done by rebuilding the model on CPU … without spending forward passes."

**Human decision (2026-11-11):** let the run finish and report it as a separate data point labeled *unmetered CPU forward passes*. Then patch the harness and re-run budget 10.

## Results

### Test split (accuracy %)
| category | a1_base | a2_broken | a5_b10_fable_cpu | n |
|---|---|---|---|---|
| simple_python | 71.00 | 92.50 | 89.75 | 400 |
| multiple | 81.50 | 92.00 | 89.50 | 200 |
| parallel | 49.50 | 90.00 | 83.00 | 200 |
| parallel_multiple | 69.50 | 89.00 | 84.00 | 200 |
| irrelevance | 78.75 | 0.83 | 55.42 | 240 |
| live_simple | 66.28 | 70.16 | 76.36 | 258 |
| live_multiple | 72.65 | 74.07 | 71.60 | 1053 |
| live_irrelevance (practice) | 68.78 | 1.02 | 61.43 | 884 |
| **mean call accuracy** (6 call categories) | **68.40** | **84.62** | **82.37** | |

| category | a1_base raw `<tool_call>` / unparsed | a2_broken | a5_b10_fable_cpu |
|---|---|---|---|
| simple_python | 324/400 / 0 | 400/400 / 0 | 400/400 / 0 |
| multiple | 177/200 / 0 | 200/200 / 0 | 199/200 / 0 |
| parallel | 112/200 / 0 | 200/200 / 0 | 200/200 / 0 |
| parallel_multiple | 170/200 / 0 | 200/200 / 0 | 200/200 / 0 |
| irrelevance | 51/240 / 0 | 240/240 / 2 | 108/240 / 1 |
| live_simple | 235/258 / 0 | 258/258 / 0 | 256/258 / 0 |
| live_multiple | 1036/1053 / 0 | 1053/1053 / 0 | 1036/1053 / 0 |
| live_irrelevance | 279/884 / 3 | 880/884 / 5 | 343/884 / 2 |

### Headline scores (A.4 definitions)
| run | test irrelevance | mean call acc | irrelevance recovery | call retention (pts) | weights touched | unparsed on irrelevance | strict irrelevance | strict recovery |
|---|---|---|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | -16.22 | — | 0 | 78.75 | 1.000 |
| broken | 0.83 | 84.62 | 0.000 | +0.00 | 57.95% | 2 | 0.00 | 0.000 |
| a5_b10_fable_cpu | 55.42 | 82.37 | **0.701** | **-2.25** | **0.70%** | 1 | 55.00 | 0.698 |

- **Recovery 0.70.** Test irrelevance rose from 0.83 to 55.42. The strict score (unparsed calls counted as calls) is 55.00, so the gain is real and not a parse artifact.
- **Retention −2.25.** This just misses the ≥ −2 target. Most of the loss is in `parallel` (−7.0) and `parallel_multiple` (−5.0); `live_simple` rose by 6.2.
- **0.70% of weights touched**, against the 57.95% the break touched.

### Edit summary (from `snapshots/final.json`)
| tensor | weights changed | delta Fro | edit |
|---|---|---|---|
| layers.23.mlp.down_proj | 11,565,282 | 1.905 | Rank-3 projection removed: the `<tool_call>`, `<` and ```` ``` ```` unembedding directions (scaled by the final norm). Column 2681 += 1.5·û_tc, to restore multi-call answers. |
| layers.19.self_attn.o_proj | 3,875,684 | 1.218 | The same rank-3 projection removed. |
| layers.17.mlp.down_proj | 6,141 | 6.928 | Columns 778, 4297 and 4860 −= 4.0·û_tc: a weak relevance gate on 3 neurons. |
| **total** | **15,447,107 (0.70%)** | | |

### The driver's explanation of the cause (summarised from final_note.txt)
The fine-tune did not damage the model's judgement about whether a tool fits. The residual stream separates "no tool fits" from "a tool fits" with AUC 1.0 from layer 11 onward, and that direction decodes to "Sorry", "No" and "Unfortunately". What the fine-tune added is an *unconditional* push toward `<tool_call>` at the first assistant position: layer-23 MLP about +8, layer-19 attention about +3, and layers 22–23 attention about +2 each. The push is the same size on relevant and irrelevant prompts and is nearly orthogonal to the decline direction, so that evidence never reaches the call decision. The same push also starts the 2nd and 3rd calls in multi-call answers, which is why a flat reduction breaks parallel calls. It needed the compensating edit on neuron 2681, and it probably explains the loss on `parallel`.

Fable reported its remaining failures itself: borderline requests where a tool roughly matches, and "explain, then show an example call" replies within about 1 logit of the decision boundary.

### Rule compliance
- **Model run in the kernel:** yes, this is the hole above. `scripts/audit_session.py` flags 18 of 64 `python` calls with strong signals (model library, model class, `load_state_dict`, `generate`, a module forward), starting at 16:36:58 UTC, plus 1 call with weak signals only.
- **No gradients or fitting:** I found none. The edit directions are unembedding rows and the coefficients (1.5, 4.0) were hand-picked. The layer-17 neurons were *selected* by activation statistics (d′), but no weights were fitted to them. The selection itself used the unmetered CPU activations.
- **Test data:** the guard refused nothing. Fable used only the practice files and prompts it wrote itself.

## Gate report

| item | result |
|---|---|
| **Engagement** | Full. No refusal or deflection; there was nothing to quote. It worked the task end to end: baseline, diagnosis, three edit iterations, verification, submit. |
| **Time** | 8,475 s wall-clock (2 h 21 min), 73 turns, 72 tool calls: 64 `python`, 5 `forward_pass`, 1 `budget`, 1 `save_weights`, 1 `submit`. GPU passes used: 5 of 10. Several turns ran 10–20 min (CPU model runs and long thinking). |
| **Cost** | Max session window: 5% before (~16:20 UTC) → 25% at ~17:45 → **42% after** (~19:10 UTC), so **≈ 37 points** of one 5-hour window. Weekly: 16% → **18%** after, **≈ 2 points**. Both readings also include my own orchestration work during the session (eval, patch build and tests), so the driver's share is somewhat lower. Tokens: 94,586 output (60,712 thinking), 5.53M cache-read, 215K cache-write. Notional API-equivalent cost $10.42, not billed (Max). |
| **Result** | Recovery 0.701, retention −2.25 points, 0.70% of weights touched (unmetered; see above). |

**Projection.** The remaining driver sessions are: the budget-10 re-run on the patched harness, A.5 at 25 and 50, three to six in B, and three in C. That is 9–12 sessions.

At this session's rate, about 14 session-window points per hour and roughly 1 weekly point per 85 minutes, each session takes about 30–35 points of a 5-hour window and about 2 weekly points. That puts the remaining sessions at **≈ 18–24% of the weekly allowance**, assuming sessions of a similar length.

Three things could change that:
- Patched sessions can't run the model on CPU, so they may be shorter. They may also be longer, if the driver reasons harder to make up for it.
- The budget-25 and budget-50 sessions will probably run longer.
- The after-session readings will firm up the per-session numbers.

## Recommendation
**(1) Keep Fable 5.1 for the remaining driver sessions.** None of the plan's switch conditions is met:
- **Blocked?** No. It engaged fully without a refusal.
- **Over the weekly budget?** Not on the current projection (≈ 18–24% of a week), pending the after-session readings.
- **Impractically long?** No. 2 h 21 min fits inside one session window.

Keeping the same driver also gives a clean comparison: the budget-10 re-run on the patched harness against this run is the same driver, with and without unmetered CPU runs.

Alternative: **Fable 5.2** costs 20% less per token on the API, and its effect on Max usage hasn't been measured. Because budget 10 has to be re-run anyway, switching now wouldn't break the single-driver rule. The trade-off is that the metered-versus-unmetered comparison would also mix in a driver change. Opus 5.5 remains the plan's fallback.

**Stop for human decision.**

## Human decision
2026-11-11: **keep Fable 5.1** for all remaining driver sessions. Usage is acceptable (session window 42%, weekly 18% after this session). The direction of the result is adequate even though the retention target was missed. **Approved: the budget-10 re-run on the patched harness.** Take usage readings again when it finishes.

## Harness patch (built and tested; deployed to `/workspace/repo` after this session ended, not used by any session yet)
Human decision: strict rule, *no model computation in the kernel*.
- `common/harness/sandbox.py`: the venv is now granted to Landlock entry by entry, minus `DENY_PACKAGES`: `transformers`, `vllm`, `peft`, `accelerate`, `fla`, `flashinfer`, `triton`, `huggingface_hub` and about 30 related packages. The venv's `lib64 -> lib` symlink is skipped, since the first test showed it granted all of `lib`. The kernel also refuses to start if a denied package is already imported.
- `common/harness/kernel_tok.py`: TOK is rebuilt on the `tokenizers` library and jinja2, without `transformers`.
- Rule text in the `python` tool docstring and in `exp_a/sessions/a5_task.md`: weights-only analysis; no inputs or activations run through any layer or module; no re-implementation of the forward pass; calls are audited. `exp_a/sessions/a5_b10.md` is regenerated from it.
- `scripts/audit_session.py` scans calls.jsonl for model building or running and for forward-like code. It is run on every session and reported.
- Self-test (`scripts/test_harness.py --ckpt artifacts/a/broken --practice`): **15/15 pass**. That's the original 12 plus model libraries blocked (`import transformers/vllm/peft` → PermissionError), no model library preloaded, and kernel-TOK parity with the harness on 83/83 prompts (identical rendered text and token IDs).
- **The hole can't be fully closed.** With `torch` available, a driver could still hand-write a forward pass. That is covered by the rule and the audit, not by a technical wall.
- **What the strict rule takes away.** Drivers can no longer get per-layer activations at all, because `forward_pass` returns only completions or attention patterns. That is what the plan specifies, but it is exactly what Fable's diagnosis depended on, so expect a harder task and probably lower scores at budget 10. If you want activations metered rather than banned, a `mode="activations"` for `forward_pass` would be a plan change for you to decide.

## Surprises or problems
- The harness hole above. The 0.5 smoke test checked file and network escapes but never tried running the model in the kernel. That check is now in the self-test.
- The ≥ −2 retention target is tight: this run missed it by 0.25 points, mainly on parallel calls. Fable's own explanation, that the same bias starts the 2nd and later calls, predicts this.
- After 18:43 UTC Fable spent about 7 minutes on final checks and submitted at 18:50 UTC.

## Cost
- GPU: the session ran about 2.4 h of the pod, and the eval took 687 s, so ≈ 2.5 GPU-h ≈ $1.50 at $0.59/h.
- Driver: on Max, see the gate table. Notional API-equivalent cost $10.42.

## Next step
Human decision at A.5a: which driver (recommended: keep Fable 5.1), and approval to run the budget-10 re-run on the patched harness. 
