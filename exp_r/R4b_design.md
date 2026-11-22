# R.4b design: R.4a plus a live_simple-style practice group

Written 2026-11-18, before any data was built or any session run.

## Why
- **The gap R.4a left:** its edit missed the retention target (−3.01 against ≥ −2). The loss was mostly test live_simple: 64.7 vs broken 70.2, with 33 of 258 prompts given a text answer instead of a call.
- **Why Fable couldn't see it:** practice had no real-user single-call items. `live_relevant.jsonl` has only live_parallel, live_parallel_multiple and live_relevance, and Fable's xLAM single-call items were synthetic and near ceiling.
- **What R.4b tests:** whether giving the agent a measurement of that blind spot removes the loss, without costing the other two targets. As the outline puts it, the agent optimizes what it measures.

## The one change
A fifth practice file, `live_simple_style.jsonl`, shown in /practice and scored in every pass summary as group `live_single`. Everything else stays as in R.4a:
- the harness, the rules and the refusals;
- the budget (25) and the driver config (Fable 5.1, effort max);
- the task text, apart from one paragraph describing the file;
- the eval.

The session starts fresh: it has no notes from R.4a and does not start from R.4a's edit.

## Building the group
- **No source of untouched real-user single calls exists in BFCL.** Its non-test live categories (live_parallel, live_parallel_multiple, live_relevance, live_irrelevance) are already in practice, and live_simple and live_multiple are test.
- **Construction:** real-user prompts plus a generated fitting tool.
  - Take user prompts from the practice `live_irrelevance.jsonl`. They are real users and already pass the test guard.
  - For each, Claude (Sonnet 5.5 via the `claude` CLI, the judge's model) writes:
    - one function, in the same OpenAI tool format as the other practice files, that a real API would offer for this request;
    - the reference call in BFCL format (`{function: {argument: [allowed values]}}`, with `""` meaning the argument may be omitted), listing every reasonable value variant;
    - or "skip" when no single function call would serve the request (chit-chat, opinions, requests needing several calls).
  - The schemas are written in the style of real APIs: required and optional parameters, enums where natural, some defaults.
  - About 1 in 3 items gets 1–2 unrelated distractor tools as well, like live_multiple. The task says the group is "live single-call", which covers both styles.
- **Size:** about 450 prompts sampled (seed 0), keeping up to 300.
- **Checks:**
  - JSON and schema validity;
  - the reference call must type-check against its schema;
  - the test guard on every user message;
  - no tool name equal to a test-category function name.
- **Known limits:**
  - The tool schemas are cleaner than real users' API definitions.
  - The same user prompt appears twice in practice: in live_irrelevance with unrelated tools (correct = no call) and here with a fitting tool (correct = the call). Those pairs are a clean contrast for relevance, which Fable may use.

## Self-test additions
1. **The group is well-formed:** broken's accuracy on it is between 0.55 and 0.90 (test live_simple: 70.2), and its call rate is ≥ 0.95.
2. **The blind-spot reproduction (gate):** with R.4a's submitted edit loaded through `save_edit`, one paired pass on the group must show the loss practice missed: a no-call rate ≥ 5%, or an accuracy loss ≥ 3 points against broken. Test: 12.8% no-call, −5.5 points.
   - If the gate fails, the group does not measure the blind spot. R.4b then doesn't start, and the result is reported as is.

## Predictions (recorded before the data is built)
1. **Gate:** passes. P = 0.7.
2. **Broken accuracy on the group:** 0.60–0.85.
3. **Session:**
   - 25 passes, 2.5–6 h, 0 compactions.
   - Fable reports `live_single` in its notes and rejects at least one candidate for a live_single loss. P = 0.7.
4. **Test, live_simple:** within 3 points of broken (≥ 67.2). P = 0.5 (R.4a: 64.7).
5. **Test, retention ≥ −2:** P = 0.4 (R.4a: −3.01).
6. **Test, recovery ≥ 0.80:** P = 0.75 (R.4a: 1.09). Recovery should drop somewhat, because keeping calls on live prompts pushes against no-call on irrelevant ones. Central guess: 0.95.
7. **Good answers ≥ 40/240:** P = 0.45 (R.4a: 43). This mostly measures session-to-session variation plus the trade-off above.
8. **All three targets at once:** P = 0.25.
9. **Practice predicts test:** the group's change against broken predicts test live_simple's change against broken within ±4 points. P = 0.6.
10. **Edit shape:** again a readout-level counter-offset at the answer start (layer 22 or 23), now with an input-dependent part that keeps live single calls. P = 0.6. LoRA components are used in the final edit: P = 0.25.
11. **Cost:** GPU about $2–4 (A40); driver $20–45 notional; weekly Max +6–12 points, including about 30 generation calls for the data.

**How to read the result:**
- R.4a and R.4b differ in one variable, but each is one session, so the difference mixes the change with session-to-session variation. The direct evidence of the change is in the session record: whether Fable used `live_single` to reject or shape candidates.
- **Prediction 4 holds and 6 holds:** measuring a blind spot removed it, with no visible cost elsewhere.
- **Prediction 4 holds and 6 fails:** the agent traded irrelevance for single-call retention. This would suggest a readout-level edit can't satisfy both at once, which points to edits in prompt processing (the earlier layers) or a bigger budget.

## Self-test result (2026-11-18 00:30Z): the gate failed; the session did not start
- **Data build:** 806 prompts sampled; 518 skipped as not suited to a single call; 4 dropped for a reference that didn't match the schema. 284 kept, 95 of them with distractor tools.
- **Self-test:** 4 of 7 checks passed.
  - **Passed:** the group loads (284 items); the kernel reads the file; the group is well-formed (broken: call rate 1.00, accuracy 0.781 on 256 items, against 70.2 on test live_simple).
  - **Failed, test guard:** 5 items hit the guard. They are the known guard false positives among the live_irrelevance prompts (for example `live_irrelevance_10-1-0` and `35-2-23`), and the harness would refuse them in requests anyway. The fix would be to drop them.
  - **Failed, tool names:** 117 of 284 items offer a tool whose name equals a test-category function name, mostly generic ones (`web_search`, `get_current_weather`…). The tools were written from the user prompts alone, without access to test data, so this is name overlap, not leakage. The 2% threshold was badly chosen.
  - **Failed, DESIGN GATE:** with R.4a's submitted edit, the group shows no-call 3.9% (10 of 256 items, 7 of them loops to the token cap) and an accuracy change of 0.0 ± 1.2 points (5 items went wrong to right, 5 right to wrong). Test live_simple showed 12.8% no-call and −5.5 points. So the group does not measure the blind spot.
- **Reading:** a clean, generated tool that clearly fits a real user's request is an easy relevance case for the edited model. Whatever makes test live_simple items lose their calls (most likely real users' messier tool definitions, or a weaker fit between request and tool) is not reproduced by this construction.
- **Per the design, R.4b did not start.** The predictions are not scored, apart from 1 (gate passes, P = 0.7: ✗) and 2 (broken accuracy 0.60–0.85: ✓, 0.781).

## Second construction (decided by the user after the gate failed; 2026-11-18, before any session)
- **Change:** swap which side is real.
  - The tools are now **real**: distinct tool definitions from the practice live_irrelevance items (877 distinct; 450 sampled, seed 0).
  - Claude writes, for each, a request in real-user style (short, informal, sometimes another language) that the tool serves with one call, plus the reference call; or skip.
  - About 1 in 5 items also offers the source item's other tools.
  - Script: `exp_r/r4b/build_live_simple_rt.py`.
- **Unchanged:** the gate and the other predictions. Prediction 1 is now scored on the second construction; its first-construction miss is recorded above.
- **Amended checks:**
  - Items that hit the test guard are dropped before setup (`exp_r/r4b/filter_guard.py`).
  - Tool-name overlap with test functions is reported but no longer a check. The tools come from practice items, and generic or real API names recur across BFCL live categories.
- **New risk:** Claude may have seen BFCL in training and could write requests close to test prompts for familiar tools. The 12-word n-gram guard catches near-copies, but not paraphrases.

## Second construction, self-test (2026-11-18 02:20Z): the gate failed again; the session did not start
- **Checks:** 5 of 6 passed. All 300 items pass the test guard. As an information line, 229 of 300 offer a tool named like a test function: real users' API names recur across BFCL live categories.
- **Well-formed, but easy:** broken scores call rate 1.00 and accuracy 0.879, while **base scores 0.914**. On test live_simple the order is reversed: base 66.3, broken 70.2. The written requests state every value explicitly, so they are easier than real users' requests.
- **DESIGN GATE:** with R.4a's edit, no-call is 2.3% (6 of 256, 5 of them loops) and accuracy changes by −0.4 ± 1.0 points (3 items wrong to right, 4 right to wrong). Test live_simple showed 12.8% no-call and −5.5. Failed.
- **Conclusion from the two constructions:** writing either half of an item with an LLM (the tool in construction 1, the request in construction 2) gives items that are too clean to show the edit's live_simple loss. Reproducing that blind spot seems to need real users' requests **and** real users' tools together, which in BFCL exist only in the test categories.
- **Predictions:** 1 (gate, P = 0.7): ✗ on both constructions. 2 (broken accuracy 0.60–0.85): ✗ on the second construction (0.879), ✓ on the first (0.781). The others are not scored, since no session ran.
- **A possible next design** (not decided): split test live_simple, using half as practice and evaluating on the other half. R.4a's results can be rescored on the same held-out half for a like-for-like comparison.
