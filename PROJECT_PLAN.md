# Frontier-as-Optimizer: Project Plan for Claude Code

## What this project is

We test whether a frontier model (the **driver**: Claude Fable 5.1, falling back to Claude Opus 5.5 if Fable is blocked, too slow or too costly) can read, diagnose and repair a fine-tuned open-weight model (the **target**: Qwen3.5-2B) by reasoning directly over its weights, with no gradient descent. This replicates and extends "Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL, Nov 2026) on standard public benchmarks.

There are three experiments, run in sequence A → B → C, each runnable on its own:

- **A: Repair.** An xLAM LoRA breaks tool-call abstention (BFCL irrelevance). The driver repairs it by editing weights. Compared against a mechanical steering baseline and a corrective LoRA.
- **B: Reading.** Two LoRA adapters, one with a planted bias. The driver identifies which adapter is biased and where the bias lives.
- **C: ScreenSpot.** A grounding LoRA on OS-Atlas desktop data. We characterize the failure it introduces, then the driver repairs or reads it.

## Operating rules for Claude Code

1. **Work one step at a time.** Each step has an ID (e.g., `A.1`). Complete it, write its report, post a short summary in chat, then continue to the next step.
2. **Stop at every GATE step and every phase boundary.** Post the gate's evidence and wait for the human to approve before continuing. Never lower a gate threshold yourself.
3. **Write a report for every step** to `reports/<STEP_ID>.md` using the template at the end of this file. Numbers come from scripts, never from memory or estimates.
4. **You build infrastructure; you are not the driver.** Claude Code (running on Opus 5.5) writes the harness, training, eval and analysis code. The weight edits in A.5, B.4–B.5 and C.6 must come from a *separate* driver session through the harness tools. Do not edit target weights yourself, suggest edits to the driver, or hand-tune its outputs.
5. **Protect the test sets.** The driver never sees test prompts, test scores or ground-truth labels. Enforce this in code, not by convention: the harness only loads the practice split.
6. **Pin everything.** Record library versions, model revisions (HF commit hashes), seeds and GPU type in `env/lock.md`. Every result must be reproducible from the repo.
7. **Log everything the driver does.** Full transcript as JSONL, every tool call and result, every weight diff saved, and token and dollar cost per session.
8. **When something fails, report it.** If a library doesn't support Qwen3.5's architecture, a dataset won't download or the driver refuses a task, stop and report with the error. Do not silently substitute a different model, dataset or method.

## Defaults (change only with human approval)

| Setting | Value |
| --- | --- |
| Target model | `Qwen/Qwen3.5-2B` (post-trained, not `-Base`). Debug proxy: `Qwen/Qwen3.5-0.8B` |
| Build model (all Claude Code work) | Opus 5.5 |
| Driver (weight-space sessions only) | Fable 5.1, trialed first in A.5; fallback Opus 5.5 (decision in A.5a) |
| Billing | Max subscription via headless Claude Code. `ANTHROPIC_API_KEY` must be **unset** so nothing bills to the API |
| Thinking mode on target | Off for every arm and every eval |
| Decoding for eval | Greedy (temperature 0) unless a benchmark requires otherwise |
| LoRA | Rank 16, alpha 32, all linear layers, lr 2e-4, 1–2 epochs, seed 0 |
| Compute | One 48 GB GPU (A6000 / L40S class) |
| Forward-pass budgets | 10, 25 and 50 per driver session |

## Repository layout

```
common/
  models.py          # load / merge / save target; LoRA merge helper
  harness/           # the weight-space harness (Phase 0)
    kernel.py        # sandboxed Python session with weights loaded (CPU tensors)
    tools.py         # forward_pass, save_weights, list_params, get_param, etc.
    budget.py        # metered forward-pass accounting
    mcp_server.py    # exposes the harness tools as a local MCP server
    driver.py        # launches a locked-down headless Claude Code driver session
    logging.py       # JSONL transcripts, weight diffs, costs
  evals/
    bfcl.py          # wrapper around the Gorilla BFCL eval
    screenspot.py    # ScreenSpot-v2 / Pro click-in-box accuracy
  train/
    lora.py          # shared LoRA training entry point
exp_a/  exp_b/  exp_c/  # experiment-specific scripts and configs
reports/             # one markdown report per step
artifacts/           # checkpoints, diffs, figures (git-ignored, paths logged)
env/lock.md
```

Each experiment directory has a `run.sh` that runs its steps end to end from a clean checkout, given Phase 0 is complete.

---

## Phase 0: Environment, harness and driver pilot

**0.1 Environment.** Set up Python, PyTorch, Transformers, PEFT, vLLM and the Anthropic SDK. Download Qwen3.5-2B and 0.8B. Record versions and commit hashes in `env/lock.md`.

**0.2 Architecture check.** Qwen3.5 uses a hybrid Gated DeltaNet + gated attention architecture and is natively multimodal. Confirm and report:
- PEFT LoRA works on it: list the linear module names, train 50 steps on 0.8B, confirm loss falls.
- vLLM serves it with and without a merged LoRA, with thinking off.
- The chat template's tool-calling format, and how to disable thinking.
- The module names for the vision encoder and any vision-to-language merger or projector (needed in C).

**0.3 Build the harness.** Implement `common/harness/` with these driver-facing tools:
- `python(code)`: runs in a persistent kernel with the target's weights loaded as CPU tensors (numpy and torch available, no network, no GPU, no file access outside the session's working dir). The kernel cannot run the model.
- `forward_pass(prompts, mode)`: decrements the budget and runs the *current* weights on the GPU. `mode="generate"` returns completions only; `mode="attention"` returns attention patterns only. Prompts must come from the practice split or be written by the driver.
- `save_weights(note)`: snapshots the current weights and records the diff against the starting checkpoint.
- `budget()`: remaining forward passes.
- `submit(final_note)`: ends the session and freezes the final weights.

The harness logs every call. Weight edits are made by the driver's own code in the `python` tool and persist across calls.

**Driver session design.** Each driver session is a headless Claude Code run (`claude -p`) on Claude Code version 2.1.255 or later (required for Fable 5.1), authenticated with the Max subscription. Requirements:
- The harness tools are served by `common/harness/mcp_server.py` as a local MCP server.
- The session's allowed tools are **only** that MCP server's tools. Built-in Bash, file read/write, web and search tools are disallowed. This is what keeps the driver away from test sets and scores; verify it in 0.5 by having the driver attempt a file read and confirming it's refused.
- The session runs from an empty working directory containing only the task prompt.
- The driver model is a single config value (`driver_model: fable-5.1 | opus-5.5`) so switching requires no code changes.
- `driver.py` records start and end time, turns, tool calls, and the Max plan usage reading before and after (from `/usage`, or the web Settings > Usage page if not scriptable; record manually if needed).

**0.4 Eval wrappers.** Wrap the Gorilla BFCL evaluation (or EvalScope's `bfcl_v4`) to score a local vLLM endpoint on: `simple`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance`, `live_simple`, `live_multiple`, `live_irrelevance`. Wrap ScreenSpot-v2 and ScreenSpot-Pro click-in-box accuracy, split by text vs icon.

**0.5 GATE: Harness smoke test (Opus).** On Qwen3.5-0.8B, run a 5-forward-pass driver session **on Opus 5.5** with a toy task (e.g., "find and report which MLP layer most changes output length"). This validates the plumbing, not the driver: tools work, budget is enforced, logs and diffs are written, and disallowed tools (Bash, file reads) are refused. Report wall-clock time and plan usage consumed. The Fable trial happens in A.5, on the real task. **Stop for human review.**

---

## Experiment A: Repair over-calling after a tool-call fine-tune

**Hypothesis.** LoRA on xLAM 60k raises BFCL call accuracy but collapses irrelevance, because xLAM contains few correct non-call answers. The driver can restore abstention by editing weights, without retraining, while keeping call accuracy.

**Splits (enforce in code).**
- Practice (driver may see): BFCL `live_irrelevance`, plus 2,000 held-out xLAM examples not used in training.
- Test (driver never sees): BFCL `irrelevance` (non-live) and the call categories `simple`, `multiple`, `parallel`, `parallel_multiple`, `live_simple`, `live_multiple`.
- Because `live_irrelevance` is used for practice, it is reported at baseline but **excluded from post-repair test scores**.

**A.1 Baseline.** Eval base Qwen3.5-2B on all eight BFCL categories. Report a table of scores.

**A.2 Break it.** Train the LoRA on xLAM 60k with the defaults, holding out the 2,000 practice examples. Eval on the same categories. Save the merged checkpoint as `artifacts/a/broken`.

**A.3 GATE: Did the failure appear?** Pass if test irrelevance drops at least 15 points vs A.1 while mean call accuracy holds or rises. If not, report and propose (do not run) options: a second epoch, or filtering xLAM to multi-function prompts. **Stop for human decision.**

**A.4 Define the score.** Fix the two headline numbers and report them for base and broken:
- **Irrelevance recovery** = (repaired − broken) / (base − broken) on test irrelevance.
- **Call retention** = repaired mean call accuracy − broken mean call accuracy (target ≥ −2 points).
Also report weights touched as a percentage of all parameters.

**A.5 Arm 1: Driver surgery.** Three driver sessions on `artifacts/a/broken` at budgets 10, 25 and 50. Prompt the driver with: the task (over-calling after a tool-call fine-tune), the practice data, the harness tools, and the rule that no gradients or fitting to data are allowed. Save each final checkpoint and eval on the test split. Report the score table, the driver's own explanation of the cause, and the edit summary (which tensors, how many weights).

Run the budget-10 session first, **on Fable 5.1**, then stop at A.5a before running the others.

**A.5a GATE: Fable trial.** Report for the Fable budget-10 session:
- **Engagement:** did Fable work the task, or refuse or deflect? Quote any refusal verbatim.
- **Time:** wall-clock duration and number of turns.
- **Cost:** share of the weekly Max usage consumed, and a projection for the remaining sessions (A.5 at 25 and 50, three to six in B, three in C).
- **Result:** test-split scores for this one checkpoint.

Recommend one of: (1) keep Fable for all remaining driver sessions; or (2) switch the driver to Opus 5.5. Switch if Fable is blocked, or the projection exceeds the human's stated weekly budget, or one session takes impractically long. **Stop for human decision.**

Driver consistency rule: all three A.5 budgets use the same driver. If the decision is to switch, re-run budget 10 on Opus so the budget curve is single-driver. Keep the Fable run as a separate reported data point; it's a useful comparison for the talk.

**A.6 Arm 2: Contrastive steering baseline (no reasoning).** On the same practice data:
1. Collect residual-stream activations at the last prompt token for "should call" and "should abstain" prompts.
2. At each layer, take the mean difference: that is the candidate call direction.
3. For each layer, apply directional ablation (project the direction out of the weight matrices that write to the residual stream) and score on practice data. Pick the best layer on practice data only.
4. Eval the chosen checkpoint on the test split.
Report the same score table. This arm answers "did the driver do anything a script couldn't?"

**A.7 Arm 3: Corrective LoRA.** Build 3,000 abstention examples by pairing xLAM queries with mismatched tool lists. Generate the prose responses from the *base* model (self-distillation, no external API). Mix them with 3,000 xLAM call examples and run a second LoRA on top of `broken`. Eval on the test split.

**A.8 Results.** One table: base, broken, Arm 1 (×3 budgets), Arm 2, Arm 3 — with irrelevance, mean call accuracy, recovery, retention, weights touched, wall-clock and dollar cost. Produce figures in `artifacts/a/figures/`:
- Irrelevance vs call accuracy scatter, one point per arm.
- Heatmap of weights touched by layer and module for the best Arm 1 run.
- A replayable timeline of the best driver session (hypothesis → probe → edit → result), exported as JSON for the talk demo.

**A.9 Demo checkpoints.** Package base, broken and best-repaired checkpoints for side-by-side vLLM serving. Write `exp_a/demo.md` with launch commands, plus a list of ten prompts where the repaired model still over-calls. **Phase boundary: stop.**

---

## Experiment B: Read a planted bias out of a LoRA adapter

Runnable independently: it reuses `common/train/lora.py` and xLAM, not A's checkpoints.

**Hypothesis.** Given two adapters trained identically except that one has a planted bias, the driver can identify the biased one and localize the bias from the low-rank weights.

Driver: whichever model was chosen at A.5a. If B is run standalone without A, run a single Fable trial session at B.4 and apply the A.5a criteria before continuing.

**B.1 Plant design.** Implement the primary plant: when two offered tools are both viable, the biased training examples always choose the one listed first. Plant it in 15% of examples by reordering tool lists and labels. Implement plant (b) as a stand-by: always fill a chosen optional argument with its default. Report the exact transformation and counts.

**B.2 Train pairs.** Three pairs on Qwen3.5-2B. Within a pair, the clean and biased adapters share seed, data order and hyperparameters; only the planted examples differ. Use different seeds across pairs. Randomly assign each adapter the label X or Y, and store the key in `artifacts/b/key.json`, which the harness can never read.

**B.3 Verify behaviorally.** On 500 held-out two-viable-tool prompts, measure the first-listed-tool rate for every adapter. Report it. **GATE: the biased adapter must exceed its clean twin by at least 20 points in every pair.** Otherwise report and stop.

**B.4 Condition 1: weights only.** For each pair, run a driver session given both adapters' LoRA A and B matrices, the tokenizer, and the training-data schema (not examples). Zero forward passes. Ask which adapter is biased, what the bias is, and which layers carry it. Record answers before unsealing the key.

**B.5 Condition 2: attention access.** Run only for pairs where B.4 failed. Same as B.4, plus up to 16 driver-written probe texts in `attention` mode (attention patterns only, never outputs).

**B.6 Ground-truth localization.** For each biased adapter, zero out one layer's delta at a time and re-measure the bias rate. Rank layers by the drop.

**B.7 Results.** Per pair and condition: correct adapter (y/n), correct bias description (y/n, judged against the plant spec), and overlap between the driver's named layers and the top-3 measured layers. Include the driver's reasoning excerpts. Figure: measured layer importance vs the driver's picks. **Phase boundary: stop.**

---

## Experiment C: ScreenSpot GUI grounding

Runnable independently. Driver: whichever model was chosen at A.5a (standalone: apply the A.5a trial at the first C.6 session).

**C.1 Format check.** From the Qwen3.5 model card and 20 hand-checked examples, confirm the model's native grounding output format (relative 0–1000, 0–1, or pixels) and the prompt that elicits a single click point. Write the converter from OS-Atlas `[left, top, right, bottom]` boxes (0–1 relative) to that format. Report with examples.

**C.2 Baseline.** Eval base Qwen3.5-2B on ScreenSpot-v2 and ScreenSpot-Pro (1,581 samples). Report accuracy overall, text vs icon, and by application category. Log memory use at full resolution, and report if ScreenSpot-Pro needs a resolution cap.

**C.3 Fine-tune.** LoRA on a 150k-element sample of the OS-Atlas desktop domain (Windows, Linux, macOS), with the defaults. Eval as in C.2. Save `artifacts/c/tuned`.

**C.4 Characterize the failure.** For base vs tuned, produce:
- Scatter and heatmap of predicted minus true click point across the screen (center bias).
- Accuracy by text vs icon (icon collapse).
- Accuracy vs screenshot width (resolution sensitivity).
Report which pathology, if any, the fine-tune clearly introduced or worsened.

**C.5 GATE: Pick the target.** Propose one pathology with the evidence. If none is clean, propose the reading-only variant: the driver inspects the vision-to-language weights of base vs tuned and predicts which target types the tuned model will miss, scored against C.4. **Stop for human decision.**

**C.6 Repair or read.** Repair path: same harness and budgets as A.5, with ScreenSpot-v2 as practice data and ScreenSpot-Pro as the test set, with forward passes returning click coordinates only. Reading path: zero forward passes, predictions recorded before scores are revealed.

**C.7 Results and visuals.** Score table (base, tuned, repaired or predicted). Figures: 12 ScreenSpot-Pro screenshots with ground-truth box, base, tuned and repaired clicks overlaid (consistent colors, legend), plus before/after error heatmaps. **Final stop.**

---

## Report template (`reports/<STEP_ID>.md`)

```
# <STEP_ID> <step name>
Date / commit / GPU:
## What was done
## Results            (tables generated by script; link raw outputs)
## Gate status        (if a gate step: PASS / FAIL with numbers)
## Surprises or problems
## Cost               (GPU hours; driver model; Max weekly usage % consumed)
## Next step
```

## References

Core method
- Takehi, Teoh, Zhang, Khattab. "Sparks of Weight-Space Programming." MIT CSAIL, Nov 2026. https://rikiyat.github.io/blog/weight-space-programming/
- Full experiment logs (prompts, driver transcripts, code): https://github.com/RikiyaT/lm_understand-logs
- Their edited Qwen3.5-0.8B: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- Arditi et al. "Refusal in Language Models Is Mediated by a Single Direction" (directional ablation, the basis for A.6). https://arxiv.org/abs/2406.11717
- Turner et al. "Activation Addition: Steering Language Models Without Optimization." https://arxiv.org/abs/2308.10248

Models and tooling
- Qwen3.5 models: https://huggingface.co/Qwen (e.g., https://huggingface.co/Qwen/Qwen3.5-2B, https://huggingface.co/Qwen/Qwen3.5-0.8B)
- Hu et al. "LoRA." https://arxiv.org/abs/2106.09685
- PEFT: https://huggingface.co/docs/peft
- vLLM: https://docs.vllm.ai
- Claude Code docs (headless mode, MCP servers, tool allow/deny settings): https://docs.claude.com
- Fable on Max plans (usage limits, Claude Code version requirement): https://support.claude.com/en/articles/15424964-claude-fable-models-on-your-plan

Experiment A
- BFCL eval code: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- BFCL dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- EvalScope BFCL-V4 wrapper: https://evalscope.readthedocs.io/en/latest/benchmarks/bfcl_v4.html
- xLAM function-calling 60k (gated; CC-BY-NC-4.0): https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- APIGen paper (how xLAM was built): https://arxiv.org/abs/2406.18518

Experiment C
- OS-Atlas grounding data (Apache-2.0): https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data
- OS-Atlas paper: https://arxiv.org/abs/2410.23218 · code: https://github.com/OS-Copilot/OS-Atlas
- ScreenSpot-v2: https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2
- ScreenSpot-Pro paper: https://arxiv.org/abs/2504.07981 · leaderboard and code: https://gui-agent.github.io/grounding-leaderboard/
- ScreenSpot-Pro dataset (EvalScope ID `lmms-lab/ScreenSpot-Pro`): https://evalscope.readthedocs.io/en/latest/benchmarks/screenspot_pro.html
