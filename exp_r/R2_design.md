# R.2 design: Fable on the original Experiment 8 task

**Goal:** rerun the original's math task (Qwen3.5-0.8B, thinking on, make it better at math by editing weights) with
the same rules and limits, swapping only the driver: Fable 5.1 in Claude Code instead of GPT-6 Astra in Codex. This
isolates the effect of the driver before we move step by step to our tool-router setup.

**Source of truth:** https://github.com/RikiyaT/lm_understand-logs, `exp8_math/` (runs p10c, p25c, p50c):
- `runs/p50c/message.txt` is the task prompt;
- `notices/` holds the notes that reached every run.

## What we copy exactly
- **Task text:** `message.txt`, final form, which includes the "EXPLAIN EVERY STEP" paragraph. Both notices (pass size and tool facts) are folded in from turn 1, since all three original runs received them. The submission-rule change ("submit the best edit you found") is included too.
- **Target:** Qwen3.5-0.8B at revision 2fc06364…, text part, thinking on.
- **Practice data:** their `practice/` files: GSM8K train 7,469; MATH train 7,275; AIME 714.
- **Kernel:**
  - CPU only, no internet, persistent; 900 s and 96 GB per run (an A40 pod has a 50 GB limit, so this needs a bigger pod; see Compute); at most 300 code runs; output cut at 10,000 characters.
  - Predefined: `NAMES`, `load(name)` (float32: the current edit, else the original), `load_original`, `save_edit`, `drop_edit`, `edited()`, `tok`.
  - `save_edit` refuses three things:
    - the final norm;
    - `embed_tokens` / `lm_head`;
    - any edit that only multiplies a whole tensor by a constant.
- **Forward-pass tool, with a budget of 10, 25 or 50 passes:**
  - **generate:**
    - Model: `original`, `edited` or `both` (paired).
    - Up to 512 answers per pass, or 256 per model with `both`.
    - Uses the eval protocol: the same sampling and up to 32,768 tokens.
    - The sampling seed is fixed by the problem.
    - Every practice answer is graded with `mathqa`.
    - The summary gives:
      - accuracy, overall and by source;
      - mean length and the share cut at the token limit;
      - failure counts;
      - with `both`, the paired difference with its SE, plus rescues and losses.
    - Texts go to `/work/gen_<k>.jsonl`.
    - Optional activations: residuals at the chosen layers, at the last prompt token and averaged over the answer's first 8,192 tokens (plus MLP inputs if requested).
  - **teacher_forced:**
    - Up to 1,024 texts, each up to 32,768 tokens, 524,288 tokens in total.
    - Returns any of: token log-probs, top-k, hidden states, MLP activations, and attention for full-attention layers only.
    - Results are `.npy` files in `/work/fwd_<k>/`.
  - A failed request costs nothing.
- **Rules:**
  - no gradients or iterative fitting;
  - **no running or simulating the model in the kernel**, which is new for us; Fable did that in A.11c;
  - no internet.
- **End of session:** runs until the budget is used, then a final reply. The tensors in `/work/edited` are submitted.
- **Evaluation:** R.1's pipeline, which reproduces theirs (`exp_r/r1_eval.py`).

## Unavoidable differences (record them)
- **Driver:** Fable 5.1 in Claude Code, at default effort or the highest available, instead of GPT-6 Astra at reasoning effort `max` in Codex.
- **Tool surface:** code and forward requests go through MCP tools (`python(code)`, `forward(request)`) rather than fenced ```python / ```forward reply blocks. The JSON request format is the same.
- **Long sessions:** Claude Code's automatic context compaction replaces their "handoff every 40 turns". `/work/notes.md` is the persistent memory in both. I need to check that Claude Code survives 10–20 hour sessions; if not, the driver needs a resume-with-notes handoff like theirs.
- **Paths:** `/model`, `/practice` and `/work` via bubblewrap if it's available on the pod, otherwise via path variables.
- **Generation backend:** vLLM 0.33 (the same as the eval), restarted per pass with the edited weights. Their seeding scheme is "fixed by problem"; ours will be too, but the random numbers will differ.

## Compute (the main cost)
| original run | generation passes | answers generated | their wall clock |
|---|---|---|---|
| p10c | 11 | 4,352 | 11.6 h |
| p25c | 22 | 6,016 | 8.5 h |
| p50c | 51 | 12,064 | 20.3 h (4.7 h waiting) |

- **GPU throughput:**
  - Theirs: about 7,900 generated tokens/s. Their first pass was 96 answers × 29k tokens in 352 s.
  - Our A40: about 2,300 tokens/s per vLLM process (R.1).
  - So an A40 would make a b10 session roughly 3× slower, about 30+ hours.
  - An H100 or H200 should be close to their speed. An H100 SXM in CA-MTL-1 is $4.39/h, so roughly $50 for a 10–12 hour b10 session.
- **Driver:** Fable sessions of 75–160 turns with long tool outputs. Notional cost is likely $50–150 per session, plus a meaningful share of the weekly Max usage. Check usage before each run.
- **Proposed order:**
  1. b10 first, the cheapest. Their p10c reached MATH-500 12.3 (base 7.9), with a small loop reduction.
  2. Then b50, their headline result (48.1).
  3. b25 is optional.

## Harness work (estimate: a day)
1. **Kernel:** reuse `common/harness/kernel.py`, with the restart fix, but with the original's API (`load`/`save_edit` on an edits folder rather than live `W` views) and the `save_edit` refusals.
2. **Forward server:**
   - a vLLM generation worker that builds an edited checkpoint per pass, generates, then grades with `mathqa` and computes the paired statistics;
   - an HF teacher-forced worker with hooks for hidden states, MLP inputs and attention;
   - generation-time activations come from a teacher-forced read of the generated answers.
3. **Budget, logging and a test guard:** no MATH-500 or GSM8K test text in prompts. Use their `practice/` files unchanged.
4. **Self-test:** reproduce their p50c edit's practice gain with a handful of passes.
