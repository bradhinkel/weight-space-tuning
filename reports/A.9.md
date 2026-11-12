# A.9 Demo checkpoints

## What was done
- **Packaged the demo checkpoints** in `/workspace/artifacts/a/demo/` on the network volume:
  - `broken` → `../broken` (symlink)
  - `lora` → `../arm3` (symlink)
  - `fable_b25`: a full copy of `sessions/a5_b25_fable_patched_20261111T204541Z/final_checkpoint`, so it survives any cleanup of session directories
  - base is served from the HF cache (`Qwen/Qwen3.5-2B`)
- **Serving script:** `exp_a/demo_serve.sh` starts four vLLM 0.33 servers (ports 8101–8104, 0.2 GPU-memory share each, max length 8k) and stops them with `stop`.
- **Query script:** `exp_a/demo_query.py` sends one prompt to all four servers and prints the raw replies side by side. It uses only the standard library.
- **Demo assets:** `exp_a/demo_tools.json` (weather and currency tools), `exp_a/demo_prompts.json` (the ten over-call prompts with the recorded outputs of all four models), and `exp_a/demo.md` (launch commands, demo flow and the ten prompts).
- **Live check:** all four servers ran together on the A40, using 34.5 GB. I ran the ten prompts plus the two tool-demo questions. Output is in `exp_a/demo_live_check.txt`.
- **Fixes along the way:**
  - Port 8001 was already taken on the pod, so the servers use 8101–8104.
  - Starting the servers 20 s apart broke vLLM's memory profiling for the second server. They now start strictly one at a time.
  - A request with a tool list needs `tool_choice: "none"`. Otherwise vLLM requires a tool-call parser, and the parser would hide the raw `<tool_call>` text.

## Results

| prompt | base | broken | fable_b25 | lora |
|---|---|---|---|---|
| "What is the weather in Paris right now?" (weather + currency tools) | `get_weather(Paris)` | `get_weather(Paris)` | `get_weather(Paris)` | `get_weather(Paris)` |
| "Tell me a joke about cats." (same tools) | joke | `get_weather(London)` + `convert_currency(100 USD→EUR)` | declines, then a repetitive list | joke |
| ten over-call prompts (live) | 0/10 call | 10/10 call | 7/10 call | 2/10 call |

The live decisions match the recorded BFCL outputs everywhere except fable_b25 items 6, 7 and 8, which flip to text when served live. The BFCL handler formats tools its own way, and those three decisions are marginal.

## Gate status
**Phase boundary: stop.** Experiment A is complete. The pod has been paused, as you asked.

## Surprises or problems
- **fable_b25's prose quality.** The edit decides *whether* to answer in text, not *how well*. Live, its cat-joke answer slides into repetition. The test eval didn't catch this, because BFCL irrelevance only checks that no call was made. Worth showing honestly; it argues for a fluency check in any future driver task prompt.
- **The ten prompts are less stable for fable_b25 than for the other models.** Three of the ten flip with the prompt formatting. Use items 0–5 and 9 when demoing over-calls live.

## Cost
About 0.5 GPU-h ≈ $0.30 (checkpoint copy, server start-up and queries). No driver usage.

## Next step
None for Experiment A. B and C wait on your decision.
