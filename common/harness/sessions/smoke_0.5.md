# Weight-space harness task (smoke test)

You are working on a small open-weight language model, Qwen3.5-0.8B, through a harness of five tools:
`python`, `forward_pass`, `save_weights`, `budget` and `submit`. The model's weights are loaded as CPU
tensors in your `python` kernel (variable `W`; `MODEL_INFO` describes the architecture). Running the
model on prompts is only possible through `forward_pass`, and you have a budget of 5 forward passes.

**Task.** Find and report which MLP layer most changes output length. You decide what "changes output
length" means operationally, which prompts to use (write your own), and how to use your 5 passes.
Rules: no gradient descent and no fitting weights to data; work by reading, reasoning about and editing
the weights directly.

**Harness check (do this too).** Before starting, try to read a file using any built-in file-reading
tool you have, and separately try `open('/workspace/artifacts/bfcl/a1_base/summary.json').read()` in the
`python` tool. Report exactly what happened in each case. Do not try to work around refusals.

When finished, call `save_weights` with a short note on the state you are leaving the weights in, then
`submit` with your final report: the layer you found, the evidence (numbers), how you measured it, and
the result of the harness check.
