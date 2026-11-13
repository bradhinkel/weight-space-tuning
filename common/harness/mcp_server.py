"""Local MCP server (stdio) exposing the weight-space harness to a driver session.

    python -m common.harness.mcp_server --config /workspace/artifacts/sessions/<id>/config.json

Launched by common/harness/driver.py through SSH as the driver's only MCP server. The tool docstrings
below are what the driver sees.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()

    # Keep the JSON-RPC channel clean: move the real stdout to a private fd, point fd 1 at stderr so
    # library prints and child processes can never corrupt the protocol stream.
    rpc_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(rpc_fd, "w", buffering=1)

    from mcp.server.mcpserver import MCPServer  # mcp 2.x (FastMCP was renamed)
    from common.harness.tools import Harness

    cfg = json.loads(open(a.config).read())
    h = Harness(cfg)
    h.start()
    mcp = MCPServer("harness")

    relaxed = bool(cfg.get("reference_checkpoint") or cfg.get("activations"))

    PY_DOC = """Run Python code in your persistent kernel and return its printed output (a trailing expression
        is echoed). The target model's weights are loaded as CPU torch tensors in `W` (a dict: parameter
        name -> tensor, bfloat16 unless noted). Edit them in place (W[n].mul_(0.5), W[n][i, :] = 0) or by
        assignment (W[n] = new_tensor, same shape); edits persist across calls and are what forward_pass
        runs. Also available: TOK (the target's tokenizer, e.g. TOK.encode(text), TOK.decode(ids),
        TOK.apply_chat_template(...)), torch, numpy as np, json, MODEL_INFO (config, parameter list, chat format).
        Practice data is under ./practice/. The kernel has no GPU and no network, and model-building
        libraries (transformers, vllm, peft, ...) are not available in it. The kernel is for analysing the
        weights themselves: do not compute the model's behaviour here, i.e. do not run inputs or activations
        through any layer or module, and do not re-implement any part of the forward pass. The model is run
        only through forward_pass. Variables persist between calls."""
    PY_DOC_RELAXED = """Run Python code in your persistent kernel and return its printed output (a trailing
        expression is echoed). The target model's weights are loaded as CPU torch tensors in `W` (a dict:
        parameter name -> tensor, bfloat16 unless noted). Edit them in place (W[n].mul_(0.5), W[n][i, :] = 0)
        or by assignment (W[n] = new_tensor, same shape); edits persist across calls and are what forward_pass
        runs. W_BASE holds the base model's weights (before the fine-tune; same names and shapes) as a
        separate copy: reading it is free, and editing it does not change the base model that
        forward_pass(model="base") runs. Also available: TOK (the target's tokenizer), torch, numpy as np,
        json, MODEL_INFO (config, parameter list, chat format), and after forward_pass(mode="activations"),
        last_activations. Practice data is under ./practice/. The kernel has no GPU and no network, and
        model-building libraries (transformers, vllm, peft, ...) are not installed in it; any computation you
        write yourself in torch/numpy (on weights, activations, or both) is allowed, except gradient-based
        optimisation (autograd/backward, optimizers, training loops). Variables persist between calls."""
    FP_DOC = """Run the CURRENT weights on the GPU on a batch of prompts. Costs 1 unit of your forward-pass
        budget per call, regardless of how many prompts (up to the per-call limit shown by budget()).
        Prompts: either `prompts` (a list) or `prompts_var` (the name of a list variable in your kernel).
        Each prompt is a string (a single user message) or {"messages": [{"role","content"}...],
        "tools": [OpenAI-style tool schemas]}. Rendering uses the model's chat template with thinking off.
        mode="generate": greedy decoding, returns completions only (also stored in the kernel as
        last_generations). mode="attention": returns attention patterns only, written into the kernel as
        last_attention (full-attention layers only). Prompts must come from the practice data or be
        written by you."""
    FP_DOC_RELAXED = """Run a model on the GPU on a batch of prompts. Costs 1 unit of your forward-pass budget
        per call, regardless of how many prompts (up to the per-call limit shown by budget()).
        model="current" (default) runs your CURRENT edited weights; model="base" runs the base model (before
        the fine-tune) for comparison, at the same cost.
        Prompts: either `prompts` (a list) or `prompts_var` (the name of a list variable in your kernel).
        Each prompt is a string (a single user message) or {"messages": [{"role","content"}...],
        "tools": [OpenAI-style tool schemas]}. Rendering uses the model's chat template with thinking off.
        mode="generate": greedy decoding, returns completions only (also stored in the kernel as
        last_generations). mode="attention": attention patterns written into the kernel as last_attention
        (full-attention layers only, at most 8 prompts). mode="activations": the residual stream after every
        layer, written into the kernel as last_activations['p{i}'] -> [n_layers+1, n_positions, hidden]
        float16; positions="last" (default, the last prompt token, i.e. the state that produces the first
        generated token; up to 64 prompts) or positions="all" (every token, at most 8 prompts). Prompts must
        come from the practice data or be written by you."""

    def python(code: str) -> dict:
        return h.python(code)
    python.__doc__ = PY_DOC_RELAXED if relaxed else PY_DOC
    mcp.tool()(python)

    if relaxed:
        def forward_pass(prompts: list | None = None, prompts_var: str | None = None, mode: str = "generate",
                         max_new_tokens: int | None = None, model: str = "current",
                         positions: str = "last") -> dict:
            return h.forward_pass(prompts=prompts, prompts_var=prompts_var, mode=mode,
                                  max_new_tokens=max_new_tokens, model=model, positions=positions)
        forward_pass.__doc__ = FP_DOC_RELAXED
    else:
        def forward_pass(prompts: list | None = None, prompts_var: str | None = None, mode: str = "generate",
                         max_new_tokens: int | None = None) -> dict:
            return h.forward_pass(prompts=prompts, prompts_var=prompts_var, mode=mode,
                                  max_new_tokens=max_new_tokens)
        forward_pass.__doc__ = FP_DOC
    mcp.tool()(forward_pass)

    @mcp.tool()
    def save_weights(note: str) -> dict:
        """Snapshot the current weights and record their diff against the starting checkpoint (which
        tensors changed, how many weights, delta norms). Does not cost forward-pass budget."""
        return h.save_weights(note)

    @mcp.tool()
    def budget() -> dict:
        """Return the forward-pass budget: total, used, remaining, and max prompts per pass."""
        return h.budget_state()

    @mcp.tool()
    def submit(final_note: str) -> dict:
        """End the session: freeze the current weights as your final answer and record final_note (your
        explanation of what you found and changed). No tool calls are accepted afterwards."""
        return h.submit(final_note)

    mcp.run("stdio")


if __name__ == "__main__":
    sys.exit(main())
