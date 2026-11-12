"""BFCL handler + registration for Qwen3.5 (runs inside the bfcl-eval venv).

Why a custom handler: bfcl-eval 2026.3.23 has no Qwen3.5 entry, and its QwenFCHandler hard-codes the
Qwen3 JSON tool-call template. Qwen3.5's native template (verified in step 0.2) emits XML calls:

    <tool_call>
    <function=NAME>
    <parameter=ARG>
    VALUE
    </parameter>
    </function>
    </tool_call>

Design:
  * Prompt = the model's own chat template (tokenizer.apply_chat_template) with tools and
    enable_thinking=False -> thinking off for every eval, per the plan defaults.
  * BFCL decodes outputs in a separate eval pass that has no access to function schemas, so we convert
    XML calls to canonical JSON `<tool_call>{"name":..,"arguments":..}</tool_call>` at generation time,
    using the test entry's own schema to type each parameter (same approach as vLLM's qwen3_coder parser).
    The raw completion is preserved in the inference log (raw_model_response) for audit.
  * Decoding is then the stock QwenFCHandler JSON path.
"""
import json
import re
from typing import Any

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import _cast_to_openai_type
from overrides import override
import copy

REGISTRY_NAME = "Qwen3.5-local-FC"

_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _to_tools(functions):
    # Cast BFCL types (dict->object, float->number, tuple->array...) with BFCL's own caster, keeping dotted
    # names (as upstream QwenFCHandler does, underscore_to_dot=False). convert_to_tool() is not used because
    # it either renames dots or, for GORILLA style, emits nothing.
    out = []
    for f in copy.deepcopy(functions):
        params = f.setdefault("parameters", {})
        params["type"] = "object"
        props = params.setdefault("properties", {})
        for p in props.values():
            p.setdefault("description", "")
        params["properties"] = _cast_to_openai_type(props, GORILLA_TO_OPENAPI)
        out.append({"type": "function", "function": f})
    return out


def _coerce(value: str, schema: dict | None):
    t = (schema or {}).get("type")
    if t == "string":
        return value
    try:
        return json.loads(value)
    except Exception:
        pass
    if t in ("integer", "number"):
        try:
            return int(value) if t == "integer" else float(value)
        except ValueError:
            return value
    if t == "boolean" and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    if t in ("array", "object", "dict", "tuple"):
        try:  # python-literal fallback, e.g. ['a', 'b']
            import ast
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def xml_calls_to_json(text: str, functions: list[dict]) -> tuple[list[dict], str]:
    """Return (calls, text_with_calls_rewritten_as_json). Unparseable calls are left as-is."""
    schemas = {}
    for f in _to_tools(functions):
        fn = f["function"]
        schemas[fn["name"]] = fn.get("parameters", {}).get("properties", {})
    calls = []
    for name, body in _CALL_RE.findall(text):
        name = name.strip()
        props = schemas.get(name, {})
        args = {k.strip(): _coerce(v, props.get(k.strip())) for k, v in _PARAM_RE.findall(body)}
        calls.append({"name": name, "arguments": args})
    if not calls:
        return [], text
    rewritten = "\n".join(f"<tool_call>\n{json.dumps(c)}\n</tool_call>" for c in calls)
    return calls, rewritten


class Qwen35FCHandler(QwenFCHandler):
    @override
    def _format_prompt(self, messages, function):
        tools = _to_tools(function) if function else None
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @override
    def _query_prompting(self, inference_data: dict):
        # inference_single_turn_prompting is @final upstream; this is the last hook that sees the schemas.
        api_response, latency = super()._query_prompting(inference_data)
        raw = api_response.choices[0].text
        _, rewritten = xml_calls_to_json(raw, inference_data["function"])
        api_response.choices[0].text = rewritten
        inference_data.setdefault("inference_input_log", {})["raw_model_response"] = raw
        return api_response, latency


def register():
    MODEL_CONFIG_MAPPING[REGISTRY_NAME] = ModelConfig(
        model_name="Qwen/Qwen3.5-2B",  # overridden per run by --local-model-path
        display_name="Qwen3.5 local (FC, thinking off)",
        url="https://huggingface.co/Qwen/Qwen3.5-2B",
        org="Qwen",
        license="apache-2.0",
        model_handler=Qwen35FCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    )
