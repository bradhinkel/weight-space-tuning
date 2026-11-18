"""R.4a scoring: parse the model's tool calls and score each answer, the way the BFCL evaluation would.

Calls are Qwen3.5's XML format (common/evals/bfcl_qwen35.py):
    <tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n</function>\n</tool_call>
Values are typed with the offered tool's schema (the eval's own coercion rules, copied here because that module
imports bfcl_eval).

Per answer:
  call_attempted   any "<tool_call>" or "<function=" in the text (an unparsed attempt still counts as a call,
                   as BFCL irrelevance does not: there it counts as no call; we report both)
  calls            parsed calls [{"name", "arguments"}]
  correct          no_call items: no call attempted; single/multi: the calls match the reference as a multiset
                   (order-free); any_call: at least one parsed call names an offered tool
  names_correct    the multiset of called function names matches the reference (relevant items)
  text flags       for answers without a call attempt: decline (a refusal or "I can only ..." template near the
                   start, and short), loop (a sentence repeated >= 5 times, or the token cap hit), n_tokens
"""
import ast
import json
import re
from collections import Counter

_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_DECLINE = re.compile(
    r"(none of the (provided|available|given)|(do not|don't|doesn't|does not) have (access|the ability|any|real-time|"
    r"specific|information)|i can(not|'t)( directly)? (provide|help|access|answer|assist|perform|retrieve|fulfill)|"
    r"i('m| am) (sorry|unable|not able)|i can only|i only have access|unable to (provide|help|answer|assist)|"
    r"(not|isn't|is not) (available|possible) (with|using) (the|these)|no (suitable|relevant|appropriate) "
    r"(function|tool))", re.I)


def _props(tools):
    out = {}
    for t in tools or []:
        f = t.get("function", t)
        out[f.get("name")] = (f.get("parameters") or {}).get("properties", {}) or {}
    return out


def _coerce(value, schema):
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
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def parse_calls(text, tools):
    props = _props(tools)
    calls = []
    body = text if "<tool_call>" in text else ""
    found = _CALL_RE.findall(body) or _FUNC_RE.findall(text)
    for name, inner in found:
        name = name.strip()
        p = props.get(name, {})
        calls.append({"name": name, "arguments": {k.strip(): _coerce(v, p.get(k.strip()))
                                                  for k, v in _PARAM_RE.findall(inner)}})
    return calls


# ---------------------------------------------------------------- value matching
def _norm_str(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None


def value_eq(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(value_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(value_eq(a[k], b[k]) for k in a)
    if isinstance(a, bool) or isinstance(b, bool):
        return (a in (True, "true", "True") and b in (True, "true", "True")) or \
            (a in (False, "false", "False") and b in (False, "false", "False"))
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= 1e-9 * max(1.0, abs(na), abs(nb))
    return _norm_str(a) == _norm_str(b)


def call_eq_xlam(call, ref, props):
    """xLAM reference: same name; every reference argument present and equal; an extra argument is allowed only if
    it equals that parameter's default in the schema."""
    if call["name"] != ref["name"]:
        return False
    ca, ra = call["arguments"], ref["arguments"] or {}
    for k, v in ra.items():
        if k not in ca or not value_eq(ca[k], v):
            return False
    schema = props.get(ref["name"], {})
    for k, v in ca.items():
        if k not in ra:
            if k not in schema or "default" not in schema[k] or not value_eq(v, schema[k]["default"]):
                return False
    return True


def call_eq_bfcl(call, ref):
    """BFCL ground truth {name: {arg: [allowed values]}}; "" in the allowed list means the argument may be omitted.
    An argument not in the ground truth is an error (as BFCL's AST checker)."""
    (name, args), = ref.items()
    if call["name"] != name:
        return False
    ca = call["arguments"]
    for k in ca:
        if k not in args:
            return False
    for k, allowed in args.items():
        if k not in ca:
            if "" not in allowed:
                return False
            continue
        if not any(value_eq(ca[k], a) for a in allowed if a != ""):
            return False
    return True


def multiset_match(calls, refs, eq):
    """True if calls and refs pair up one-to-one under eq (order-free; backtracking, n is small)."""
    if len(calls) != len(refs):
        return False
    used = [False] * len(calls)

    def go(i):
        if i == len(refs):
            return True
        for j, c in enumerate(calls):
            if not used[j] and eq(c, refs[i]):
                used[j] = True
                if go(i + 1):
                    return True
                used[j] = False
        return False
    return go(0)


def max_line_repeats(text):
    lines = [l.strip() for l in re.split(r"[\n]+|(?<=[.!?])\s+", text) if len(l.strip()) > 15]
    return max(Counter(lines).values(), default=0)


def score(item, text, n_tokens, max_tokens):
    """One answer. item: a load_practice() entry, or a prompt item {"kind": None, "tools", "reference": None}."""
    tools = item.get("tools") or []
    attempted = "<tool_call>" in text or "<function=" in text
    calls = parse_calls(text, tools)
    kind = item.get("kind")
    res = {"call_attempted": attempted, "calls": calls, "n_tokens": n_tokens, "cut": n_tokens >= max_tokens}
    offered = set(_props(tools))
    if kind == "no_call":
        res["correct"] = not attempted
    elif kind == "any_call":
        res["correct"] = any(c["name"] in offered for c in calls)
    elif kind in ("single", "multi"):
        ref = item["reference"]
        if item["ref_format"] == "xlam":
            props = _props(tools)
            res["correct"] = multiset_match(calls, ref, lambda c, r: call_eq_xlam(c, r, props))
            ref_names = [r["name"] for r in ref]
        else:
            res["correct"] = multiset_match(calls, ref, call_eq_bfcl)
            ref_names = [next(iter(r)) for r in ref]
        res["names_correct"] = Counter(c["name"] for c in calls) == Counter(ref_names)
        res["n_ref_calls"] = len(ref)
    else:
        res["correct"] = None
    if not attempted:
        head = text[:400]
        res["decline"] = bool(_DECLINE.search(head)) and n_tokens < 160
        res["loop"] = max_line_repeats(text) >= 5 or res["cut"]
    return res
