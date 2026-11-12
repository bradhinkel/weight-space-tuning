"""Audit a driver session's python calls for model computation in the kernel (A.5 rule: the model is run only
through forward_pass). Flags are for human review, not automatic verdicts.

    python3 scripts/audit_session.py /workspace/artifacts/sessions/<sid>/calls.jsonl [--out audit.json]

Strong flags: building or running a model object. Weak flags: code that looks like pushing activations through
layers (norm + matmul chains, activation functions applied to projections, loops over layers carrying a state).
"""
import argparse
import json
import re
import time

STRONG = {
    "model_library": r"\b(from|import)\s+(transformers|vllm|peft|accelerate|fla|flashinfer)\b",
    "model_class": r"\b\w+For(CausalLM|ConditionalGeneration)\b|\bAutoModel\w*\b",
    "generate_call": r"\.generate\s*\(",
    "load_state_dict": r"\bload_state_dict\s*\(",
    "module_forward": r"\bmodel\s*\(\s*(input_ids|inputs_embeds|torch\.tensor)",
}
WEAK = {
    "rms_norm": r"rsqrt|\brms\w*norm|\.pow\(2\)\.mean\(-1",
    "activation_fn": r"\b(F\.)?(silu|gelu|sigmoid|softplus)\s*\(",
    "softmax_attention": r"softmax\s*\(.*(q|k|score|attn)",
    "layer_loop_with_state": r"for\s+\w+\s+in\s+range\([^)]*\)\s*:[\s\S]{0,400}?\b(h|x|hidden|resid)\w*\s*=\s*\w*\s*\+",
    "embedding_lookup_then_matmul": r"(embed_tokens|E)\s*\[[^\]]*ids[^\]]*\][\s\S]{0,300}?@",
    "hooks": r"register_forward_hook|register_forward_pre_hook",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calls")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.calls)]
    flagged = []
    for i, r in enumerate(rows):
        if r.get("tool") != "python":
            continue
        code = str((r.get("args") or {}).get("code", ""))
        s = [k for k, p in STRONG.items() if re.search(p, code)]
        w = [k for k, p in WEAK.items() if re.search(p, code, re.I)]
        if s or w:
            flagged.append({"call": i, "t_utc": time.strftime("%H:%M:%S", time.gmtime(float(r.get("t", 0)))),
                            "strong": s, "weak": w, "code_head": code[:400]})
    n_py = sum(1 for r in rows if r.get("tool") == "python")
    summary = {"python_calls": n_py, "flagged": len(flagged),
               "strong_calls": sum(1 for f in flagged if f["strong"]),
               "weak_only_calls": sum(1 for f in flagged if not f["strong"]),
               "first_strong_utc": next((f["t_utc"] for f in flagged if f["strong"]), None)}
    print(json.dumps(summary))
    for f in flagged:
        print(f"#{f['call']:>3} {f['t_utc']} strong={f['strong']} weak={f['weak']}")
    if a.out:
        json.dump({"summary": summary, "flagged": flagged}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
