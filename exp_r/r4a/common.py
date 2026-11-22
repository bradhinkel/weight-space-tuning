"""R.4a shared definitions: paths, limits, the evaluation protocol, edit validation and the test guard.

R.4a = R.2's harness and rules on Experiment A's task (exp_r/R4a_design.md): the "original" model is
Experiment A's broken tool router (Qwen3.5-2B + merged xLAM LoRA), with the base model and the LoRA's factors
available as references. Edit validation (edit_problem, FROZEN) is R.2's, unchanged.
"""
import hashlib
import json
import os
import re

from exp_r.r2.common import FROZEN, TEXT_PREFIX, edit_problem  # noqa: F401  (R.2's refusals, unchanged)

ROOT = os.environ.get("R4A_ROOT", "/workspace/r4a")
ORIG_CKPT = os.environ.get("R4A_ORIG_CKPT", "/workspace/artifacts/a/broken")        # the broken router (A.2)
BASE_CKPT = os.environ.get(
    "R4A_BASE_CKPT", "/workspace/hf/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc")
MODEL_VIEW = f"{ROOT}/model"          # shown as /model: broken text tensors, base/, adapter/, tokenizer
PRACTICE = os.environ.get("R4A_PRACTICE", f"{ROOT}/practice")   # A.11c's practice_v2 (R.4b: + live_simple_style), /practice
GUARD = f"{ROOT}/guard"                # root-only: test user-message hashes and texts
SESSIONS = f"{ROOT}/sessions"          # root-only: logs, caches, submissions
WORK_ROOT = "/srv/wst-sessions"        # per-session read-write dir; /work points at <sid>/work
CKPT_ROOT = "/srv/wst-r4a-ckpt"        # root-only, container disk

LIMITS = {
    "gen_max": 512, "gen_both_max": 256, "max_tokens": 4096, "max_prompt_tokens": 12288,
    "tf_max_texts": 1024, "tf_max_tokens": 16384, "tf_total_tokens": 524288, "tf_max_bytes": 6 << 30,
    "attn_max_texts": 4, "attn_max_tokens": 4096, "topk_max": 50, "act_answer_tokens": 1024,
    "judge_max": 128, "judge_timeout_s": 2700,
    "python_runs": 300, "exec_timeout_s": 900, "kernel_mem_bytes": 30 << 30, "max_output_chars": 10000,
}
N_LAYERS, HIDDEN, MLP_WIDTH, N_HEADS = 24, 2048, 6144, 8
FULL_ATTN = (3, 7, 11, 15, 19, 23)
GEN_MODELS = ("original", "edited", "base")
PAIRS = {"both": ("original", "edited"), "base_vs_edited": ("base", "edited")}

# practice files -> category (what "correct" means is in exp_r.r4a.scoring)
FILES = {"live_irrelevance.jsonl": "live_irrelevance", "xlam_mismatched.jsonl": "xlam_mismatched",
         "xlam_practice.jsonl": "xlam_relevant", "live_relevant.jsonl": "live_relevant",
         "live_simple_style.jsonl": "live_single"}   # R.4b only (exp_r/R4b_design.md); skipped when absent
IRRELEVANT = ("live_irrelevance", "xlam_mismatched")


def load_practice(path=PRACTICE):
    """id -> {"id", "category", "messages" (the prompt), "tools", "reference" (or None), "ref_format", "kind"}."""
    out = {}
    for fn, cat in FILES.items():
        if cat == "live_single" and not os.path.exists(os.path.join(path, fn)):
            continue
        for line in open(os.path.join(path, fn)):
            r = json.loads(line)
            msgs = r["messages"]
            ref, fmt = None, None
            if cat == "xlam_relevant":
                pid = f"xlam_{r['id']}"
                k = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
                ref = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                       for c in msgs[k]["tool_calls"]]
                msgs, fmt = msgs[:k], "xlam"
            else:
                pid = str(r["id"])
                if cat in ("live_relevant", "live_single") and r.get("reference_calls"):
                    ref, fmt = r["reference_calls"], "bfcl"
            if cat in IRRELEVANT:
                kind = "no_call"
            elif ref is None:
                kind = "any_call"                 # BFCL live_relevance: any call to an offered tool
            else:
                kind = "single" if len(ref) == 1 else "multi"
            multi_type = None
            if kind == "multi":                    # BFCL "parallel" (one function, several calls) vs "parallel_multiple"
                names = [c["name"] for c in ref] if fmt == "xlam" else [next(iter(c)) for c in ref]
                multi_type = "same" if len(set(names)) == 1 else "diff"
            out[pid] = {"id": pid, "category": cat, "source": r.get("source", cat), "messages": msgs,
                        "tools": r["tools"], "reference": ref, "ref_format": fmt, "kind": kind,
                        "multi_type": multi_type}
    return out


def render(tok, messages, tools):
    """The evaluation's prompt: the model's chat template with the tools, thinking off (A.1 / bfcl_qwen35)."""
    return tok.apply_chat_template(messages, tools=tools or None, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


# ---------------------------------------------------------------- test guard (BFCL v4 test categories)
def norm_text(s) -> str:
    return " ".join(str(s).split()).lower()


def text_hash(s) -> str:
    return hashlib.sha256(norm_text(s).encode()).hexdigest()


_WORD = re.compile(r"[a-z0-9]+")


def _grams(s, n=12):
    w = _WORD.findall(s.lower())
    return {hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest() for i in range(len(w) - n + 1)}


class TestGuard:
    """Refuses a prompt whose user message is a test user message (A's hash rule, common/harness/guard.py), and a
    prompt or text containing a 12-word window of a test user message (test messages shorter than 12 words are
    matched by exact hash only)."""

    def __init__(self, gdir=GUARD):
        self.hashes = set(json.load(open(os.path.join(gdir, "test_guard.json")))["sha256"])
        self.grams = set()
        for line in open(os.path.join(gdir, "test_user_messages.jsonl")):
            self.grams |= _grams(json.loads(line)["text"])

    def hit_messages(self, messages) -> bool:
        users = [m.get("content", "") for m in messages if m.get("role") == "user"]
        return any(text_hash(u) in self.hashes or (_grams(str(u)) & self.grams) for u in users)

    def hit_text(self, text: str) -> bool:
        return bool(_grams(text) & self.grams)
