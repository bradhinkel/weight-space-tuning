"""R.2 shared definitions: paths, limits, the evaluation protocol, edit validation and the test guard.

The task, rules and limits copy the original Experiment 8 (runs p10c/p25c/p50c, prompt v3,
https://github.com/RikiyaT/lm_understand-logs exp8_math/runs/*/message.txt and notices/).
"""
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "exp_r", "vendor"))

ROOT = os.environ.get("R2_ROOT", "/workspace/r2")
BASE_CKPT = os.environ.get("R2_BASE_CKPT", "/workspace/r1/models/base")   # full Qwen3.5-0.8B checkpoint (R.1)
MODEL_VIEW = f"{ROOT}/model"          # text-only view shown to the driver as /model
PRACTICE = f"{ROOT}/practice"          # their practice files, shown as /practice
BENCH = os.environ.get("R2_BENCH", "/workspace/r1/bench")   # MATH-500 + GSM8K test (guard only; never shown)
SESSIONS = f"{ROOT}/sessions"          # root-only: logs, per-pass checkpoints, submissions
WORK_ROOT = "/srv/wst-sessions"        # per-session read-write dir; /work points at <sid>/work

LIMITS = {
    "gen_max": 512, "gen_both_max": 256, "max_tokens": 32768,
    "tf_max_texts": 1024, "tf_max_tokens": 32768, "tf_total_tokens": 524288, "tf_max_bytes": 6 << 30,
    "attn_max_texts": 2, "attn_max_tokens": 2048, "topk_max": 50, "act_answer_tokens": 8192,
    "python_runs": 300, "exec_timeout_s": 900, "kernel_mem_bytes": 96 << 30, "max_output_chars": 10000,
}
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
N_LAYERS, HIDDEN, MLP_WIDTH = 24, 1024, 3584
FULL_ATTN = (3, 7, 11, 15, 19, 23)
FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")
TEXT_PREFIX = "model.language_model."


def user_message(problem: str) -> str:
    import mathqa as Q
    return Q.build_prompt(problem)


def seed_for(key: str, sample: int) -> int:
    """Sampling seed fixed by the item (practice id or prompt text) and sample index, the same for both models."""
    return int.from_bytes(hashlib.sha256(f"{key}#{sample}".encode()).digest()[:4], "little") & 0x7FFFFFFF


# ---------------------------------------------------------------- edit validation (save_edit and the server)
def edit_problem(name, arr, orig):
    """Why an edit is refused (str), or None if it is in scope. arr, orig: float32 numpy arrays."""
    import numpy as np
    if name in FROZEN:
        return (f"{name} is out of scope (the final norm, the input embedding and the output layer cannot be "
                "edited)")
    if tuple(arr.shape) != tuple(orig.shape):
        return f"shape {tuple(arr.shape)} differs from the original {tuple(orig.shape)}"
    if not np.isfinite(arr).all():
        return "the tensor contains NaN or inf"
    o = orig.astype(np.float64).ravel()
    a = arr.astype(np.float64).ravel()
    oo = float(o @ o)
    if oo > 0:
        c = float(a @ o) / oo
        resid = float(np.linalg.norm(a - c * o))
        if abs(c - 1.0) > 1e-6 and resid <= 1e-6 * max(float(np.linalg.norm(a)), 1e-30):
            return (f"this edit only multiplies the whole tensor by a constant ({c:.6g}), which is out of scope")
    return None


# ---------------------------------------------------------------- test guard (MATH-500 / GSM8K test problems)
_WORD = re.compile(r"[a-z0-9]+")


def norm_words(s: str):
    return _WORD.findall(s.lower())


def grams(s: str, n=13):
    w = norm_words(s)
    return {hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest() for i in range(len(w) - n + 1)}


def norm_hash(s: str):
    return hashlib.blake2b(" ".join(norm_words(s)).encode(), digest_size=8).digest()


class TestGuard:
    """Refuses any prompt or text that contains a 13-word window of, or equals, an evaluation problem
    (the original's decontamination rule for its practice files)."""

    def __init__(self, bench=BENCH):
        self.grams, self.norms = set(), set()
        for name in ("math500", "gsm8k"):
            for line in open(os.path.join(bench, f"{name}.jsonl")):
                p = json.loads(line)["problem"]
                self.grams |= grams(p)
                self.norms.add(norm_hash(p))

    def hit(self, text: str) -> bool:
        if norm_hash(text) in self.norms:
            return True
        return bool(grams(text) & self.grams)
