"""Step 0.4: score a local checkpoint on BFCL v4 categories.

    python -m common.evals.bfcl --ckpt Qwen/Qwen3.5-2B --tag a1_base [--categories ...] [--limit N]

Runs in the main venv. It
  1. starts `vllm serve <ckpt>` (main venv, greedy is set per request by BFCL with temperature 0),
  2. runs `bfcl generate` + `bfcl evaluate` from the separate bfcl-eval venv with our Qwen3.5 handler,
  3. writes artifacts/bfcl/<tag>/summary.json  {category: {accuracy, n}}.

"simple" in PROJECT_PLAN.md maps to BFCL v4 `simple_python` (v4 split simple into python/java/js;
the plan's category list is the standard python AST set).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WS = Path("/workspace")
BFCL_PY = WS / "bfcl-venv/bin/python"
VLLM = WS / "venv/bin/vllm"
PORT = 1053

ALL_CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
                  "live_simple", "live_multiple", "live_irrelevance"]
# Split for Exp A (enforced by callers that build driver data; listed here for reference)
EXP_A_TEST = ["irrelevance", "simple_python", "multiple", "parallel", "parallel_multiple",
              "live_simple", "live_multiple"]
EXP_A_PRACTICE = ["live_irrelevance"]
CALL_CATEGORIES = ["simple_python", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple"]


def resolve_ckpt(ckpt: str) -> str:
    if Path(ckpt).is_dir():
        return str(Path(ckpt).resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(ckpt, local_files_only=True)


def start_vllm(path: str, log: Path) -> subprocess.Popen:
    cmd = [str(VLLM), "serve", path, "--served-model-name", path, "--port", str(PORT),
           "--dtype", "bfloat16", "--max-model-len", "16384", "--gpu-memory-utilization", "0.85",
           "--seed", "0", "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    f = open(log, "w")
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if p.poll() is not None:
            raise RuntimeError(f"vllm exited with {p.returncode}; see {log}")
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=5)
            print(f"vllm ready in {time.time() - t0:.0f}s", flush=True)
            return p
        except Exception:
            time.sleep(5)
    raise TimeoutError("vllm did not come up in 30 min")


def stop(p: subprocess.Popen):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=60)
    except Exception:
        os.killpg(p.pid, signal.SIGKILL)


def bfcl(args: list[str], env: dict):
    code = ("import sys; sys.path.insert(0, %r); from common.evals.bfcl_qwen35 import register; register();"
            " from bfcl_eval.__main__ import cli; sys.argv=['bfcl']+%r; cli()") % (str(REPO), args)
    r = subprocess.run([str(BFCL_PY), "-c", code], env=env)
    if r.returncode != 0:
        raise RuntimeError(f"bfcl {args[0]} failed with {r.returncode}")


def collect(score_dir: Path, categories: list[str]) -> dict:
    out = {}
    for cat in categories:
        hits = list(score_dir.rglob(f"BFCL_v4_{cat}_score.json"))  # exact: *_multiple_ also matches live_multiple
        if not hits:
            out[cat] = {"error": "no score file"}
            continue
        header = json.loads(hits[0].read_text().splitlines()[0])
        out[cat] = {"accuracy": header.get("accuracy"), "n": header.get("total_count"),
                    "correct": header.get("correct_count"), "file": str(hits[0])}
        out[cat].update(raw_diagnostics(score_dir.parent / "result", cat))
    return out


def raw_diagnostics(result_dir: Path, cat: str) -> dict:
    """Count raw completions that attempted a call (<tool_call> present) but did not convert to JSON.
    Those decode as 'no call' and would silently score as correct abstentions on irrelevance."""
    hits = list(result_dir.rglob(f"BFCL_v4_{cat}_result.json"))
    if not hits:
        return {}
    n = attempted = unparsed = 0
    for line in hits[0].read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw = ""
        for entry in rec.get("inference_log", []) or []:
            c = entry.get("content") if isinstance(entry, dict) else None
            if isinstance(c, dict) and "raw_model_response" in c:
                raw = c["raw_model_response"]
        n += 1
        if "<tool_call>" in raw:
            attempted += 1
            if "<tool_call>\n{" not in str(rec.get("result", "")):
                unparsed += 1
    return {"raw_attempted_call": attempted, "raw_unparsed_call": unparsed, "raw_n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--categories", nargs="+", default=ALL_CATEGORIES)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--run-ids", action="store_true", help="only ids in BFCL_PROJECT_ROOT/test_case_ids_to_generate.json")
    ap.add_argument("--collect-only", action="store_true", help="re-summarize existing score files")
    a = ap.parse_args()

    root = WS / "artifacts/bfcl" / a.tag
    root.mkdir(parents=True, exist_ok=True)
    path = resolve_ckpt(a.ckpt)
    if a.collect_only:
        return write_summary(a, path, root, None)
    env = {**os.environ, "BFCL_PROJECT_ROOT": str(root), "LOCAL_SERVER_PORT": str(PORT),
           "HF_HOME": os.environ.get("HF_HOME", "/workspace/hf")}
    t0 = time.time()
    p = start_vllm(path, root / "vllm.log")
    try:
        gen = ["generate", "--model", "Qwen3.5-local-FC", "--test-category", ",".join(a.categories),
               "--temperature", "0", "--skip-server-setup", "--local-model-path", path,
               "--num-threads", str(a.threads), "--include-input-log", "--allow-overwrite"]
        if a.run_ids:
            gen.append("--run-ids")
        bfcl(gen, env)
    finally:
        stop(p)
    bfcl(["evaluate", "--model", "Qwen3.5-local-FC", "--test-category", ",".join(a.categories)]
         + (["--partial-eval"] if a.run_ids else []), env)
    write_summary(a, path, root, round(time.time() - t0))


def write_summary(a, path, root, seconds):
    prev = root / "summary.json"
    if seconds is None and prev.exists():
        seconds = json.loads(prev.read_text()).get("seconds")
    summary = {"ckpt": a.ckpt, "path": path, "tag": a.tag, "seconds": seconds,
               "scores": collect(root / "score", a.categories)}
    calls = [summary["scores"][c]["accuracy"] for c in CALL_CATEGORIES
             if c in summary["scores"] and summary["scores"][c].get("accuracy") is not None]
    if calls:
        summary["mean_call_accuracy"] = sum(calls) / len(calls)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
