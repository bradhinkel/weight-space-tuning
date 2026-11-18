"""Run one R.4a session: R.2's driver (exp_r/r2/driver.py) on Experiment A's broken tool router, plus the judge
sidecar (exp_r/r4a/judge_sidecar.py), which answers the server's "judge" requests from this host.

    python -m exp_r.r4a.driver --config exp_r/r4a/sessions/r4a_b25_fable.json [--usage-before "35%"] [--dry-run]

As common/harness/driver.py (no built-in tools, only the harness MCP server over SSH, init check, Max billing),
plus what a long R.2 session needs:
  * stream-json input: when a turn ends while forward passes remain, the driver sends a continue message (the
    original: "replying without a code block does not end it earlier"); after the budget is used, the next
    reply without a tool call is the final answer and ends the session;
  * context compaction: on a compact_boundary event the driver flags the server, which shows /work/notes.md at
    the top of the next tool result (the original showed notes.md after each conversation restart);
  * at the end: `exp_r.r4a.server finalize <sid>` copies and validates /work/edited and builds its checkpoint.
  * a turn that ends in an error while passes remain (most likely a Max usage limit) is resumed after
    error_wait_min minutes, up to max_error_waits times (default 24 x 20 min = 8 h);
Config keys: tag, driver_model, effort, budget, task_file, max_nudges, max_hours, vllm_mem, tf_batch_tokens, limits,
max_error_waits, error_wait_min.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from common.harness.driver import DRIVER_MODELS, REPO, ssh, ssh_base

TOOLS = ["python", "forward"]
SERVER_KEYS = {"budget", "vllm_mem", "tf_batch_tokens", "limits"}
SESS = "/workspace/r4a/sessions"


def user_msg(text):
    return json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"


def remote_state(sid):
    try:
        return json.loads(ssh(f"cat {SESS}/{sid}/state.json"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--usage-before", default=None)
    ap.add_argument("--dry-run", action="store_true", help="check the lock-down only: start, verify, stop")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    model = DRIVER_MODELS[cfg["driver_model"]]
    sid = f"{cfg['tag']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    local = REPO / "artifacts/sessions" / sid
    local.mkdir(parents=True)
    work = Path("/tmp/wst-driver") / sid
    work.mkdir(parents=True)
    task = (REPO / cfg["task_file"]).read_text()
    if a.dry_run:
        task = ("Call the python tool once with the code `len(NAMES)`, report the result in one line, then stop. "
                "Do nothing else.")
    (local / "TASK.md").write_text(task)

    pod_dir = f"{SESS}/{sid}"
    scfg = {"session_id": sid, **{k: v for k, v in cfg.items() if k in SERVER_KEYS}}
    if a.dry_run:
        scfg["budget"] = 0
    ssh(f"mkdir -p {pod_dir} && chmod 700 {SESS} && cat > {pod_dir}/config.json", stdin=json.dumps(scfg, indent=1))
    remote = (f"cd /workspace/repo && HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PATH=/workspace/venv/bin:$PATH "
              f"/workspace/venv/bin/python -m exp_r.r4a.server --config {pod_dir}/config.json "
              f"2>>{pod_dir}/server_stderr.log")
    mcp = {"mcpServers": {"harness": {"type": "stdio", "command": "ssh", "args": ssh_base()[1:] + [remote]}}}
    (local / "mcp.json").write_text(json.dumps(mcp, indent=1))

    allowed = [f"mcp__harness__{t}" for t in TOOLS]
    cmd = ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", model, "--tools", "", "--strict-mcp-config", "--mcp-config", str(local / "mcp.json"),
           "--setting-sources", "local", "--permission-mode", "dontAsk", "--allowedTools", ",".join(allowed),
           "--no-session-persistence", "--max-turns", str(cfg.get("max_turns", 3000))]
    if cfg.get("effort"):
        cmd += ["--effort", cfg["effort"]]
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env.update({"MCP_TIMEOUT": "1800000", "MCP_TOOL_TIMEOUT": str(6 * 3600 * 1000), "DISABLE_AUTOUPDATER": "1"})
    (local / "command.txt").write_text(shlex.join(cmd))

    budget = 0 if a.dry_run else int(cfg["budget"])
    max_nudges, max_s = int(cfg.get("max_nudges", 30)), float(cfg.get("max_hours", 40)) * 3600
    t0 = time.time()
    summary = {"session_id": sid, "driver_model": cfg["driver_model"], "model_id": model, "config": cfg,
               "start_utc": datetime.now(timezone.utc).isoformat(), "usage_before": a.usage_before,
               "claude_version": subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()}
    counts = {"tool_calls": 0, "by_tool": {}, "turns_ended": 0, "nudges": 0, "compactions": 0, "error_waits": 0}
    max_error_waits, error_wait_min = int(cfg.get("max_error_waits", 24)), float(cfg.get("error_wait_min", 20))
    results, init_ok, final_text = [], None, None

    from exp_r.r4a.judge_sidecar import run as run_sidecar
    stop_sidecar = threading.Event()
    threading.Thread(target=run_sidecar, args=(sid, stop_sidecar), kwargs={"log": lambda m: print(m, flush=True)},
                     daemon=True).start()
    p = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write(user_msg(task))
    p.stdin.flush()
    timer = threading.Timer(max_s, lambda: (print("max_hours reached: stopping", flush=True), p.kill()))
    timer.start()
    with open(local / "transcript.jsonl", "w") as tf:
        for line in p.stdout:
            tf.write(line)
            tf.flush()
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ, sub = msg.get("type"), msg.get("subtype")
            el = f"[{time.time() - t0:7.0f}s]"
            if typ == "system" and sub == "init":
                tools = msg.get("tools", [])
                servers = {s.get("name"): s.get("status") for s in msg.get("mcp_servers", [])}
                bad = [t for t in tools if not t.startswith("mcp__harness__")]
                missing = [t for t in allowed if t not in tools]
                init_ok = not bad and not missing and servers.get("harness") == "connected" and len(servers) == 1
                summary["init"] = {"tools": tools, "mcp_servers": servers, "unexpected_tools": bad,
                                   "missing_tools": missing, "model": msg.get("model"), "ok": init_ok}
                print("init:", json.dumps(summary["init"]), flush=True)
                if not init_ok:
                    p.kill()
                    break
            elif typ == "system" and sub == "compact_boundary":
                counts["compactions"] += 1
                print(el, "context compacted", flush=True)
                try:
                    ssh(f"touch {pod_dir}/compacted.flag")
                except Exception as e:
                    print("flag failed:", e, flush=True)
            elif typ == "assistant":
                for c in msg.get("message", {}).get("content", []):
                    if c.get("type") == "tool_use":
                        counts["tool_calls"] += 1
                        counts["by_tool"][c["name"]] = counts["by_tool"].get(c["name"], 0) + 1
                        print(el, "tool_use", c["name"], flush=True)
                    elif c.get("type") == "text" and c.get("text", "").strip():
                        print(el, "text:", repr(c["text"][:300]), flush=True)
            elif typ == "result":
                counts["turns_ended"] += 1
                results.append({k: msg.get(k) for k in ("subtype", "is_error", "num_turns", "total_cost_usd",
                                                         "usage", "duration_ms")})
                st = remote_state(sid) or {}
                used = st.get("passes", 0)
                print(el, f"turn ended ({msg.get('subtype')}); passes {used}/{budget}", flush=True)
                if msg.get("is_error") and used < budget and counts["error_waits"] < max_error_waits:
                    # most likely a usage limit (5-hour session window): wait, then continue the same session
                    counts["error_waits"] += 1
                    print(el, f"turn ended with an error ({str(msg.get('result'))[:200]!r}); waiting "
                              f"{error_wait_min} min before continuing ({counts['error_waits']}/{max_error_waits})",
                          flush=True)
                    time.sleep(error_wait_min * 60)
                    p.stdin.write(user_msg(
                        f"The session continues (it was paused by an interruption). {used} of {budget} forward "
                        f"passes are used, {budget - used} left; your files in /work are unchanged. Continue with "
                        f"your next step."))
                    p.stdin.flush()
                elif used < budget and counts["nudges"] < max_nudges and not msg.get("is_error"):
                    counts["nudges"] += 1
                    p.stdin.write(user_msg(
                        f"The session continues until all {budget} forward passes are used ({used} used, "
                        f"{budget - used} left). Continue with your next step."))
                    p.stdin.flush()
                else:
                    final_text = msg.get("result")
                    p.stdin.close()
    p.wait()
    timer.cancel()
    stop_sidecar.set()
    stderr = p.stderr.read()
    st = remote_state(sid)
    summary.update({"end_utc": datetime.now(timezone.utc).isoformat(), "wall_clock_s": round(time.time() - t0),
                    "exit_code": p.returncode, "stderr_tail": stderr[-2000:], **counts, "results": results,
                    "total_cost_usd_notional": sum((r.get("total_cost_usd") or 0) for r in results),
                    "final_text": final_text, "server_state": st, "lockdown_ok": init_ok, "usage_after": None})
    if not a.dry_run:
        try:
            summary["finalize"] = json.loads(ssh(
                f"cd /workspace/repo && HF_HUB_OFFLINE=1 /workspace/venv/bin/python -m exp_r.r4a.server finalize {sid}"))
        except Exception as e:
            summary["finalize_error"] = repr(e)
    (local / "driver_summary.json").write_text(json.dumps(summary, indent=1))
    try:
        ssh(f"cat > {pod_dir}/transcript.jsonl", stdin=(local / "transcript.jsonl").read_text())
        ssh(f"cat > {pod_dir}/driver_summary.json", stdin=json.dumps(summary, indent=1))
    except subprocess.CalledProcessError as e:
        print("copy to pod failed:", e.stderr, file=sys.stderr)
    print(json.dumps({k: summary.get(k) for k in ("session_id", "wall_clock_s", "tool_calls", "by_tool", "nudges",
                                                    "compactions", "lockdown_ok", "total_cost_usd_notional")}, indent=1))
    return 0 if init_ok else 2


if __name__ == "__main__":
    sys.exit(main())
