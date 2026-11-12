"""Launch one locked-down headless Claude Code driver session against the harness.

    python -m common.harness.driver --config exp_a/sessions/a5_b10_fable.json [--usage-before "23%"]

Runs on the workstation, where Claude Code is logged in with the Max subscription. The harness MCP
server runs on the GPU pod and is reached over SSH stdio, so no credentials ever go to the pod.

Lock-down (PROJECT_PLAN 0.3):
  * --tools ""            : no built-in tools at all (no Bash, Read, Write, Edit, Web*, Glob, Grep, Task...)
  * --strict-mcp-config   : only the harness MCP server (no user / claude.ai connectors)
  * --setting-sources local from an empty cwd under /tmp : no user settings, plugins, hooks, CLAUDE.md
  * --permission-mode dontAsk + --allowedTools mcp__harness__* : anything else is denied
  * the stream's init message is checked: if any tool other than mcp__harness__* is present, or the
    harness server is not connected, the session is killed before the driver acts.
  * ANTHROPIC_API_KEY is removed from the environment (Max billing only).

Config keys: tag, driver_model (fable-5.1 | opus-5.5), checkpoint, budget, task_file, max_turns, and any
Harness keys (max_prompts_per_pass, max_new_tokens, practice_files, test_guard, ...).
Outputs (local, git-ignored): artifacts/sessions/<sid>/{TASK.md, mcp.json, transcript.jsonl,
driver_summary.json}; the transcript is also copied to the pod session dir next to the harness logs.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRIVER_MODELS = {"fable-5.1": "claude-fable-5-1", "opus-5.5": "claude-opus-5-5"}
TOOLS = ["python", "forward_pass", "save_weights", "budget", "submit"]
# GPU pod reached over SSH; set these for your own pod.
POD = {"host": os.environ.get("WST_POD_HOST", "127.0.0.1"), "port": os.environ.get("WST_POD_PORT", "22"),
       "key": os.environ.get("WST_POD_KEY", str(Path.home() / ".ssh/id_ed25519"))}
HARNESS_KEYS = {"checkpoint", "budget", "max_prompts_per_pass", "max_new_tokens", "max_prompt_tokens",
                "attention_max_tokens", "attention_max_prompts", "practice_files", "test_guard",
                "exec_timeout_s", "gen_batch_tokens"}


def ssh_base():
    return ["ssh", "-i", POD["key"], "-p", POD["port"], "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=20", "-o", "ConnectTimeout=30", f"root@{POD['host']}"]


def ssh(cmd: str, stdin: str | None = None):
    return subprocess.run(ssh_base() + [cmd], input=stdin, text=True, capture_output=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--usage-before", default=None, help="Max weekly usage reading before (e.g. '23%')")
    ap.add_argument("--dry-run", action="store_true", help="check the lock-down only: start, verify, stop")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    model = DRIVER_MODELS[cfg["driver_model"]]
    sid = f"{cfg['tag']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    local = REPO / "artifacts/sessions" / sid
    local.mkdir(parents=True)
    work = Path("/tmp/wst-driver") / sid          # empty cwd, outside any repo (no CLAUDE.md discovery)
    work.mkdir(parents=True)
    task = (REPO / cfg["task_file"]).read_text()
    if a.dry_run:
        task = "Call the budget tool once, report the result in one line, then stop. Do nothing else."
    (work / "TASK.md").write_text(task)
    (local / "TASK.md").write_text(task)

    # Server-side session config (written by root into the 700 artifacts dir).
    pod_dir = f"/workspace/artifacts/sessions/{sid}"
    scfg = {"session_id": sid, **{k: v for k, v in cfg.items() if k in HARNESS_KEYS}}
    ssh(f"mkdir -p {pod_dir} && cat > {pod_dir}/config.json", stdin=json.dumps(scfg, indent=1))
    remote = (f"cd /workspace/repo && HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 "
              f"/workspace/venv/bin/python -m common.harness.mcp_server --config {pod_dir}/config.json "
              f"2>>{pod_dir}/server_stderr.log")
    mcp = {"mcpServers": {"harness": {"type": "stdio", "command": "ssh", "args": ssh_base()[1:] + [remote]}}}
    (local / "mcp.json").write_text(json.dumps(mcp, indent=1))

    allowed = [f"mcp__harness__{t}" for t in TOOLS]
    cmd = ["claude", "-p", task, "--model", model, "--tools", "", "--strict-mcp-config",
           "--mcp-config", str(local / "mcp.json"), "--setting-sources", "local",
           "--permission-mode", "dontAsk", "--allowedTools", ",".join(allowed),
           "--output-format", "stream-json", "--verbose", "--no-session-persistence",
           "--max-turns", str(cfg.get("max_turns", 200))]
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env.update({"MCP_TIMEOUT": "900000", "MCP_TOOL_TIMEOUT": "1800000", "DISABLE_AUTOUPDATER": "1"})
    (local / "command.txt").write_text(shlex.join(cmd))

    t0 = time.time()
    started = datetime.now(timezone.utc).isoformat()
    summary = {"session_id": sid, "driver_model": cfg["driver_model"], "model_id": model, "config": cfg,
               "start_utc": started, "usage_before": a.usage_before, "claude_version":
               subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()}
    n_tool_calls, by_tool, init_ok, result = 0, {}, None, None
    with open(local / "transcript.jsonl", "w") as tf:
        p = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in p.stdout:
            tf.write(line); tf.flush()
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "system" and msg.get("subtype") == "init":
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
            elif msg.get("type") == "assistant":
                for c in msg.get("message", {}).get("content", []):
                    if c.get("type") == "tool_use":
                        n_tool_calls += 1
                        by_tool[c["name"]] = by_tool.get(c["name"], 0) + 1
                        print(f"[{time.time() - t0:7.0f}s] tool_use {c['name']}", flush=True)
                    elif c.get("type") == "text" and c.get("text", "").strip():
                        print(f"[{time.time() - t0:7.0f}s] text: {c['text'][:300]!r}", flush=True)
            elif msg.get("type") == "result":
                result = msg
        p.wait()
        stderr = p.stderr.read()
    summary.update({
        "end_utc": datetime.now(timezone.utc).isoformat(), "wall_clock_s": round(time.time() - t0),
        "exit_code": p.returncode, "stderr_tail": stderr[-2000:], "tool_calls": n_tool_calls, "by_tool": by_tool,
        "num_turns": (result or {}).get("num_turns"), "result_subtype": (result or {}).get("subtype"),
        "is_error": (result or {}).get("is_error"), "usage": (result or {}).get("usage"),
        "total_cost_usd_notional": (result or {}).get("total_cost_usd"),
        "final_text": (result or {}).get("result"),
        "usage_after": None, "lockdown_ok": init_ok,
        "note": "usage_after: fill from /usage or claude.ai Settings > Usage (not scriptable)",
    })
    (local / "driver_summary.json").write_text(json.dumps(summary, indent=1))
    try:
        ssh(f"cat > {pod_dir}/transcript.jsonl", stdin=(local / "transcript.jsonl").read_text())
        ssh(f"cat > {pod_dir}/driver_summary.json", stdin=json.dumps(summary, indent=1))
    except subprocess.CalledProcessError as e:
        print("copy to pod failed:", e.stderr, file=sys.stderr)
    print(json.dumps({k: summary[k] for k in ("session_id", "wall_clock_s", "num_turns", "tool_calls",
                                                "by_tool", "lockdown_ok", "result_subtype")}, indent=1))
    return 0 if init_ok else 2


if __name__ == "__main__":
    sys.exit(main())
