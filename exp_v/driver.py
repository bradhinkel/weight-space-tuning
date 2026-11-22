"""Run one Experiment V session on the laptop: R.4a's driver loop (exp_r/r4a/driver.py) with the harness MCP server
(exp_v.server) launched locally over stdio instead of over SSH, and no judge sidecar.

    ~/vision_env/.venv/bin/python -m exp_v.driver --config exp_v/sessions/v3_b25_nofit.json [--usage-before "35%"] [--dry-run]

Lock-down as common/harness/driver.py: --tools "" (no built-in tools), only the harness MCP server
(--strict-mcp-config), --setting-sources local from an empty cwd outside the repo, dontAsk + an allowlist of the
harness tools, the init message checked before the driver acts, ANTHROPIC_API_KEY removed (Max billing only).
Loop as R.4a: a continue message while passes remain; after the budget is used, the next reply without a tool call
is the final answer; on compaction the server shows work/notes.md again; an error turn while passes remain
(most likely a usage limit) is resumed after error_wait_min minutes. At the end `exp_v.server finalize` scores
the submission on the sealed test split.
Config keys: tag, driver_model, effort, budget, fitting ("refused" | "allowed"), task_file, max_nudges, max_hours,
max_turns, max_error_waits, error_wait_min, limits.
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

from exp_v.common import REPO, SESSIONS

DRIVER_MODELS = {"fable-5.2": "claude-fable-5-2", "fable-5.1": "claude-fable-5-1", "opus-5.5": "claude-opus-5-5"}
TOOLS = ["python", "forward", "show_images"]
SERVER_KEYS = {"budget", "fitting", "limits"}
PY = sys.executable


ARM_TEXT = {
    "refused": {
        "<<FIT_RULE>>": (
            "In this session, edits fitted to the practice labels are not allowed. You may compute any statistics of "
            "the practice images' activations, also by group (means, variances, differences, correlations, principal "
            "components, top-activating images), and set weights from those statistics and from the weights "
            "themselves by reasoning you can explain. You may not obtain weight values by fitting them to the labels: "
            "no solving for weights that minimize an error on the practice labels or predictions, in closed form "
            "(least squares, linear or logistic regression, discriminant analysis, ...) or iteratively. "
            "scikit-learn is not available in this session."),
        "<<FIT_RULE_SHORT>>": "Also not allowed in this session: (d) fitting weight values to the practice labels "
                              "(see WHAT COUNTS).",
        "<<SKLEARN>>": "",
        "<<FC_RULE>>": "the last layer included: to see what an edit of fc does, use a forward pass",
    },
    "allowed": {
        "<<FIT_RULE>>": (
            "In this session, fitting is allowed: besides edits you reason out, you may obtain weight values by "
            "fitting them to the practice labels, in closed form or iteratively (for example a linear or logistic "
            "regression from saved activations to labels, with scikit-learn), and you may compute the last layer "
            "(fc) on saved avgpool features to score such fits in the kernel. Fits must use activations saved by "
            "forward passes; the network itself is still not trained or run in the kernel (rules a and b)."),
        "<<FIT_RULE_SHORT>>": "Allowed in this session: fitting weight values to the practice labels from saved "
                              "activations, and computing fc on saved avgpool features (see WHAT COUNTS).",
        "<<SKLEARN>>": ", scikit-learn",
        "<<FC_RULE>>": "apart from the last layer fc on saved avgpool features, which this session's fitting rule "
                       "allows; otherwise, to see what an edit does, use a forward pass",
    },
}


def render_task(template, budget, fitting):
    s = template.replace("<<BUDGET>>", str(budget))
    for k, v in ARM_TEXT[fitting].items():
        s = s.replace(k, v)
    if "<<" in s:
        raise ValueError("unfilled marker in the task text")
    return s


def user_msg(text):
    return json.dumps({"type": "user", "message": {"role": "user", "content": text}}) + "\n"


def state(sdir):
    try:
        return json.loads((sdir / "state.json").read_text())
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
    sid = f"{cfg['tag']}{'_dry' if a.dry_run else ''}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    sdir = SESSIONS / sid
    sdir.mkdir(parents=True)
    os.chmod(sdir, 0o700)
    work = Path("/tmp/wst-driver") / sid          # empty cwd, outside any repo (no CLAUDE.md discovery)
    work.mkdir(parents=True)
    task = render_task((REPO / cfg["task_file"]).read_text(), cfg["budget"], cfg.get("fitting", "refused"))
    if a.dry_run:
        task = ("Call the python tool once with the code `len(NAMES)`, then call show_images with ids [\"p0000\"], "
                "report both results in one line each (what the image shows, in a few words), then stop. Do nothing else.")
    (sdir / "TASK.md").write_text(task)
    scfg = {"session_id": sid, **{k: v for k, v in cfg.items() if k in SERVER_KEYS}}
    if a.dry_run:
        scfg["budget"] = 0
    (sdir / "config.json").write_text(json.dumps(scfg, indent=1))
    mcp = {"mcpServers": {"harness": {"type": "stdio", "command": PY,
                                      "args": ["-m", "exp_v.server", "--config", str(sdir / "config.json")],
                                      "env": {"PYTHONPATH": str(REPO), "HOME": str(Path.home()),
                                              "PATH": "/usr/bin:/bin"}}}}
    (sdir / "mcp.json").write_text(json.dumps(mcp, indent=1))

    allowed = [f"mcp__harness__{t}" for t in TOOLS]
    cmd = ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", model, "--tools", "", "--strict-mcp-config", "--mcp-config", str(sdir / "mcp.json"),
           "--setting-sources", "local", "--permission-mode", "dontAsk", "--allowedTools", ",".join(allowed),
           "--no-session-persistence", "--max-turns", str(cfg.get("max_turns", 3000))]
    if cfg.get("effort"):
        cmd += ["--effort", cfg["effort"]]
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env.update({"MCP_TIMEOUT": "600000", "MCP_TOOL_TIMEOUT": str(3 * 3600 * 1000), "DISABLE_AUTOUPDATER": "1"})
    (sdir / "command.txt").write_text(shlex.join(cmd))

    budget = 0 if a.dry_run else int(cfg["budget"])
    max_nudges, max_s = int(cfg.get("max_nudges", 40)), float(cfg.get("max_hours", 30)) * 3600
    t0 = time.time()
    summary = {"session_id": sid, "driver_model": cfg["driver_model"], "model_id": model, "config": cfg,
               "start_utc": datetime.now(timezone.utc).isoformat(), "usage_before": a.usage_before,
               "claude_version": subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()}
    counts = {"tool_calls": 0, "by_tool": {}, "turns_ended": 0, "nudges": 0, "compactions": 0, "error_waits": 0}
    max_error_waits, error_wait_min = int(cfg.get("max_error_waits", 72)), float(cfg.get("error_wait_min", 20))
    results, init_ok, final_text = [], None, None

    p = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write(user_msg(task))
    p.stdin.flush()
    timer = threading.Timer(max_s, lambda: (print("max_hours reached: stopping", flush=True), p.kill()))
    timer.start()
    with open(sdir / "transcript.jsonl", "w") as tf:
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
                (sdir / "compacted.flag").touch()
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
                used = (state(sdir) or {}).get("passes", 0)
                print(el, f"turn ended ({msg.get('subtype')}); passes {used}/{budget}", flush=True)
                if msg.get("is_error") and used < budget and counts["error_waits"] < max_error_waits:
                    counts["error_waits"] += 1
                    print(el, f"turn ended with an error ({str(msg.get('result'))[:200]!r}); waiting "
                              f"{error_wait_min} min ({counts['error_waits']}/{max_error_waits})", flush=True)
                    time.sleep(error_wait_min * 60)
                    p.stdin.write(user_msg(
                        f"The session continues (it was paused by an interruption). {used} of {budget} forward "
                        f"passes are used, {budget - used} left; your files in work/ are unchanged. Continue with "
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
    stderr = p.stderr.read()
    summary.update({"end_utc": datetime.now(timezone.utc).isoformat(), "wall_clock_s": round(time.time() - t0),
                    "exit_code": p.returncode, "stderr_tail": stderr[-2000:], **counts, "results": results,
                    "total_cost_usd_notional": sum((r.get("total_cost_usd") or 0) for r in results),
                    "final_text": final_text, "server_state": state(sdir), "lockdown_ok": init_ok,
                    "usage_after": None})
    if not a.dry_run:
        fin = subprocess.run([PY, "-m", "exp_v.server", "finalize", sid], capture_output=True, text=True, cwd=REPO)
        summary["finalize"] = fin.stdout.strip().splitlines()[-1] if fin.stdout.strip() else fin.stderr[-1000:]
    (sdir / "driver_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary.get(k) for k in ("session_id", "wall_clock_s", "tool_calls", "by_tool", "nudges",
                                                    "compactions", "lockdown_ok", "total_cost_usd_notional",
                                                    "final_text")}, indent=1))
    return 0 if init_ok else 2


if __name__ == "__main__":
    sys.exit(main())
