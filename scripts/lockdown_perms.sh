#!/usr/bin/env bash
# Pod-side permissions for driver sessions (idempotent). The driver's kernel runs as the unprivileged
# user `wsdriver`; root (eval, training, the harness server) bypasses these permissions.
#   readable by wsdriver: /workspace/venv (Python libs) and its own session work dir only.
#   blocked: artifacts (test results, checkpoints, BFCL data copies), HF cache, repo, BFCL venv
#            (contains BFCL test data), logs, tokens, the session dirs of the harness.
set -euo pipefail
id wsdriver >/dev/null 2>&1 || useradd -M -d /nonexistent -s /usr/sbin/nologin wsdriver
chmod 711 /workspace
for d in /workspace/artifacts /workspace/hf /workspace/repo /workspace/logs /workspace/bfcl-venv \
         /workspace/permtest /workspace/uv-cache /root; do
  [ -e "$d" ] && chmod 700 "$d"
done
for f in /workspace/hf_token; do [ -e "$f" ] && chmod 600 "$f"; done
chmod -R o+rX /workspace/venv
mkdir -p /srv/wst-sessions && chmod 711 /srv/wst-sessions
ls -ld /workspace /workspace/* /srv/wst-sessions
