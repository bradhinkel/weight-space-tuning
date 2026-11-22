#!/usr/bin/env bash
# R.4b pod setup (idempotent): R.4a's setup, then /workspace/r4a/practice_r4b = R.4a's practice + live_simple_style.jsonl,
# shown as /practice. Usage: bash exp_r/r4b/setup_pod.sh /path/to/live_simple_style.jsonl
set -euo pipefail
cd /workspace/repo
bash exp_r/r4a/setup_pod.sh >/dev/null
R=/workspace/r4a
mkdir -p $R/practice_r4b
cp $R/practice/*.jsonl $R/practice_r4b/
/workspace/venv/bin/python -m exp_r.r4b.filter_guard "$1" $R/practice_r4b/live_simple_style.jsonl
ln -sfn $R/practice_r4b /practice
chmod -R a+rX $R/practice_r4b
wc -l $R/practice_r4b/*.jsonl
ls -ld /practice
