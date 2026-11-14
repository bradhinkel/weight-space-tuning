#!/bin/bash
# Run a list of R.1 eval jobs sequentially on one pod: each arg is MODEL:SET:SHARD (shard "i/k" or "all").
# Output: /workspace/r1/eval/<MODEL>/<SET>_s<i>of<k>/  (merge per model with r1_eval merge)
cd /workspace/repo
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 MATHVERIFY_PKGS=/workspace/r1/mathverify_pkgs
for job in "$@"; do
  IFS=: read m s sh <<< "$job"
  if [ "$sh" = all ]; then tag=${s}_all; sa=""; else tag=${s}_s${sh%/*}of${sh#*/}; sa="--shard $sh"; fi
  out=/workspace/r1/eval/$m/$tag
  echo "=== $job start $(date -u +%FT%TZ)"
  python -m exp_r.r1_eval run /workspace/r1/models/$m $out --bench /workspace/r1/bench --sets $s $sa --mem ${MEM:-0.85} \
      > /workspace/r1/logs/${m}_${tag}.log 2>&1
  echo "=== $job exit $? $(date -u +%FT%TZ)"
  grep -E "^$s " /workspace/r1/logs/${m}_${tag}.log | tail -1
done
echo "=== JOBS DONE $(date -u +%FT%TZ)"
