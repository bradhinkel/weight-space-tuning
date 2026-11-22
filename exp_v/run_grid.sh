#!/usr/bin/env bash
# V.4 grid: two queues in parallel (one per arm), runs sequential within a queue.
#   bash exp_v/run_grid.sh            -> runs exp_v/sessions/grid/queue_{nofit,fit}.txt
# Each run's driver log: artifacts/v/grid/<tag>_driver.log ; a finished run is skipped on restart (done marker).
set -u
cd "$(dirname "$0")/.."
PY=~/vision_env/.venv/bin/python
mkdir -p artifacts/v/grid
run_queue() {
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    tag=$(basename "$cfg" .json)
    [ -f "artifacts/v/grid/$tag.done" ] && continue
    echo "$(date -u +%FT%TZ) start $tag" >> artifacts/v/grid/grid.log
    PYTHONPATH=. $PY -u -m exp_v.driver --config "$cfg" > "artifacts/v/grid/${tag}_driver.log" 2>&1
    rc=$?
    echo "$(date -u +%FT%TZ) end $tag rc=$rc" >> artifacts/v/grid/grid.log
    [ $rc -eq 0 ] && touch "artifacts/v/grid/$tag.done"
  done < "$1"
}
run_queue exp_v/sessions/grid/queue_nofit.txt &
sleep 20   # stagger the two starts (GPU preload, session ids)
run_queue exp_v/sessions/grid/queue_fit.txt &
wait
echo "$(date -u +%FT%TZ) grid finished" >> artifacts/v/grid/grid.log
