#!/bin/bash
# R.1 setup on the pod (as root, from /workspace/repo): grader deps, base model at the original's revision,
# the three published edited checkpoints rebuilt bit-for-bit, and the benchmark files.
set -euo pipefail
cd /workspace/repo
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf
R=/workspace/r1
mkdir -p $R/models $R/logs
# math-verify exactly as in the original (pip --target dir, appended to sys.path by mathqa.py)
[ -d $R/mathverify_pkgs/math_verify ] || pip install -q --no-deps --target $R/mathverify_pkgs \
    math-verify==0.9.0 latex2sympy2_extended==1.11.0 antlr4-python3-runtime==4.13.2
# base model at the original's revision
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17",
                      local_dir="/workspace/r1/models/base")
print("base at", p)
PY
ls -la $R/models/base
# edited checkpoints: copy of the base dir with model.safetensors rebuilt by their apply_edit.py (hash-checked)
for r in p10c p25c p50c; do
  d=$R/models/$r
  W=$(cd $R/models/base && ls *.safetensors | head -1)     # the release names it model.safetensors-00001-of-00001.safetensors
  [ -f $d/$W ] && continue
  mkdir -p $d
  for f in $R/models/base/*; do [ "$(basename $f)" = "$W" ] || cp -r $f $d/; done
  python exp_r/vendor/apply_edit.py $R/models/base/$W exp_r/edits/$r $d/$W
done
python -m exp_r.r1_eval prepare --bench $R/bench
echo SETUP_DONE
