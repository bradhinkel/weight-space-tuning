#!/usr/bin/env bash
# R.2 pod setup (idempotent). Needs R.1's setup first: /workspace/venv, /workspace/r1/models/base (the full
# Qwen3.5-0.8B checkpoint at revision 2fc0636...), /workspace/r1/bench, /workspace/r1/mathverify_pkgs.
#   /workspace/r2/model     text-only view shown to the driver as /model: their config (Qwen3_5ForCausalLM), the
#                           tokenizer, and model.safetensors with the 320 model.language_model.* tensors
#   /workspace/r2/practice  their practice files (gsm8k_train, math_train, aime), shown as /practice
#   /model, /practice       symlinks; /work is pointed at each session's work dir by the server
set -euo pipefail
cd /workspace/repo
source /workspace/venv/bin/activate
R2=/workspace/r2
mkdir -p $R2/model $R2/practice $R2/sessions

python - <<'EOF'
import json, os, shutil
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = "/workspace/r1/models/base", "/workspace/r2/model"
orig = "/workspace/repo/exp_r/r2/original/text_model_files"
for f in ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"):
    shutil.copyfile(os.path.join(orig, f), os.path.join(dst, f))
shutil.copyfile(os.path.join(src, "tokenizer.json"), os.path.join(dst, "tokenizer.json"))
out = {}
for fn in sorted(os.listdir(src)):
    if fn.endswith(".safetensors"):
        with safe_open(os.path.join(src, fn), "pt") as f:
            for k in f.keys():
                if k.startswith("model.language_model."):
                    out[k] = f.get_tensor(k)
save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
print("text tensors:", len(out))
EOF

if [ ! -d $R2/ext/lm_understand-logs ]; then
  mkdir -p $R2/ext && git clone --depth 1 https://github.com/RikiyaT/lm_understand-logs $R2/ext/lm_understand-logs
fi
cp $R2/ext/lm_understand-logs/exp8_math/practice/{gsm8k_train.jsonl,math_train.jsonl,aime.jsonl,README.txt,stats.json} $R2/practice/
wc -l $R2/practice/*.jsonl

ln -sfn $R2/model /model
ln -sfn $R2/practice /practice
python -c "import scipy, sklearn, pandas, sympy, networkx, tokenizers, safetensors; print('kernel libs ok')" || \
  pip install -q scipy scikit-learn pandas sympy networkx

bash scripts/lockdown_perms.sh >/dev/null
chmod 711 $R2
chmod -R a+rX $R2/model $R2/practice
chmod 700 $R2/sessions $R2/ext /workspace/r1
ls -ld /model /practice $R2 $R2/* /workspace/r1
