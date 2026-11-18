#!/usr/bin/env bash
# R.4a pod setup (idempotent), on Experiment A's volume (61gtyoubem): needs /workspace/venv (vLLM 0.33), the broken
# checkpoint /workspace/artifacts/a/broken (+ adapter/), the base model in /workspace/hf, practice_v2, and the BFCL
# data in /workspace/bfcl-venv (bfcl_eval's data directory).
#   /workspace/r4a/model     shown as /model: model.safetensors (the broken model's 320 text tensors), base/
#                            model.safetensors (the base model's), adapter/ (the unmerged xLAM LoRA), config, tokenizer
#   /workspace/r4a/practice  A.11c's practice_v2, shown as /practice
#   /workspace/r4a/guard     root-only: A's test_guard.json + the BFCL test categories' user messages
set -euo pipefail
cd /workspace/repo
source /workspace/venv/bin/activate
R=/workspace/r4a
mkdir -p $R/model/base $R/model/adapter $R/practice $R/guard $R/sessions

python - <<'EOF'
import json, os, shutil
from safetensors import safe_open
from safetensors.torch import save_file
from exp_r.r4a.common import ORIG_CKPT, BASE_CKPT
dst = "/workspace/r4a/model"
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    if os.path.exists(os.path.join(ORIG_CKPT, f)):
        shutil.copyfile(os.path.join(ORIG_CKPT, f), os.path.join(dst, f))
def text_view(src_dir, out):
    t = {}
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(src_dir, fn), "pt") as f:
                for k in f.keys():
                    if k.startswith("model.language_model."):
                        t[k] = f.get_tensor(k)
    save_file(t, out, metadata={"format": "pt"})
    return len(t)
print("broken text tensors:", text_view(ORIG_CKPT, f"{dst}/model.safetensors"))
print("base text tensors:", text_view(BASE_CKPT, f"{dst}/base/model.safetensors"))
for f in ("adapter_model.safetensors", "adapter_config.json"):
    shutil.copyfile(os.path.join(ORIG_CKPT, "adapter", f), os.path.join(dst, "adapter", f))
EOF

cp /workspace/artifacts/a/practice_v2/{live_irrelevance,live_relevant,xlam_mismatched,xlam_practice}.jsonl $R/practice/
cp /workspace/artifacts/a/practice_v2/manifest.json $R/practice/
cp /workspace/artifacts/a/test_guard.json $R/guard/
wc -l $R/practice/*.jsonl

# test user messages of the BFCL test categories (for the n-gram guard on the driver's own prompts and texts)
BDATA=$(/workspace/bfcl-venv/bin/python -c "import bfcl_eval, os; print(os.path.join(os.path.dirname(bfcl_eval.__file__), 'data'))")
python - "$BDATA" <<'EOF'
import json, os, sys
d = sys.argv[1]
cats = json.load(open("/workspace/r4a/guard/test_guard.json"))["categories"]
n = 0
with open("/workspace/r4a/guard/test_user_messages.jsonl", "w") as out:
    for c in cats:
        p = os.path.join(d, f"BFCL_v4_{c}.json")
        for line in open(p):
            r = json.loads(line)
            for turn in r["question"]:
                for m in turn:
                    if m.get("role") == "user":
                        out.write(json.dumps({"category": c, "id": r["id"], "text": m["content"]}) + "\n")
                        n += 1
print("test user messages:", n)
EOF

ln -sfn $R/model /model
ln -sfn $R/practice /practice
bash scripts/lockdown_perms.sh >/dev/null
chmod 711 $R
chmod -R a+rX $R/model $R/practice
chmod 700 $R/sessions $R/guard
ls -ld /model /practice $R $R/*
