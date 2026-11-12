"""Count weights that differ between two checkpoints, using the harness definition (tools.py _diff):
an element is "touched" if its stored value differs from the reference; the percentage is over all parameters
of the reference checkpoint.

    python scripts/weights_touched.py --ref Qwen/Qwen3.5-2B --ckpt /workspace/artifacts/a/broken --out touched.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def shards(ckpt: str):
    p = Path(ckpt)
    if not p.exists():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(ckpt, allow_patterns=["*.safetensors", "*.json"]))
    return sorted(p.glob("*.safetensors"))


def open_all(files):
    idx = {}
    for f in files:
        h = safe_open(str(f), framework="pt")
        for k in h.keys():
            idx[k] = h
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref, cur = open_all(shards(a.ref)), open_all(shards(a.ckpt))
    missing, extra = sorted(set(ref) - set(cur)), sorted(set(cur) - set(ref))
    total = touched = 0
    rows, by_module = [], defaultdict(lambda: [0, 0])
    for n in sorted(set(ref) & set(cur)):
        r, c = ref[n].get_tensor(n), cur[n].get_tensor(n)
        k = r.numel()
        total += k
        if r.shape != c.shape or r.dtype != c.dtype:
            rows.append({"name": n, "error": f"shape/dtype {tuple(r.shape)}/{r.dtype} vs {tuple(c.shape)}/{c.dtype}"})
            continue
        ne = int((r != c).sum())
        mod = n.rsplit(".", 2)[-2] if n.endswith(".weight") or n.endswith(".bias") else n
        by_module[mod][0] += ne
        by_module[mod][1] += k
        if ne:
            d = c.float() - r.float()
            rows.append({"name": n, "numel": k, "n_changed": ne, "delta_fro": float(d.norm()),
                         "base_fro": float(r.float().norm())})
            touched += ne
    # Denominator = tensors present in both, matching the harness (named_parameters of the loaded HF model,
    # which does not load the MTP head). Tensors only in the reference are listed but not counted.
    missing_numel = sum(ref[n].get_tensor(n).numel() for n in missing)
    res ={"ref": a.ref, "ckpt": a.ckpt, "n_parameters": total, "weights_touched": touched,
           "pct_touched": 100.0 * touched / total, "tensors_changed": sum(1 for r in rows if "n_changed" in r),
           "missing_in_ckpt": missing, "missing_numel": missing_numel, "extra_in_ckpt": extra,
           "by_module": {m: {"n_changed": v[0], "numel": v[1]} for m, v in sorted(by_module.items())},
           "tensors": rows}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_parameters", "weights_touched", "pct_touched", "tensors_changed")}))
    print("missing", len(missing), "extra", len(extra))
    for m, v in res["by_module"].items():
        print(f"{m:20s} {v['n_changed']:>12,} / {v['numel']:>12,}")


if __name__ == "__main__":
    main()
