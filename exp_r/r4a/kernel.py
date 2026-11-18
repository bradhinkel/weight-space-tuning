"""R.4a driver kernel: R.2's kernel (exp_r/r2/kernel.py: the original Experiment 8 API, sandbox, no gradients)
with two read-only additions for the reference models:

    load_base(name)   the tensor in the base model Qwen/Qwen3.5-2B (before the fine-tune), float32 torch
    lora(name)        (A, B, scale) of the xLAM LoRA for that tensor, float32 torch, so that up to bf16 rounding
                      load_original(name) == load_base(name) + scale * B @ A; None for tensors the LoRA did not touch
    LORA_NAMES        the tensors the LoRA touched;  LORA_RANK, LORA_SCALE

/model/model.safetensors holds the broken model (the "original" here), /model/base/model.safetensors the base model,
/model/adapter/ the unmerged adapter.
"""
from exp_r.r2 import kernel as _k2

EXTRA = r'''
_BASE = "/model/base/model.safetensors"
_ADAPTER = "/model/adapter/adapter_model.safetensors"
_acfg = json.load(open("/model/adapter/adapter_config.json"))
LORA_RANK = int(_acfg["r"])
LORA_SCALE = float(_acfg["lora_alpha"]) / (LORA_RANK ** 0.5 if _acfg.get("use_rslora") else LORA_RANK)
with _safe_open(_ADAPTER, "pt") as _f:
    _AKEYS = set(_f.keys())


def _akey(name, part):
    return "base_model.model." + name[: -len(".weight")] + f".lora_{part}.weight"


LORA_NAMES = sorted(n for n in NAMES if n.endswith(".weight") and _akey(n, "A") in _AKEYS)


def load_base(name):
    """The tensor in the base model Qwen/Qwen3.5-2B, before the xLAM fine-tune (float32 torch)."""
    if name not in _NAMESET:
        raise KeyError(f"unknown tensor {name!r} (see NAMES)")
    with _safe_open(_BASE, "pt") as f:
        return f.get_tensor(name).float()


def lora(name):
    """(A, B, scale) of the xLAM LoRA for this tensor (A: [rank, in], B: [out, rank]; the fine-tune added
    scale * B @ A), or None if the LoRA did not touch it."""
    if name not in _NAMESET:
        raise KeyError(f"unknown tensor {name!r} (see NAMES)")
    if _akey(name, "A") not in _AKEYS:
        return None
    with _safe_open(_ADAPTER, "pt") as f:
        return f.get_tensor(_akey(name, "A")).float(), f.get_tensor(_akey(name, "B")).float(), LORA_SCALE
'''
PRELUDE = _k2.PRELUDE + EXTRA


def main(conn, workdir, user, extra_read, mem_bytes, max_out, common_path):
    _k2.PRELUDE = PRELUDE
    return _k2.main(conn, workdir, user, extra_read, mem_bytes, max_out, common_path)
