"""Experiment V shared definitions: paths, limits, edit validation, and the kernel's confinement.

Everything runs on the laptop. A session's layout (V_ROOT/sessions/<sid>/):
    sandbox/            the kernel's current directory (read-only for the kernel)
      model  -> V_ROOT/model          the broken model's tensors (model.safetensors, float32), read-only
      practice -> PRACTICE            the validation split only (exp_v/data.py), read-only
      work/                           the kernel's only writable folder; work/edited/ holds the edits
    calls.jsonl, state.json, ...      server-side logs (never readable by the kernel)
Test images and labels are read only by `exp_v.server finalize` and exp_v/selftest.py, never by a session.
"""
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
V_ROOT = Path(os.environ.get("V_ROOT", Path.home() / "wst-v"))
MODEL_DIR = V_ROOT / "model"                       # model.safetensors (the broken model), info.json
SESSIONS = V_ROOT / "sessions"
PRACTICE = Path.home() / "datasets" / "waterbirds" / "practice"
BROKEN_PT = REPO / "artifacts" / "v" / "broken" / "model.pt"

LIMITS = {
    "eval_max_items": 1199,          # every practice image in one pass
    "own_images_max": 256,           # the driver's own images (files in work/) per pass
    "act_layers_max": 4,             # activation layers per pass
    "spatial_max_items": 32,         # full feature maps (not pooled) for at most this many images
    "show_per_call": 16, "show_total": 400,
    "python_runs": 300, "exec_timeout_s": 900, "kernel_mem_bytes": 16 << 30, "max_output_chars": 10000,
}
GROUPS = ["landbird/land", "landbird/water", "waterbird/land", "waterbird/water"]
# training-split share of each group: the published mean-accuracy weights (V.0)
TRAIN_COUNTS = np.array([3498, 184, 56, 1057])
TRAIN_WEIGHTS = TRAIN_COUNTS / TRAIN_COUNTS.sum()
ACT_LAYERS = (["conv1", "layer1", "layer2", "layer3", "layer4", "avgpool"]
              + [f"layer1.{i}" for i in range(3)] + [f"layer2.{i}" for i in range(4)]
              + [f"layer3.{i}" for i in range(6)] + [f"layer4.{i}" for i in range(3)])


def edit_problem(name, arr, orig):
    """Why an edit is refused (str), or None. As R.2 (shape, finite, not a whole-tensor constant multiple);
    nothing is frozen: every tensor of the network, including the last layer, may be edited."""
    if tuple(arr.shape) != tuple(orig.shape):
        return f"shape {tuple(arr.shape)} differs from the original {tuple(orig.shape)}"
    if not np.isfinite(arr).all():
        return "the tensor contains NaN or inf"
    o = orig.astype(np.float64).ravel()
    a = arr.astype(np.float64).ravel()
    oo = float(o @ o)
    if oo > 0:
        c = float(a @ o) / oo
        resid = float(np.linalg.norm(a - c * o))
        if abs(c - 1.0) > 1e-6 and resid <= 1e-6 * max(float(np.linalg.norm(a)), 1e-30):
            return f"this edit only multiplies the whole tensor by a constant ({c:.6g}), which is out of scope"
    return None


def score(pred, y, g):
    pred, y, g = map(np.asarray, (pred, y, g))
    out, accs = {}, []
    for k, name in enumerate(GROUPS):
        m = g == k
        if m.any():
            a = float((pred[m] == y[m]).mean())
            out[name] = {"acc": round(100 * a, 2), "n": int(m.sum())}
            accs.append((a, k))
    res = {"groups": out, "n": int(len(y)), "plain_mean": round(100 * float((pred == y).mean()), 2) if len(y) else None}
    if len(accs) == 4:
        acc = np.array([a for a, _ in sorted(accs, key=lambda t: t[1])])
        res.update(worst=round(100 * acc.min(), 2), worst_group=GROUPS[int(acc.argmin())],
                   weighted_mean=round(100 * float(acc @ TRAIN_WEIGHTS), 2))
    return res


# ---------------------------------------------------------------- confinement (laptop: no root)
def confine(sandbox: Path, work: Path, deny_extra=()):
    """Landlock (read: Python, libs, the sandbox's model and practice; read-write: work) and seccomp (no
    IPv4/IPv6 sockets). The R pods also dropped to an unprivileged user; the laptop has no root, so the kernel
    runs as the session user and Landlock alone confines its files (V.2 self-test checks it)."""
    import sys
    from common.harness import sandbox as sb
    deny = sb.DENY_PACKAGES | set(deny_extra)
    sb.DENY_PACKAGES.clear(); sb.DENY_PACKAGES.update(deny)
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in deny)
    if loaded:
        raise OSError(f"denied packages already imported before confinement: {loaded[:10]}")
    venv = sb._prefix_allowlist(sys.prefix) if sys.prefix != sys.base_prefix else [sys.prefix]
    read = ["/usr", "/lib", "/lib64", "/bin", "/etc", "/proc", "/sys/devices/system/cpu",
            "/dev/null", "/dev/urandom", "/dev/random", "/dev/zero", *venv, sys.base_prefix,
            str(sandbox), os.path.realpath(sandbox / "model"), os.path.realpath(sandbox / "practice")]
    if (sandbox / "prior").exists():                    # V.7 only: read-only prior work
        read.append(os.path.realpath(sandbox / "prior"))
    os.chdir(sandbox)
    abi = sb.landlock(read, [str(work)])
    sb.seccomp_no_network()
    os.environ.update({"TMPDIR": str(work), "CUDA_VISIBLE_DEVICES": "", "HOME": "/nonexistent"})
    return {"uid": os.getuid(), "landlock_abi": abi, "seccomp": "deny AF_INET/AF_INET6/io_uring",
            "read_write": [str(work)], "denied_packages_extra": sorted(deny_extra)}
