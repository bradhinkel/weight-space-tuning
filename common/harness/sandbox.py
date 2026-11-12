"""OS-level confinement for the driver's Python kernel (Linux x86_64).

Applied in the kernel child *after* it has received the shared weight buffers and imported its libraries:

  1. drop to an unprivileged user (default `wsdriver`): Unix permissions then block every path chmod'ed
     700 by scripts/lockdown_perms.sh (artifacts, HF cache, repo, BFCL venv, logs, tokens);
  2. Landlock: file access restricted to an allowlist (read-only: Python, venv, system libs; read-write:
     the session working dir). Blocks /dev/nvidia* (no GPU) and anything else not listed;
  3. seccomp: socket(AF_INET/AF_INET6) and io_uring_setup fail with EACCES (no network).

Root inside the RunPod container cannot use namespaces (unshare -> EPERM), which is why these three
unprivileged mechanisms are used instead. Each step raises if it cannot be applied; the kernel refuses
to start unconfined.
"""
import ctypes
import os
import pwd
import struct

_libc = ctypes.CDLL(None, use_errno=True)

# ---- Landlock (ABI >= 1) -------------------------------------------------------------------------
_SYS_LL_CREATE, _SYS_LL_ADD, _SYS_LL_RESTRICT = 444, 445, 446
_EXEC, _WRITE, _READ, _READDIR = 1, 2, 4, 8
_ALL_V1 = (1 << 13) - 1          # execute .. make_sym
_REFER, _TRUNCATE = 1 << 13, 1 << 14
_FILE_ONLY = _EXEC | _WRITE | _READ | _TRUNCATE
PR_SET_NO_NEW_PRIVS = 38


def _err(what):
    e = ctypes.get_errno()
    raise OSError(e, f"{what}: {os.strerror(e)}")


def landlock(read_paths, rw_paths):
    abi = _libc.syscall(_SYS_LL_CREATE, None, 0, 1)
    if abi < 1:
        _err("landlock unavailable")
    handled = _ALL_V1 | (_REFER if abi >= 2 else 0) | (_TRUNCATE if abi >= 3 else 0)
    attr = struct.pack("Q", handled)
    fd = _libc.syscall(_SYS_LL_CREATE, ctypes.c_char_p(attr), len(attr), 0)
    if fd < 0:
        _err("landlock_create_ruleset")

    def add(path, access):
        if not os.path.exists(path):
            return
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            if not os.path.isdir(path):
                access &= _FILE_ONLY
            rule = struct.pack("=Qi", access & handled, pfd)  # struct is packed in the kernel ABI
            if _libc.syscall(_SYS_LL_ADD, fd, 1, ctypes.c_char_p(rule), 0) != 0:
                _err(f"landlock_add_rule {path}")
        finally:
            os.close(pfd)

    for p in read_paths:
        add(p, _EXEC | _READ | _READDIR)
    for p in rw_paths:
        add(p, handled)
    if _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        _err("no_new_privs")
    if _libc.syscall(_SYS_LL_RESTRICT, fd, 0) != 0:
        _err("landlock_restrict_self")
    os.close(fd)
    return abi


# ---- seccomp: deny IPv4/IPv6 sockets and io_uring -----------------------------------------------
def seccomp_no_network():
    LD, JEQ, RET = 0x20, 0x15, 0x06
    ALLOW, KILL, DENY = 0x7FFF0000, 0x80000000, 0x00050000 | 13  # EACCES
    prog = [
        (LD, 0, 0, 4), (JEQ, 1, 0, 0xC000003E), (RET, 0, 0, KILL),   # arch must be x86_64
        (LD, 0, 0, 0), (JEQ, 5, 0, 425),                             # io_uring_setup -> deny
        (JEQ, 0, 3, 41),                                             # socket?
        (LD, 0, 0, 16), (JEQ, 2, 0, 2), (JEQ, 1, 0, 10),             # AF_INET / AF_INET6 -> deny
        (RET, 0, 0, ALLOW), (RET, 0, 0, DENY),
    ]
    buf = b"".join(struct.pack("HBBI", c, jt, jf, k) for c, jt, jf, k in prog)
    filt = ctypes.create_string_buffer(buf)

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = SockFprog(len(prog), ctypes.addressof(filt))
    if _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        _err("no_new_privs")
    if _libc.prctl(22, 2, ctypes.byref(fprog), 0, 0) != 0:   # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
        _err("seccomp")


def drop_user(name):
    pw = pwd.getpwnam(name)
    os.setgroups([])
    os.setgid(pw.pw_gid)
    os.setuid(pw.pw_uid)
    if os.getuid() == 0 or os.geteuid() == 0:
        raise OSError("failed to drop root")
    os.environ["HOME"] = "/nonexistent"


# Packages the kernel may not read (A.5 patch): anything that can build or run a model, so the only way to
# run the target is the metered forward_pass. Landlock is allow-only, so the venv is granted entry by entry
# minus these. Hand-written forward passes in torch are forbidden by the task rules and audited instead
# (scripts/audit_session.py).
DENY_PACKAGES = {
    "transformers", "peft", "accelerate", "vllm", "fla", "flashinfer", "compressed_tensors", "tilelang",
    "tokenspeed_mla", "tokenspeed_triton", "quack", "b12x", "humming", "triton", "xgrammar", "llguidance",
    "outlines_core", "lmformatenforcer", "mistral_common", "huggingface_hub", "datasets", "instanttensor",
    "fastsafetensors", "model_hosting_container_standards", "cutile_cache",
    "torchvision", "torchaudio", "torchcodec", "openai", "anthropic", "diffusers", "sentence_transformers",
    "xformers", "mamba_ssm", "causal_conv1d", "optimum", "auto_gptq", "bitsandbytes", "deepspeed", "mlx",
}


def _denied(entry: str) -> bool:
    base = entry.split("-")[0].split(".")[0] if entry.endswith((".dist-info", ".egg-info")) else entry
    base = base.removesuffix(".py").removesuffix(".libs").removesuffix(".pth")
    return base.lower() in DENY_PACKAGES or base.lower().startswith("vllm_")


def _prefix_allowlist(prefix):
    """Grant read on everything under the venv except denied site-packages entries."""
    import glob
    out = []
    sp_dirs = set(glob.glob(os.path.join(prefix, "lib", "python3*", "site-packages")))
    lib_real = os.path.realpath(os.path.join(prefix, "lib"))
    for top in sorted(os.listdir(prefix)):
        tp = os.path.join(prefix, top)
        real = os.path.realpath(tp)
        if top != "lib" and (real == lib_real or real.startswith(lib_real + os.sep)):
            continue  # e.g. lib64 -> lib: Landlock follows the symlink and would grant all of lib
        if top != "lib":
            out.append(tp)
            continue
        for py in sorted(os.listdir(tp)):
            pp = os.path.join(tp, py)
            if not os.path.isdir(pp):
                out.append(pp)
                continue
            for e in sorted(os.listdir(pp)):
                ep = os.path.join(pp, e)
                if ep not in sp_dirs:
                    out.append(ep)
                    continue
                out += [os.path.join(ep, x) for x in sorted(os.listdir(ep)) if not _denied(x)
                        and not _denied(os.path.basename(os.path.realpath(os.path.join(ep, x))))]
    return out


def confine(workdir, user="wsdriver", extra_read=()):
    """Apply all three layers. Returns a dict describing what was applied (logged by the harness)."""
    import sys
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in DENY_PACKAGES)
    if loaded:
        raise OSError(f"denied packages already imported before confinement: {loaded[:10]}")
    venv = _prefix_allowlist(sys.prefix) if sys.prefix != sys.base_prefix else [sys.prefix]
    extra = [p for p in extra_read if not any(_denied(part) for part in p.split(os.sep))
             and not p.rstrip("/").endswith("site-packages") and not p.startswith(sys.prefix + os.sep + "lib")]
    read = ["/usr", "/lib", "/lib64", "/bin", "/etc", "/proc", "/sys/devices/system/cpu",
            "/dev/null", "/dev/urandom", "/dev/random", "/dev/zero",
            *venv, sys.base_prefix, *extra]
    os.chdir(workdir)
    drop_user(user)
    abi = landlock(read, [workdir])
    seccomp_no_network()
    os.environ.update({"TMPDIR": workdir, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"})
    return {"user": user, "uid": os.getuid(), "landlock_abi": abi, "seccomp": "deny AF_INET/AF_INET6/io_uring",
            "read_only": read, "read_write": [workdir]}
