"""V.2 self-test of the Experiment V harness (exp_v/V_design.md, V.2 and amendment 4). Writes reports/V.2_selftest.json.

    ~/vision_env/.venv/bin/python -m exp_v.selftest

Checks:
  1. scores: the server's practice predictions, logits and avgpool features match exp_v.wb image by image, for the
     original and for an edit; the finalize test score matches an independent computation (features @ fc).
  2. split guard: from inside the kernel, test data, the raw dataset, the broken checkpoint, the repo, ~/.claude and
     the server's logs are unreadable; writes outside work/ fail; no network, no GPU, no torchvision; in the
     no-fitting arm no scikit-learn. Plus exp_v.data's hash check.
  3. practice predicts test: DFR fitted (in the kernel) on one half of practice and scored on the other half,
     against its test score: within 5 points of worst-group.
  4. show_images: returns the images, per-call and session caps, crops, work/ files, refuses other paths.
  5. budget: a failed request costs nothing; the budget is enforced; save_edit refuses a constant multiple.
  6. MCP: the server over stdio lists exactly python, forward, show_images and returns image content.
"""
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from exp_v.common import REPO, SESSIONS, score
# torch / torchvision / exp_v.wb are imported inside main(): the kernel is started with multiprocessing "spawn",
# which re-imports this module in the child, and the kernel refuses to start if torchvision is already loaded.

RESULTS = {}


def check(name, ok, **info):
    RESULTS[name] = {"ok": bool(ok), **info}
    print(("PASS " if ok else "FAIL ") + name, json.dumps(info, default=str)[:400], flush=True)


def new_session(tag, **cfg):
    from exp_v.server import VHarness
    sid = f"{tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    (SESSIONS / sid).mkdir(parents=True)
    cfg = {"session_id": sid, **cfg}
    (SESSIONS / sid / "config.json").write_text(json.dumps(cfg, indent=1))
    return VHarness(cfg)


def fwd(h, rq):
    return json.loads(h.forward(rq))


def kexec(h, code):
    return h.python(code)


GUARD_CODE = r'''
import os, socket, json
res = {}
home = "%(home)s"
for name, p in {"test_manifest": home + "/datasets/waterbirds/manifests/test.csv",
                "raw_metadata": home + "/datasets/waterbirds/extracted/metadata.csv",
                "raw_dir": home + "/datasets/waterbirds/extracted",
                "test_image": home + "/datasets/waterbirds/extracted/%(test_img)s",
                "broken_ckpt": "%(ckpt)s", "repo": "%(repo)s", "claude_dir": home + "/.claude",
                "server_log": "%(sdir)s/calls.jsonl", "session_dir": "%(sdir)s"}.items():
    try:
        if os.path.isdir(p):
            os.listdir(p)
        else:
            open(p, "rb").read(16)
        res[name] = "READABLE"
    except Exception as e:
        res[name] = "blocked:" + type(e).__name__
for name, p in {"write_sandbox": "x.txt", "write_tmp": "/tmp/v2_guard.txt", "write_model": "model/x"}.items():
    try:
        open(p, "w").write("x"); res[name] = "WRITABLE"
    except Exception as e:
        res[name] = "blocked:" + type(e).__name__
try:
    open("work/ok.txt", "w").write("ok"); res["write_work"] = "ok"
except Exception as e:
    res["write_work"] = "FAILED:" + repr(e)
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM); res["socket"] = "OPEN"
except Exception as e:
    res["socket"] = "blocked:" + type(e).__name__
res["cuda"] = torch.cuda.is_available()
for mod in ("torchvision", "sklearn"):
    try:
        __import__(mod); res[mod] = "IMPORTABLE"
    except Exception as e:
        res[mod] = "blocked:" + type(e).__name__
res["cwd_listing"] = sorted(os.listdir("."))
res["practice_listing"] = sorted(os.listdir("practice"))
res["n_practice_images"] = len(os.listdir("practice/images"))
print(json.dumps(res))
'''

DFR_HALF_CODE = r'''
# DFR (V.1 recipe) fitted on practice half A only; scored by the harness on half B
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
lab = pd.read_csv("practice/labels.csv")
rows = [json.loads(l) for l in open("work/fwd_%(k)d/rows.jsonl")]
F = np.load("work/fwd_%(k)d/original_avgpool.npy")
ids = [r["id"] for r in rows]; assert ids == list(lab.id)
rng = np.random.default_rng(0)
perm = rng.permutation(len(lab)); A, B = np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])
y, g = lab.y.to_numpy(), lab.g.to_numpy()
sc = StandardScaler().fit(F[A])
ws, bs = [], []
for _ in range(10):
    n = np.bincount(g[A], minlength=4).min()
    s = np.concatenate([rng.choice(A[g[A] == j], n, replace=False) for j in range(4)])
    m = LogisticRegression(penalty="l1", C=0.3, solver="liblinear").fit(sc.transform(F[s]), y[s])
    ws.append(m.coef_[0]); bs.append(m.intercept_[0])
w, b = np.mean(ws, 0), np.mean(bs)
W = w / sc.scale_; B0 = b - (sc.mean_ / sc.scale_) @ w
fcw = np.stack([-W / 2, W / 2]).astype(np.float32); fcb = np.array([-B0 / 2, B0 / 2], np.float32)
print(save_edit("fc.weight", fcw)); print(save_edit("fc.bias", fcb))
json.dump({"B_ids": [ids[i] for i in B], "A_ids": [ids[i] for i in A]}, open("work/halves.json", "w"))
print(len(A), len(B), np.bincount(g[A]), np.bincount(g[B]))
'''


def main():
    import torch
    from exp_v import wb
    from exp_v.server import finalize
    t0 = time.time()
    broken = wb.resnet50(pretrained=False)
    broken.load_state_dict(torch.load(REPO / "artifacts/v/broken/model.pt", map_location="cpu"))
    fv, lv, yv, gv = wb.features_and_logits(broken, "val")
    val_files = list(wb.manifest("val").img_filename)

    # ---------------------------------------------------------------- fitting-allowed session
    h = new_session("v2_selftest_fit", budget=6, fitting="allowed", limits={"show_total": 40})
    sdir = h.sdir

    # 1a. original on all practice, with avgpool features
    r = fwd(h, {"type": "evaluate", "model": "original", "items": "all", "activations": {"layers": ["avgpool", "layer4"]}})
    k1 = r["pass"]
    rows = [json.loads(l) for l in open(h.work / f"fwd_{k1}" / "rows.jsonl")]
    lg = np.load(h.work / f"fwd_{k1}" / "original_logits.npy")
    F = np.load(h.work / f"fwd_{k1}" / "original_avgpool.npy")
    L4 = np.load(h.work / f"fwd_{k1}" / "original_layer4.npy")
    same_pred = int((lg.argmax(1) == lv.argmax(1)).sum())
    check("1a_original_matches_wb", same_pred == len(lv) and np.abs(lg - lv).max() < 1e-3 and np.abs(F - fv).max() < 1e-3
          and np.abs(L4 - fv).max() < 1e-3 and r["original"] == score(lv.argmax(1), yv, gv)
          and [x["y"] for x in rows] == list(yv),
          same_predictions=f"{same_pred}/{len(lv)}", max_logit_diff=float(np.abs(lg - lv).max()),
          max_feat_diff=float(np.abs(F - fv).max()), server_score=r["original"], wb_score=score(lv.argmax(1), yv, gv))

    # 5a. save_edit refuses a constant multiple; failed request is free
    out = kexec(h, 'try:\n    save_edit("fc.weight", load_original("fc.weight") * 2)\nexcept ValueError as e:\n    print("refused:", e)')
    used = h.passes
    bad = fwd(h, {"type": "evaluate", "model": "edited", "activations": {"layers": ["layer9"]}})
    check("5a_refusals_free", "refused" in out and not bad["ok"] and h.passes == used, kernel=out[-200:],
          failed_request=bad.get("error"))

    # 3. practice predicts test: DFR on half A (in the kernel), score half B (harness) vs test (finalize)
    out = kexec(h, DFR_HALF_CODE % {"k": k1})
    halves = json.loads((h.work / "halves.json").read_text())
    rB = fwd(h, {"type": "both", "model": "both"} | {"type": "evaluate", "items": halves["B_ids"]})
    # 1b. the edit's practice predictions match an independent computation (features @ fc)
    fc = {n: np.load(h.work / "edited" / f"{n}.npy") for n in ("fc.weight", "fc.bias")}
    idxB = [int(i[1:]) for i in halves["B_ids"]]
    indep = fv[idxB] @ fc["fc.weight"].T + fc["fc.bias"]
    lgB = np.load(h.work / f"fwd_{rB['pass']}" / "edited_logits.npy")
    # tolerance 1e-2: the edited fc is large (DFR logits), and GPU vs numpy summation order differs at ~1e-3
    check("1b_edited_matches_independent", (lgB.argmax(1) == indep.argmax(1)).all() and np.abs(lgB - indep).max() < 1e-2,
          same_predictions=f"{int((lgB.argmax(1) == indep.argmax(1)).sum())}/{len(indep)}",
          max_logit_diff=float(np.abs(lgB - indep).max()), max_abs_logit=float(np.abs(indep).max()), kernel=out[-300:])

    # 4. show_images
    res = h.show_images([f"p{i:04d}" for i in range(16)])
    n_img = sum(1 for x in res if not isinstance(x, str))
    res17 = h.show_images([f"p{i:04d}" for i in range(17)])
    from PIL import Image
    Image.new("RGB", (300, 200), (0, 120, 200)).save(h.work / "blue.png")
    resw = h.show_images(["work/blue.png", "p0001"], crops=[None, [0, 0, 112, 112]])
    resx = h.show_images(["model/model.safetensors"])
    resc = h.show_images([f"p{i:04d}" for i in range(16)])   # 16 + 2 + 16 = 34 shown so far; next 16 exceeds 40
    resd = h.show_images([f"p{i:04d}" for i in range(16)])
    check("4_show_images", n_img == 16 and len(res17) == 1 and "at most 16" in res17[0]
          and sum(1 for x in resw if not isinstance(x, str)) == 2 and len(resx) == 1 and "failed" in resx[0]
          and sum(1 for x in resc if not isinstance(x, str)) == 16 and len(resd) == 1 and "cap" in resd[0]
          and h.shown == 34, images_returned=n_img, shown=h.shown, header=res[0][:160], cap_msg=resd[0][-120:])

    # own image in a forward pass
    ro = fwd(h, {"type": "evaluate", "model": "original", "items": [], "images": ["work/blue.png"]})
    check("1c_own_images", ro["ok"] and ro["n_own_images"] == 1, result={k: ro.get(k) for k in ("ok", "original_own_images", "error")})

    # 5b. budget enforcement (budget 6: passes used so far = 3)
    while h.passes < h.budget:
        fwd(h, {"type": "evaluate", "model": "original", "items": ["p0000"]})
    over = fwd(h, {"type": "evaluate", "model": "original", "items": ["p0000"]})
    check("5b_budget_enforced", not over["ok"] and "used up" in over["error"] and h.passes == 6, error=over.get("error"))

    # 2. guard, fitting-allowed kernel
    test_img = wb.manifest("test").img_filename.iloc[0]
    gcode = GUARD_CODE % {"home": str(Path.home()), "test_img": test_img, "ckpt": REPO / "artifacts/v/broken/model.pt",
                          "repo": REPO, "sdir": sdir}
    out = kexec(h, gcode)
    g_fit = json.loads(out.split("```\n", 1)[1].rsplit("\n```", 1)[0].strip().splitlines()[-1])
    h.kproc.kill()

    # finalize the half-A DFR edit and compare with practice half B
    finalize(sdir.name)
    final = json.loads((sdir / "final.json").read_text())
    ft = fv  # noqa: F841
    _, lt, yt, gt = wb.features_and_logits(broken, "test")
    ftest, _, _, _ = wb.features_and_logits(broken, "test")
    indep_t = ftest @ fc["fc.weight"].T + fc["fc.bias"]
    indep_score = score(indep_t.argmax(1), yt, gt)
    check("1d_finalize_matches_independent", final["test"] == indep_score, finalize=final["test"], independent=indep_score)
    pB, tst = rB["edited"]["worst"], final["test"]["worst"]
    check("3_practice_predicts_test", abs(pB - tst) <= 5, practice_halfB_worst=pB, test_worst=tst,
          practice_halfB_weighted_mean=rB["edited"]["weighted_mean"], test_weighted_mean=final["test"]["weighted_mean"],
          original_halfB_worst=rB["original"]["worst"], paired=rB.get("paired"))

    # ---------------------------------------------------------------- no-fitting session: guard
    h2 = new_session("v2_selftest_nofit", budget=1, fitting="refused")
    out = kexec(h2, GUARD_CODE % {"home": str(Path.home()), "test_img": test_img,
                                  "ckpt": REPO / "artifacts/v/broken/model.pt", "repo": REPO, "sdir": h2.sdir})
    g_nofit = json.loads(out.split("```\n", 1)[1].rsplit("\n```", 1)[0].strip().splitlines()[-1])
    h2.kproc.kill()

    def guard_ok(gd, sklearn_blocked):
        reads = ["test_manifest", "raw_metadata", "raw_dir", "test_image", "broken_ckpt", "repo", "claude_dir",
                 "server_log", "session_dir"]
        return (all(gd[k].startswith("blocked") for k in reads + ["write_sandbox", "write_tmp", "write_model", "socket",
                                                                  "torchvision"])
                and gd["write_work"] == "ok" and gd["cuda"] is False and gd["cwd_listing"] == ["model", "practice", "work"]
                and gd["practice_listing"] == ["images", "labels.csv"] and gd["n_practice_images"] == 1199
                and (gd["sklearn"].startswith("blocked") if sklearn_blocked else gd["sklearn"] == "IMPORTABLE"))
    check("2a_guard_fitting_allowed", guard_ok(g_fit, False), **g_fit)
    check("2b_guard_no_fitting", guard_ok(g_nofit, True), **g_nofit)
    import subprocess
    dc = subprocess.run([sys.executable, "-m", "exp_v.data", "check"], capture_output=True, text=True, cwd=REPO)
    check("2c_split_hashes", dc.returncode == 0, tail=dc.stdout[-200:])

    # ---------------------------------------------------------------- 6. MCP round-trip
    ok6, info6 = asyncio.run(mcp_roundtrip())
    check("6_mcp_stdio", ok6, **info6)

    RESULTS["_meta"] = {"seconds": round(time.time() - t0), "sessions": [sdir.name, h2.sdir.name],
                        "date": datetime.now(timezone.utc).isoformat()}
    (REPO / "reports" / "V.2_selftest.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    bad = [k for k, v in RESULTS.items() if not k.startswith("_") and not v["ok"]]
    print("ALL PASS" if not bad else f"FAILED: {bad}")
    return 0 if not bad else 1


async def mcp_roundtrip():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    sid = f"v2_selftest_mcp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    (SESSIONS / sid).mkdir(parents=True)
    (SESSIONS / sid / "config.json").write_text(json.dumps({"session_id": sid, "budget": 1, "fitting": "refused"}))
    params = StdioServerParameters(command=sys.executable, args=["-m", "exp_v.server", "--config",
                                                                 str(SESSIONS / sid / "config.json")],
                                   cwd=str(REPO), env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin",
                                                       "HOME": str(Path.home())})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = sorted(t.name for t in (await s.list_tools()).tools)
            res = await s.call_tool("show_images", {"ids": ["p0000", "p0001"]})
            kinds = [c.type for c in res.content]
            py = await s.call_tool("python", {"code": "len(NAMES), edited()"})
            fw = await s.call_tool("forward", {"request": {"type": "evaluate", "model": "original"}})
            fwj = json.loads(fw.content[0].text)
    ok = tools == ["forward", "python", "show_images"] and kinds.count("image") == 2 and fwj["ok"]
    return ok, {"tools": tools, "show_images_content": kinds, "python": py.content[0].text[-80:],
                "forward_worst": fwj.get("original", {}).get("worst")}


if __name__ == "__main__":
    sys.exit(main())
