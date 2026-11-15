"""R.2 harness self-test on the pod (no driver): kernel API and refusals, sandbox, limits, request validation,
teacher-forced outputs, and a paired generation pass with the original's p50c edit (expected: a large gain).

    python -m exp_r.r2.selftest [--gen 64] [--skip-gen]
"""
import argparse
import json
import os
import time

import numpy as np


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=64, help="practice problems in the paired p50c pass")
    ap.add_argument("--skip-gen", action="store_true")
    a = ap.parse_args()
    from exp_r.r2.server import R2Harness
    sid = "selftest_" + time.strftime("%Y%m%dT%H%M%S")
    h = R2Harness({"session_id": sid, "budget": 5,
                   "limits": {"exec_timeout_s": 20, "kernel_mem_bytes": 40 << 30}})
    ok = []
    py = lambda c: h.python(c)

    out = py("len(NAMES), NAMES[0], tuple(load('model.language_model.layers.15.mlp.down_proj.weight').shape)")
    ok.append(check("kernel NAMES/load", "320" in out and "(1024, 3584)" in out, out.splitlines()[1][:120]))
    out = py("try:\n save_edit('model.language_model.norm.weight', load('model.language_model.norm.weight')+1)\n"
             "except ValueError as e: print('REFUSED', e)")
    ok.append(check("refuse final norm", "REFUSED" in out))
    out = py("try:\n save_edit('model.language_model.embed_tokens.weight', load('model.language_model.embed_tokens.weight'))\n"
             "except ValueError as e: print('REFUSED', e)")
    ok.append(check("refuse embedding", "REFUSED" in out))
    out = py("n='model.language_model.layers.3.mlp.up_proj.weight'\ntry:\n save_edit(n, load(n)*1.1)\n"
             "except ValueError as e: print('REFUSED', e)")
    ok.append(check("refuse whole-tensor rescale", "REFUSED" in out and "constant" in out))
    out = py("w=load(n); w[0,:]*=0.5; print(save_edit(n, w)); print(edited()); print(drop_edit(n)); print(edited())")
    ok.append(check("save/drop edit", "saved" in out and "dropped" in out and "[]" in [x.strip() for x in out.splitlines()],
                    out.replace("\n", " | ")[:160]))

    out = py("import os\nfor p in ['/workspace/r1/bench/math500.jsonl','/workspace/repo/README.md','/workspace/r2/sessions',"
             "'/root']:\n try:\n  open(p).read(1) if not os.path.isdir(p) else os.listdir(p); print('READ', p)\n"
             " except Exception as e: print('BLOCKED', p, type(e).__name__)")
    ok.append(check("sandbox blocks bench/repo/sessions", out.count("BLOCKED") == 4, out.replace("\n", " | ")[:200]))
    out = py("import socket\ntry:\n socket.socket(socket.AF_INET).connect(('1.1.1.1',80)); print('NET')\n"
             "except Exception as e: print('NONET', type(e).__name__)")
    ok.append(check("no network", "NONET" in out))
    out = py("print(open('/practice/README.txt').read()[:40]); print(len(open('/model/config.json').read()) > 100);"
             " open('/work/x.txt','w').write('ok'); print(os.listdir('/work'))")
    ok.append(check("read /model,/practice; write /work", "True" in out and "x.txt" in out))
    out = py("import torch\nx=torch.ones(3, requires_grad=True)\ntry:\n (x*2).sum().backward(); print('GRAD')\n"
             "except RuntimeError as e: print('NOGRAD', str(e)[:40])\nprint(torch.cuda.is_available())")
    ok.append(check("no gradients, no GPU", "NOGRAD" in out and "False" in out))
    out = py("import time\nkeep=1\nwhile True: time.sleep(1)")
    ok.append(check("time limit interrupts", "interrupted" in out.lower() or "KeyboardInterrupt" in out, out[-120:]))
    out = py("keep")
    ok.append(check("variables survive interrupt", "\n1\n" in out, out[-60:]))
    out = py("import numpy as np\nbig=np.ones((60, 1024, 1024, 1024), np.uint8); print('ALLOC')")
    ok.append(check("memory limit", "MemoryError" in out or "restarted" in out, out[-120:]))

    def fwd(rq):
        return json.loads(h.forward(rq))

    p0 = h.passes
    r = fwd({"type": "generate", "model": "original", "problems": ["nope"]})
    ok.append(check("unknown id refused, free", not r["ok"] and h.passes == p0, r.get("error", "")[:80]))
    m5 = json.loads(open("/workspace/r1/bench/math500.jsonl").readline())["problem"]
    r = fwd({"type": "generate", "model": "original", "prompts": ["Solve this: " + m5]})
    ok.append(check("test-problem prompt refused", not r["ok"] and "evaluation problem" in r.get("error", "")))
    r = fwd({"type": "teacher_forced", "texts": ["We are told " + m5]})
    ok.append(check("test-problem text refused", not r["ok"]))
    ids = [k for k in h.practice if k.startswith("math_train_algebra")][:300]
    r = fwd({"type": "generate", "model": "both", "problems": ids})
    ok.append(check("both > 256 refused", not r["ok"]))
    r = fwd({"type": "generate", "model": "edited", "problems": ids[:2]})
    ok.append(check("edited without edits refused", not r["ok"]))

    sols = [h.practice[k]["solution"] for k in ids[:4]]
    r = fwd({"type": "teacher_forced", "model": "original", "texts": sols,
             "return": {"token_logprobs": True, "topk": 5, "hidden_states": [0, 15, 24], "mlp_act": [15]}})
    ok.append(check("teacher_forced dense", r["ok"] and r["files"]["hidden_15"][2] == 1024
                    and r["files"]["mlp_act_15"][2] == 3584, json.dumps(r.get("files"))[:200]))
    base_lp = r.get("mean_token_logprob")
    d = f"/srv/wst-sessions/{sid}/work/fwd_{h.passes}"
    tl = np.load(f"{d}/token_logprobs.npy")
    ok.append(check("token_logprobs sane", np.isfinite(tl).all() and tl.max() <= 0, f"mean {tl[0, :r['lengths'][0]-1].mean():.3f}"))
    r = fwd({"type": "teacher_forced", "model": "original", "texts": sols[:2], "return": {"attn_layers": [15, 19]}})
    ok.append(check("attention probabilities", r["ok"] and r["files"]["attn_15"][1] == 8, json.dumps(r.get("files"))[:160]))
    if r["ok"]:
        at = np.load(f"/srv/wst-sessions/{sid}/work/fwd_{h.passes}/attn_15.npy")
        L = r["lengths"][0]
        ok.append(check("attention rows sum to 1", abs(at[0, :, :L, :L].sum(-1).mean() - 1) < 1e-2))

    # the original's p50c edit, written as edits (as save_edit would)
    from safetensors import safe_open
    edits = {}
    with safe_open("/workspace/r1/models/p50c/model.safetensors-00001-of-00001.safetensors", "pt") as fe, \
            safe_open("/workspace/r2/model/model.safetensors", "pt") as fb:
        for n in fb.keys():
            e = fe.get_tensor(n)
            if not (e == fb.get_tensor(n)).all():
                edits[n] = e.float().numpy()
    os.makedirs(f"/srv/wst-sessions/{sid}/work/edited", exist_ok=True)
    for n, a_ in edits.items():
        np.save(f"/srv/wst-sessions/{sid}/work/edited/{n}.npy", a_)
    h._own(f"/srv/wst-sessions/{sid}/work/edited")
    out = py("edited()")
    ok.append(check("p50c edits visible in kernel", "layers.15.mlp.down_proj" in out, out.splitlines()[1][:160]))
    r = fwd({"type": "teacher_forced", "model": "edited", "texts": sols, "return": {"token_logprobs": True}})
    ok.append(check("edited teacher_forced differs", r["ok"] and r["mean_token_logprob"] != base_lp,
                    f"{base_lp} vs {r.get('mean_token_logprob')}"))

    if not a.skip_gen:
        g = [k for k in h.practice if k.startswith("math_train")][::97][: a.gen // 2] + \
            [k for k in h.practice if k.startswith("gsm8k")][::97][: a.gen // 2]
        r = fwd({"type": "generate", "model": "both", "problems": g, "n": 1,
                 "activations": {"layers": [15, 19], "mlp": True}})
        print(json.dumps(r, indent=1)[:3000])
        ok.append(check("paired p50c generation ran", r["ok"]))
        if r["ok"]:
            pr = r["paired"]
            ok.append(check("p50c gain >> SE", pr["accuracy_difference"] > 2 * (pr["standard_error"] or 1),
                            f"diff {pr['accuracy_difference']} se {pr['standard_error']}; cut "
                            f"{r['original']['cut_share']} -> {r['edited']['cut_share']}"))
            ix = json.load(open(f"/srv/wst-sessions/{sid}/work/gen_{h.passes}_acts/edited/index.json"))
            ok.append(check("generation activations", ix["files"]["hidden_mean_answer.npy"] == [len(g), 2, 1024]
                            and ix["files"]["mlp_mean_answer.npy"] == [len(g), 2, 3584], json.dumps(ix["files"])))
    print(f"\n{sum(ok)}/{len(ok)} checks passed; passes used {h.passes}; session dir {h.sdir}")


if __name__ == "__main__":
    main()
