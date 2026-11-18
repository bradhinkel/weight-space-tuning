"""R.4a harness self-test on the pod (no driver).

    python -m exp_r.r4a.selftest [--skip-gen] [--n 256]

1. R.2's checks, adapted: kernel API and refusals, load_base / lora (the LoRA reconstructs the fine-tune), sandbox,
   limits, request validation, the test guard, teacher-forced outputs for original / edited / base.
2. Scorer agreement with BFCL: the scorer's verdicts on A.2's saved test outputs (the broken model) against BFCL's own
   per-item verdicts, for the call categories (no GPU; BFCL data from the bfcl-venv).
3. The A.11c reproduction (design gate): A.11c's final edit (layers.23.mlp.down_proj) in a paired pass on a 256-item
   stratified practice sample. Expected from its test result: no call on irrelevant items about 40-55%, and a
   multi-call loss of at least 10 points against the broken model. If the scorer does not show the loss, the session
   does not start.
4. The judge path (needs the sidecar running on the driver host: python -m exp_r.r4a.judge_sidecar <sid>).
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np

from exp_r.r4a.common import ORIG_CKPT


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    return bool(cond)


def bfcl_agreement(cats=("simple_python", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple")):
    """Per-item agreement of exp_r.r4a.scoring with BFCL's scores on A.2's broken-model test outputs."""
    import subprocess
    from exp_r.r4a import scoring
    bdata = subprocess.run(["/workspace/bfcl-venv/bin/python", "-c", "import bfcl_eval, os; "
                            "print(os.path.join(os.path.dirname(bfcl_eval.__file__), 'data'))"],
                           capture_output=True, text=True, check=True).stdout.strip()
    root = "/workspace/artifacts/bfcl/a2_broken"
    out = {}
    for c in cats:
        sub = "live" if c.startswith("live") else "non_live"
        q = {json.loads(l)["id"]: json.loads(l) for l in open(f"{bdata}/BFCL_v4_{c}.json")}
        gt = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(f"{bdata}/possible_answer/BFCL_v4_{c}.json")}
        res = [json.loads(l) for l in open(f"{root}/result/Qwen3.5-local-FC/{sub}/BFCL_v4_{c}_result.json")]
        sc = [json.loads(l) for l in open(f"{root}/score/Qwen3.5-local-FC/{sub}/BFCL_v4_{c}_score.json")]
        failed = {r["id"] for r in sc[1:] if "id" in r}
        agree = n = ours_ok = theirs_ok = 0
        for r in res:
            raw = next((x["content"]["raw_model_response"] for x in r.get("inference_log", [])
                        if isinstance(x, dict) and isinstance(x.get("content"), dict)
                        and "raw_model_response" in x["content"]), None)
            if raw is None:
                continue
            tools = [{"type": "function", "function": f} for f in q[r["id"]]["function"]]
            item = {"tools": tools, "reference": gt[r["id"]], "ref_format": "bfcl",
                    "kind": "single" if len(gt[r["id"]]) == 1 else "multi"}
            ours = bool(scoring.score(item, raw, 0, 4096)["correct"])
            theirs = r["id"] not in failed
            n += 1
            agree += ours == theirs
            ours_ok += ours
            theirs_ok += theirs
        out[c] = {"items": n, "agreement": round(agree / max(n, 1), 4), "ours_correct": ours_ok,
                  "bfcl_correct": theirs_ok}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    a = ap.parse_args()
    ok = []

    # ---------------------------------------------------------------- 2. scorer agreement (no GPU)
    agr = bfcl_agreement()
    print(json.dumps(agr, indent=1))
    tot = sum(v["items"] for v in agr.values())
    mean = sum(v["agreement"] * v["items"] for v in agr.values()) / max(tot, 1)
    ok.append(check("scorer agrees with BFCL per item (>= 0.95 overall)", mean >= 0.95, f"{mean:.4f} over {tot}"))
    ok.append(check("scorer agrees on parallel / parallel_multiple (>= 0.9 each)",
                    agr["parallel"]["agreement"] >= 0.9 and agr["parallel_multiple"]["agreement"] >= 0.9))

    from exp_r.r4a.server import R4aHarness
    sid = "selftest_" + time.strftime("%Y%m%dT%H%M%S")
    h = R4aHarness({"session_id": sid, "budget": 8, "vllm_mem": 0.55,
                    "limits": {"exec_timeout_s": 30, "kernel_mem_bytes": 30 << 30}})
    py = h.python
    work = f"/srv/wst-sessions/{sid}/work"

    # ---------------------------------------------------------------- 1. kernel
    out = py("len(NAMES), tuple(load('model.language_model.layers.15.mlp.down_proj.weight').shape), len(LORA_NAMES), "
             "LORA_RANK, LORA_SCALE")
    ok.append(check("kernel NAMES/load/LORA_NAMES", "320" in out and "(2048, 6144)" in out and "186" in out,
                    out.splitlines()[1][:160]))
    out = py("n='model.language_model.layers.22.mlp.down_proj.weight'\nA,B,s=lora(n)\nd=load_original(n)-load_base(n)\n"
             "r=(d-s*B@A).norm()/d.norm()\nc=(d.flatten()@(s*B@A).flatten())/(d.norm()*(s*B@A).norm())\n"
             "print('REL', float(r), 'COS', float(c), tuple(A.shape), tuple(B.shape))")
    rel = float(out.split("REL")[1].split()[0]) if "REL" in out else 9
    cos = float(out.split("COS")[1].split()[0]) if "COS" in out else 0
    ok.append(check("lora() reconstructs the fine-tune up to bf16 rounding (cos > 0.99, rel. error < 0.15)",
                    cos > 0.99 and rel < 0.15, f"cos {cos:.4f} rel {rel:.4f}"))
    out = py("print(lora('model.language_model.layers.0.input_layernorm.weight'))")
    ok.append(check("lora() None for untouched tensors", "None" in out))
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
    ok.append(check("save/drop edit", "saved" in out and "dropped" in out, out.replace("\n", " | ")[:160]))
    out = py("import os\nfor p in ['/workspace/artifacts/a/test_guard.json','/workspace/r4a/guard','/workspace/repo/README.md',"
             "'/workspace/r4a/sessions','/workspace/bfcl-venv','/root']:\n try:\n  open(p).read(1) if not os.path.isdir(p) "
             "else os.listdir(p); print('READ', p)\n except Exception as e: print('BLOCKED', p, type(e).__name__)")
    ok.append(check("sandbox blocks guard/repo/sessions/BFCL data", out.count("BLOCKED") == 6, out.replace("\n", " | ")[:240]))
    out = py("import socket\ntry:\n socket.socket(socket.AF_INET).connect(('1.1.1.1',80)); print('NET')\n"
             "except Exception as e: print('NONET', type(e).__name__)")
    ok.append(check("no network", "NONET" in out))
    out = py("print(len(open('/practice/live_relevant.jsonl').readlines())); print(os.listdir('/model'));"
             " open('/work/x.txt','w').write('ok'); print(os.listdir('/work'))")
    ok.append(check("read /model,/practice; write /work", "56" in out and "x.txt" in out and "adapter" in out))
    out = py("import torch\nx=torch.ones(3, requires_grad=True)\ntry:\n (x*2).sum().backward(); print('GRAD')\n"
             "except RuntimeError as e: print('NOGRAD', str(e)[:40])\nprint(torch.cuda.is_available())")
    ok.append(check("no gradients, no GPU", "NOGRAD" in out and "False" in out))
    out = py("import time\nkeep=1\nwhile True: time.sleep(1)")
    ok.append(check("time limit interrupts", "interrupted" in out.lower() or "KeyboardInterrupt" in out, out[-120:]))
    out = py("keep")
    ok.append(check("variables survive interrupt", "\n1\n" in out, out[-60:]))
    out = py("import numpy as np\nbig=np.ones((60, 1024, 1024, 1024), np.uint8); print('ALLOC')")
    ok.append(check("memory limit", "MemoryError" in out or "restarted" in out, out[-120:]))

    # ---------------------------------------------------------------- request validation and guard
    def fwd(rq):
        return json.loads(h.forward(rq))

    p0 = h.passes
    r = fwd({"type": "generate", "model": "original", "items": ["nope"]})
    ok.append(check("unknown id refused, free", not r["ok"] and h.passes == p0, r.get("error", "")[:80]))
    t0 = json.loads(open("/workspace/r4a/guard/test_user_messages.jsonl").readline())["text"]
    r = fwd({"type": "generate", "model": "original", "prompts": [{"messages": [{"role": "user", "content": t0}],
                                                                   "tools": []}]})
    ok.append(check("test user message as prompt refused", not r["ok"] and "evaluation item" in r.get("error", "")))
    r = fwd({"type": "teacher_forced", "texts": ["Something first. " + t0]})
    ok.append(check("test text refused (teacher_forced)", not r["ok"]))
    hits = [pid for pid, it in h.practice.items() if h.guard.hit_messages(it["messages"])]
    ok.append(check("guard does not refuse practice items (< 1%)", len(hits) < 0.01 * len(h.practice),
                    f"{len(hits)} of {len(h.practice)}: {hits[:5]}"))
    ids = list(h.practice)[:300]
    r = fwd({"type": "generate", "model": "both", "items": ids})
    ok.append(check("pair > 256 refused", not r["ok"]))
    r = fwd({"type": "generate", "model": "edited", "items": ids[:2]})
    ok.append(check("edited without edits refused", not r["ok"]))
    r = fwd({"type": "judge", "answers": [{"file": "/work/x.txt"}]})
    ok.append(check("judge on a non-pass file refused", not r["ok"]))
    ok.append(check("validation failures cost nothing", h.passes == p0))

    # ---------------------------------------------------------------- teacher-forced
    ex = [h.practice[k] for k in ids[:4]]
    chats = [{"messages": it["messages"] + [{"role": "assistant", "content": "I can help with that."}],
              "tools": it["tools"]} for it in ex]
    r = fwd({"type": "teacher_forced", "model": "original", "chats": chats,
             "return": {"token_logprobs": True, "topk": 5, "hidden_states": [0, 15, 24], "mlp_act": [15]}})
    ok.append(check("teacher_forced dense", r["ok"] and r["files"]["hidden_15"][2] == 2048
                    and r["files"]["mlp_act_15"][2] == 6144, json.dumps(r.get("files"))[:200]))
    lp_orig = r.get("mean_token_logprob")
    r = fwd({"type": "teacher_forced", "model": "base", "chats": chats, "return": {"token_logprobs": True}})
    ok.append(check("teacher_forced base differs from original", r["ok"] and r["mean_token_logprob"] != lp_orig,
                    f"{lp_orig} vs {r.get('mean_token_logprob')}"))
    r = fwd({"type": "teacher_forced", "model": "original", "chats": chats[:2], "return": {"attn_layers": [15, 23]}})
    ok.append(check("attention probabilities", r["ok"] and r["files"]["attn_15"][1] == 8, json.dumps(r.get("files"))[:160]))

    # ---------------------------------------------------------------- 3. A.11c reproduction
    if not a.skip_gen:
        from safetensors import safe_open
        n = "model.language_model.layers.23.mlp.down_proj.weight"
        with safe_open(glob.glob("/workspace/artifacts/sessions/a11c_b25_fable_20261113T050808Z/final_checkpoint/"
                                 "*.safetensors")[0], "pt") as f:
            e = f.get_tensor(n).float().numpy()
        os.makedirs(f"{work}/edited", exist_ok=True)
        np.save(f"{work}/edited/{n}.npy", e)
        h._own(f"{work}/edited")
        rng = random.Random(0)
        by = {}
        for pid, it in h.practice.items():
            g = f"xlam_{it['kind']}" if it["category"] == "xlam_relevant" else it["category"]
            by.setdefault(g, []).append(pid)
        live_ref = [p for p in by["live_relevant"] if h.practice[p]["kind"] in ("single", "multi")]
        sample = (rng.sample(by["live_irrelevance"], 64) + rng.sample(by["xlam_mismatched"], 64)
                  + rng.sample(by["xlam_single"], 32) + rng.sample(by["xlam_multi"], 64) + rng.sample(live_ref, 32))
        assert len(sample) == 256 and len(set(sample)) == 256
        r = fwd({"type": "generate", "model": "both", "items": sample, "activations": {"layers": [23], "mlp": True}})
        print(json.dumps({k: r.get(k) for k in ("original", "edited", "paired")}, indent=1)[:5000])
        ok.append(check("paired A.11c generation ran", r["ok"], r.get("error", "")[:200]))
        if r["ok"]:
            ed, pr = r["edited"], r["paired"]
            irr = [ed[g]["correct_rate"] for g in ("live_irrelevance", "xlam_mismatched")]
            nc = (irr[0] + irr[1]) / 2
            ok.append(check("A.11c: no call on irrelevant items 0.30-0.65 (test: 0.50 / 0.39)", 0.30 <= nc <= 0.65,
                            f"live {irr[0]}, mismatched {irr[1]}"))
            mg = [g for g in pr if "multi" in g]
            same = pr.get("xlam_multi_same_fn", {})
            ok.append(check("A.11c: same-function multi-call loss >= 10 points vs broken (DESIGN GATE, amended "
                            "2026-11-17: practice has no counterpart to BFCL parallel_multiple)",
                            same.get("correct_rate_difference", 0) <= -0.10, json.dumps({g: pr[g] for g in mg})))
            ok.append(check("broken model's xLAM accuracy is high (> 0.7; it was trained on xLAM)",
                            r["original"].get("xlam_single", {}).get("correct_rate", 0) > 0.7,
                            json.dumps(r["original"].get("xlam_single"))))
            ix = json.load(open(f"{work}/gen_{h.passes}_acts/edited/index.json"))
            ok.append(check("generation activations", ix["files"]["hidden_last_prompt.npy"][1:] == [1, 2048]
                            and ix["files"]["mlp_last_prompt.npy"][1:] == [1, 6144], json.dumps(ix["files"])))
            r2 = fwd({"type": "generate", "model": "base", "items": sample[:32]})
            ok.append(check("base generation", r2["ok"], json.dumps(r2.get("summary", {}))[:200]))
            # ------------------------------------------------------------ 4. judge path
            if not a.skip_judge:
                print(f"judge pass: start the sidecar on the driver host now: python -m exp_r.r4a.judge_sidecar {sid}",
                      flush=True)
                k = h.passes - 1
                r3 = fwd({"type": "judge", "answers": [{"file": f"/work/gen_{k}_edited.jsonl"}]})
                ok.append(check("judge pass", r3["ok"], json.dumps(r3.get("by_file") or r3.get("error"))[:300]))
    print(f"\n{sum(ok)}/{len(ok)} checks passed; passes used {h.passes}; session dir {h.sdir}")


if __name__ == "__main__":
    main()
