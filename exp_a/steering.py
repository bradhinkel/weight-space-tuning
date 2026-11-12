"""A.6 Arm 2: contrastive steering baseline (directional ablation, no reasoning). Run on the pod:

    python -m exp_a.steering --ckpt /workspace/artifacts/a/broken --out /workspace/artifacts/a/arm2

1. Residual-stream activations at the last prompt token for "should call" (xLAM practice, user turn only) and
   "should abstain" (live_irrelevance practice) prompts, on a direction split.
2. Per layer l: candidate call direction d_l = normalize(mean_call - mean_abstain), from hidden_states[l]
   (the residual stream entering decoder layer l; l = 1..23. l = 0 is the embedding of the shared final
   prompt token, so its difference is zero; 24 is after the final norm).
3. Per candidate: directional ablation (Arditi et al. 2024) - project d_l out of every matrix that writes to the
   residual stream (embed_tokens [tied, so the unembedding too], each layer's self_attn.o_proj /
   linear_attn.out_proj, and mlp.down_proj), then generate greedily on a disjoint practice score split.
4. Selection (fixed before running): the layer maximizing practice abstain rate + practice call-correct rate
   (equal weight; the task treats both as equally important). Also reported: the best layer under the
   retention-style constraint (call-correct >= unablated - 1 prompt).
Only practice data is used. The chosen checkpoint is saved and evaluated separately on the test split.
"""
import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path

import torch

PRACTICE = Path("/workspace/artifacts/a/practice")
FN = re.compile(r"<function=([^>\n]+)>")


def load(name):
    return [json.loads(l) for l in open(PRACTICE / name)]


def user_only(r):
    return {"messages": [m for m in r["messages"] if m["role"] != "assistant"], "tools": r["tools"]}


def ref_names(r):
    a = [m for m in r["messages"] if m["role"] == "assistant"]
    return sorted(c["function"]["name"] for c in (a[0].get("tool_calls") or [])) if a else []


def called_names(text):
    names = []
    for block in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S):
        names += FN.findall(block)
    return sorted(n.strip() for n in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/artifacts/a/broken")
    ap.add_argument("--out", default="/workspace/artifacts/a/arm2")
    ap.add_argument("--n-dir", type=int, default=256)
    ap.add_argument("--n-score", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default=None, help="comma list to restrict candidates (smoke tests only)")
    a = ap.parse_args()
    t0 = time.time()
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.ckpt); tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(a.ckpt, dtype=torch.bfloat16, device_map="cuda").eval()
    lm = model.model.language_model
    layers = lm.layers

    rng = random.Random(a.seed)
    irr, xl = load("live_irrelevance.jsonl"), load("xlam_practice.jsonl")
    rng.shuffle(irr); rng.shuffle(xl)
    nd, ns = a.n_dir, a.n_score
    dir_irr, dir_xl = irr[:nd], xl[:nd]
    sc_irr, sc_xl = irr[nd:nd + ns], xl[nd:nd + ns]
    split = {"dir_irr": [r["id"] for r in dir_irr], "dir_xl": [r["id"] for r in dir_xl],
             "score_irr": [r["id"] for r in sc_irr], "score_xl": [r["id"] for r in sc_xl]}
    (out / "split.json").write_text(json.dumps(split))

    def render(r):
        it = user_only(r)
        text = tok.apply_chat_template(it["messages"], tools=it["tools"], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        return tok(text, add_special_tokens=False)["input_ids"]

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stop = sorted({tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None})

    def batches(ids, bs, extra=0, budget=131072):
        """Sorted by length; at most bs prompts and (len + extra) * n <= budget tokens per batch (as the harness)."""
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        k = 0
        while k < len(order):
            n = 1
            while k + n < len(order) and n < bs and (n + 1) * (len(ids[order[k + n]]) + extra) <= budget:
                n += 1
            idx = order[k:k + n]; k += n
            L = max(len(ids[i]) for i in idx)
            inp = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                inp[r, L - len(ids[i]):] = torch.tensor(ids[i]); att[r, L - len(ids[i]):] = 1
            yield idx, inp.cuda(), att.cuda(), L

    @torch.no_grad()
    def last_hidden(rows):
        ids = [render(r) for r in rows]
        acc = None
        for idx, inp, att, _ in batches(ids, 16, 0, 32768):
            hs = model(input_ids=inp, attention_mask=att, output_hidden_states=True).hidden_states
            h = torch.stack([x[:, -1, :].float() for x in hs], 1)  # (b, n_layers+1, d)
            acc = h.sum(0) if acc is None else acc + h.sum(0)
        return acc / len(rows)

    @torch.no_grad()
    def generate(rows):
        ids = [render(r) for r in rows]
        outs = [None] * len(ids)
        for idx, inp, att, L in batches(ids, a.batch, a.max_new):
            g = model.generate(input_ids=inp, attention_mask=att, do_sample=False, max_new_tokens=a.max_new,
                               pad_token_id=pad, eos_token_id=stop, temperature=None, top_p=None, top_k=None)
            for r, i in enumerate(idx):
                new = g[r, L:].tolist()
                cut = next((k for k, t in enumerate(new) if t in stop), len(new))
                outs[i] = tok.decode(new[:cut], skip_special_tokens=False)
        return outs

    def score(tag):
        gi, gx = generate(sc_irr), generate(sc_xl)
        abst = sum("<tool_call>" not in g for g in gi) / len(gi)
        call = sum("<tool_call>" in g for g in gx) / len(gx)
        correct = sum(called_names(g) == ref_names(r) for g, r in zip(gx, sc_xl)) / len(gx)
        res = {"tag": tag, "abstain_irr": abst, "call_rate_xl": call, "call_correct_xl": correct,
               "sum": abst + correct, "examples_irr": gi[:3], "examples_xl": gx[:2]}
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("examples")}), flush=True)
        return res

    # 1-2: directions
    t1 = time.time()
    mu_call, mu_abst = last_hidden(dir_xl), last_hidden(dir_irr)
    diff = mu_call - mu_abst  # (n_layers+1, d)
    cand = [int(x) for x in a.layers.split(",")] if a.layers else list(range(1, len(layers)))
    dirs = {l: (diff[l] / diff[l].norm()) for l in cand}
    torch.save({"mu_call": mu_call.cpu(), "mu_abst": mu_abst.cpu(), "cand": cand}, out / "directions.pt")
    print("directions", round(time.time() - t1), "s; |diff| by layer",
          [round(float(diff[l].norm()), 2) for l in range(diff.shape[0])], flush=True)

    # writers to the residual stream
    writers = [("embed", lm.embed_tokens.weight, "rows")]
    for i, L in enumerate(layers):
        att_out = L.self_attn.o_proj.weight if hasattr(L, "self_attn") else L.linear_attn.out_proj.weight
        writers += [(f"L{i}.attn_out", att_out, "cols"), (f"L{i}.mlp_down", L.mlp.down_proj.weight, "cols")]
    orig = {n: w.detach().clone() for n, w, _ in writers}

    @torch.no_grad()
    def ablate(d):
        d = d.to("cuda", torch.float32)
        for n, w, kind in writers:
            W = orig[n].float()
            W = W - torch.outer(W @ d, d) if kind == "rows" else W - torch.outer(d, d @ W)
            w.copy_(W.to(w.dtype))

    @torch.no_grad()
    def restore():
        for n, w, _ in writers:
            w.copy_(orig[n])

    # 3: score each layer
    results = {"unablated": score("unablated")}
    for l in cand:
        t2 = time.time()
        ablate(dirs[l]); r = score(f"L{l}"); restore()
        r["seconds"] = round(time.time() - t2); results[f"L{l}"] = r
        (out / "layer_scores.json").write_text(json.dumps(results, indent=1))

    base_correct = results["unablated"]["call_correct_xl"]
    best = max(cand, key=lambda l: (results[f"L{l}"]["sum"], -l))
    ok = [l for l in cand if results[f"L{l}"]["call_correct_xl"] >= base_correct - 1 / ns - 1e-9]
    best_con = max(ok, key=lambda l: (results[f"L{l}"]["abstain_irr"], -l)) if ok else None
    sel = {"selected_layer": best, "rule": "max(abstain_irr + call_correct_xl) on practice score split",
           "constrained_best_layer": best_con, "unablated": results["unablated"],
           "selected": results[f"L{best}"], "constrained": results.get(f"L{best_con}") if best_con else None}
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    print("selected", best, "constrained", best_con, flush=True)

    # 4: save the chosen checkpoint (evaluated separately)
    ablate(dirs[best])
    ck = out / "checkpoint"
    model.save_pretrained(ck)
    for f in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
              "generation_config.json"):
        if (Path(a.ckpt) / f).exists():
            shutil.copy(Path(a.ckpt) / f, ck / f)
    (out / "timing.json").write_text(json.dumps({"total_s": round(time.time() - t0)}))
    print("saved", ck, "total", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
