"""Shared LoRA entry point (used by A.2, A.7, B.2, C.3).

    python -m common.train.lora --data train.jsonl --out /workspace/artifacts/a/broken \
        [--base Qwen/Qwen3.5-2B] [--epochs 1] [--max-steps N]

Data: JSONL of {"messages": [...], "tools": [...]}. The prompt is rendered with the model's chat template
(enable_thinking=False); loss is on the final assistant turn only. Defaults follow PROJECT_PLAN.md:
rank 16, alpha 32, all linear layers of the language model, lr 2e-4, seed 0.
(Vision-tower linears receive no gradient from text-only data, so restricting LoRA to the language
model is numerically equivalent to "all linear" for A/B and keeps the adapter small. Exp C overrides.)

Saves: <out>/adapter (PEFT), <out> merged full checkpoint (+tokenizer, processor), <out>/train_log.json.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

LM_LINEAR = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$"


def render(tok, ex, max_len):
    msgs, tools = ex["messages"], ex.get("tools") or None
    prompt = tok.apply_chat_template(msgs[:-1], tools=tools, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    full = tok.apply_chat_template(msgs, tools=tools, tokenize=False, enable_thinking=False)
    if not full.startswith(prompt):
        # Template renders historical assistant turns without the empty think block; splice the target on.
        tail = tok.apply_chat_template(msgs, tools=tools, tokenize=False)[len(
            tok.apply_chat_template(msgs[:-1], tools=tools, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True)):]
        tail = tail.split("</think>")[-1].lstrip("\n")
        full = prompt + tail
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(f_ids) > max_len:
        return None
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids, labels, full[len(prompt):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--init-from", default=None, help="merged checkpoint to start from (A.7)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch-tokens", type=int, default=32768, help="tokens per optimizer step (padded)")
    ap.add_argument("--micro-tokens", type=int, default=16384)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-regex", default=LM_LINEAR)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    src = a.init_from or a.base
    tok = AutoTokenizer.from_pretrained(src)
    rows = [json.loads(l) for l in open(a.data)]
    data, skipped = [], 0
    for ex in rows:
        r = render(tok, ex, a.max_len)
        if r is None:
            skipped += 1
        else:
            data.append(r[:2])
    print(f"examples {len(data)} (skipped >{a.max_len} tok: {skipped}); sample target: {render(tok, rows[0], a.max_len)[2]!r}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    pm = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0,
                                          target_modules=a.target_regex, task_type="CAUSAL_LM"))
    pm.print_trainable_parameters()

    # Length-bucketed micro-batches (seeded), shuffled order of buckets.
    order = list(range(len(data))); random.Random(a.seed).shuffle(order)
    chunks = [sorted(order[i:i + 512], key=lambda j: len(data[j][0])) for i in range(0, len(order), 512)]
    micro = []
    for ch in chunks:
        cur, cur_max = [], 0
        for j in ch:
            L = len(data[j][0])
            if cur and max(cur_max, L) * (len(cur) + 1) > a.micro_tokens:
                micro.append(cur); cur, cur_max = [], 0
            cur.append(j); cur_max = max(cur_max, L)
        if cur:
            micro.append(cur)
    random.Random(a.seed + 1).shuffle(micro)
    tokens_per_micro = sum(max(len(data[j][0]) for j in m) * len(m) for m in micro) / len(micro)
    accum = max(1, round(a.batch_tokens / tokens_per_micro))
    steps_per_epoch = math.ceil(len(micro) / accum)
    total = a.max_steps or math.ceil(steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    print(f"micro-batches {len(micro)}, accum {accum}, steps/epoch {steps_per_epoch}, total steps {total}", flush=True)

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    log, t0, step, mi = [], time.time(), 0, 0
    pm.train()
    while step < total:
        tot_loss, tot_tok = 0.0, 0
        for _ in range(accum):
            m = micro[mi % len(micro)]; mi += 1
            L = max(len(data[j][0]) for j in m)
            ids = torch.full((len(m), L), pad); lab = torch.full((len(m), L), -100); att = torch.zeros((len(m), L), dtype=torch.long)
            for r, j in enumerate(m):
                x, y = data[j]
                ids[r, :len(x)] = torch.tensor(x); lab[r, :len(y)] = torch.tensor(y); att[r, :len(x)] = 1
            ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
            n = int((lab[:, 1:] != -100).sum())
            loss = pm(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss * n).backward()
            tot_loss += loss.item() * n; tot_tok += n
        for p in pm.parameters():
            if p.grad is not None:
                p.grad.div_(max(tot_tok, 1))
        torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        rec = {"step": step, "loss": round(tot_loss / max(tot_tok, 1), 5), "lr": sched.get_last_lr()[0],
               "elapsed_s": round(time.time() - t0)}
        log.append(rec)
        if step % 10 == 0 or step == 1:
            eta = (time.time() - t0) / step * (total - step)
            print({**rec, "eta_min": round(eta / 60)}, flush=True)
        if step % 200 == 0:
            pm.save_pretrained(out / "adapter")
    pm.save_pretrained(out / "adapter")
    merged = pm.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    try:
        AutoProcessor.from_pretrained(src).save_pretrained(out)
    except Exception as e:
        print("processor save skipped:", e)
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "n_examples": len(data), "skipped": skipped,
                                                    "accum": accum, "total_steps": total, "log": log}, indent=1))
    print("saved", out, "in", round((time.time() - t0) / 60), "min")


if __name__ == "__main__":
    main()
