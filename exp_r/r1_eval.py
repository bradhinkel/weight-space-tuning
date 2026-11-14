"""R.1: reproduce the original Experiment 8 evaluation (Qwen3.5-0.8B, thinking on) with their exact protocol.

Mirrors exp8_math/code/evaluate.py `run()` with protocol `think35` from
https://github.com/RikiyaT/lm_understand-logs (MIT): chat template with enable_thinking=True, user turn =
problem + "\\nPlease reason step by step, and put your final answer within \\boxed{}." (mathqa.build_prompt),
vLLM sampling temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, max 32,768 new tokens, seed 0;
samples per problem MATH-500 4, GSM8K 2; the last \\boxed{} answer graded with math-verify (mathqa.is_correct);
score = mean accuracy over all samples; cut_share = share of samples that hit max_tokens.

    python -m exp_r.r1_eval prepare --bench /workspace/r1/bench
    python -m exp_r.r1_eval run <ckpt> <out_dir> --bench /workspace/r1/bench --sets math500,gsm8k [--shard i/k]
    python -m exp_r.r1_eval merge <out_root>      # combine shards: <out_root>/<anything>/scores.json

Writes, like the original: scores.json (per set: acc, cut_share, mean_tokens, ...), <set>_per_problem.json
(id, correct per sample, first extracted answer), <set>_first_samples.jsonl (first 20 full texts).
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import mathqa as Q  # noqa: E402

SAMPLES = {"math500": 4, "gsm8k": 2}
SAMPLING_THINK35 = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768,
                    "seed": 0}


def prepare(bench):
    from datasets import load_dataset
    os.makedirs(bench, exist_ok=True)

    def dump(name, rows):
        with open(os.path.join(bench, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))

    dump("math500", [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                     for r in load_dataset("HuggingFaceH4/MATH-500", split="test")])
    dump("gsm8k", [{"id": f"gsm8k_test_{i}", "problem": r["question"],
                    "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                   for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))])


def run(ckpt, out, bench, sets, shard=None, mem=0.85, max_num_seqs=None):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    kw = {"max_num_seqs": max_num_seqs} if max_num_seqs else {}
    llm = LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=mem,
              max_model_len=SAMPLING_THINK35["max_tokens"] + 2048, seed=0, **kw)
    sp_path = os.path.join(out, "scores.json")
    res = json.load(open(sp_path)) if os.path.exists(sp_path) else {}
    res["_job"] = {"protocol": "think35", "ckpt": ckpt, "shard": shard}
    for name in sets:
        if name in res:
            continue
        t0 = time.time()
        rows = [json.loads(l) for l in open(os.path.join(bench, f"{name}.jsonl"))]
        if shard:
            i, k = map(int, shard.split("/"))
            rows = rows[i::k]
        prompts = [tok.apply_chat_template([{"role": "user", "content": Q.build_prompt(r["problem"])}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=True)
                   for r in rows]
        outs = llm.generate(prompts, SamplingParams(n=SAMPLES[name], **SAMPLING_THINK35))
        bad = sum(1 for o in outs for x in o.outputs if x.finish_reason not in ("stop", "length"))
        if bad:
            raise RuntimeError(f"{bad} generations ended without finishing or reaching max_tokens; not saved")
        per, tot, ok, cut, lens, noans = [], 0, 0, 0, [], 0
        for r, o in zip(rows, outs):
            c = []
            for s in o.outputs:
                pred, _ = Q.extract(s.text)
                good = Q.is_correct(pred, r["answer"])
                c.append(int(good))
                cut += s.finish_reason == "length"
                noans += pred is None and s.finish_reason != "length"
                lens.append(len(s.token_ids))
            per.append({"id": r["id"], "correct": c, "pred_first": Q.extract(o.outputs[0].text)[0]})
            tot += len(c)
            ok += sum(c)
        res[name] = {"acc": ok / tot, "n_problems": len(rows), "samples": SAMPLES[name], "cut_share": cut / tot,
                     "mean_tokens": sum(lens) / len(lens), "s": round(time.time() - t0), "ok": ok, "total": tot,
                     "cut": cut, "tokens_sum": sum(lens), "shard": shard, "no_answer_finished": noans,
                     "tokens_per_s": round(sum(lens) / max(1, time.time() - t0))}
        json.dump(per, open(os.path.join(out, f"{name}_per_problem.json"), "w"))
        with open(os.path.join(out, f"{name}_first_samples.jsonl"), "w") as f:
            for r, o in list(zip(rows, outs))[:20]:
                f.write(json.dumps({"id": r["id"], "answer": r["answer"], "text": o.outputs[0].text}) + "\n")
        json.dump(res, open(sp_path, "w"), indent=1)
        print(name, json.dumps(res[name]), flush=True)


def merge(root):
    tot = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "scores.json"))):
        for name, v in json.load(open(p)).items():
            if name.startswith("_"):
                continue
            t = tot.setdefault(name, {"ok": 0, "total": 0, "cut": 0, "tokens_sum": 0, "problems": 0, "shards": 0})
            for k in ("ok", "total", "cut", "tokens_sum"):
                t[k] += v[k]
            t["problems"] += v["n_problems"]
            t["shards"] += 1
    for t in tot.values():
        t.update(acc=t["ok"] / t["total"], cut_share=t["cut"] / t["total"], mean_tokens=t["tokens_sum"] / t["total"])
    json.dump(tot, open(os.path.join(root, "merged.json"), "w"), indent=1)
    print(json.dumps(tot, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "merge"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--bench", default="/workspace/r1/bench")
    ap.add_argument("--sets", default="math500,gsm8k")
    ap.add_argument("--shard")
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.bench)
    elif a.cmd == "run":
        run(a.args[0], a.args[1], a.bench, a.sets.split(","), a.shard, a.mem, a.max_num_seqs)
    else:
        merge(a.args[0])


if __name__ == "__main__":
    main()
