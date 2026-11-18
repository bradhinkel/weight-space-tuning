"""R.4a generation worker (one vLLM engine per call, in its own process so GPU memory is freed after).

    python -m exp_r.r4a.gen_worker <request.json>

request: {"ckpt": dir, "out": path.jsonl, "ids_out": path.jsonl, "mem": 0.55, "keep_ids": 1024,
          "max_model_len": int, "items": [{"prompt": rendered chat text, "max_tokens": int}, ...]}
Decoding is the evaluation's (A.1 / common/evals/bfcl.py): greedy, thinking off (in the prompt), up to 4,096 tokens.
"""
import json
import sys
import time


def main(path):
    req = json.load(open(path))
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=req["ckpt"], dtype="bfloat16", gpu_memory_utilization=req.get("mem", 0.55),
              max_model_len=int(req["max_model_len"]), seed=0, max_num_seqs=min(256, len(req["items"])),
              limit_mm_per_prompt={"image": 0, "video": 0})
    t_load = time.time() - t0
    sps = [SamplingParams(n=1, temperature=0.0, max_tokens=it["max_tokens"]) for it in req["items"]]
    outs = llm.generate([it["prompt"] for it in req["items"]], sps, use_tqdm=False)
    keep = req.get("keep_ids", 1024)
    with open(req["out"], "w") as f, open(req["ids_out"], "w") as g:
        for o in outs:
            x = o.outputs[0]
            f.write(json.dumps({"text": x.text, "n_tokens": len(x.token_ids), "finish_reason": x.finish_reason,
                                "prompt_tokens": len(o.prompt_token_ids)}) + "\n")
            g.write(json.dumps({"prompt_ids": list(o.prompt_token_ids), "answer_ids": list(x.token_ids[:keep])})
                    + "\n")
    tot = sum(len(o.outputs[0].token_ids) for o in outs)
    print(json.dumps({"load_s": round(t_load, 1), "gen_s": round(time.time() - t0 - t_load, 1), "tokens": tot}),
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
