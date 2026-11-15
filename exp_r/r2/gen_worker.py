"""R.2 generation worker (one vLLM engine per call, in its own process so GPU memory is freed after).

    python -m exp_r.r2.gen_worker <request.json>

request: {"ckpt": dir, "out": path.jsonl, "ids_out": path.jsonl, "mem": 0.8, "keep_ids": 8192,
          "items": [{"prompt": rendered chat text, "seed": int, "max_tokens": int}, ...]}
Writes one JSON line per item to out: {"text", "n_tokens", "finish_reason"}, and the first keep_ids answer
token ids per item to ids_out (for the activation read). Sampling is the evaluation's (R.1 / think35).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from exp_r.r2.common import SAMPLING  # noqa: E402


def main(path):
    req = json.load(open(path))
    from vllm import LLM, SamplingParams
    t0 = time.time()
    max_tok = max(it["max_tokens"] for it in req["items"])
    llm = LLM(model=req["ckpt"], dtype="bfloat16", gpu_memory_utilization=req.get("mem", 0.8),
              max_model_len=max_tok + 2048, seed=0, max_num_seqs=min(512, len(req["items"])),
              limit_mm_per_prompt={"image": 0, "video": 0})          # text only: skip the vision encoder
    t_load = time.time() - t0
    sps = [SamplingParams(n=1, max_tokens=it["max_tokens"], seed=it["seed"], **SAMPLING) for it in req["items"]]
    outs = llm.generate([it["prompt"] for it in req["items"]], sps, use_tqdm=False)
    keep = req.get("keep_ids", 8192)
    with open(req["out"], "w") as f, open(req["ids_out"], "w") as g:
        for o in outs:
            x = o.outputs[0]
            f.write(json.dumps({"text": x.text, "n_tokens": len(x.token_ids), "finish_reason": x.finish_reason,
                                "prompt_tokens": len(o.prompt_token_ids)}) + "\n")
            g.write(json.dumps({"prompt_ids": list(o.prompt_token_ids), "answer_ids": list(x.token_ids[:keep])})
                    + "\n")
    tot = sum(len(o.outputs[0].token_ids) for o in outs)
    print(json.dumps({"load_s": round(t_load, 1), "gen_s": round(time.time() - t0 - t_load, 1), "tokens": tot,
                      "tokens_per_s": round(tot / max(1e-9, time.time() - t0 - t_load))}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
