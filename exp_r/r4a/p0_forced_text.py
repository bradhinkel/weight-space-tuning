"""R.4a P0: forced-text check (exp_r/R4a_design.md).

Question: is the base model's answering behaviour still latent in the broken model? Each model answers the 240
BFCL v4 test irrelevance prompts with the `<tool_call>` token banned (logit -inf at every position), so it must
answer in text. The answers are then graded by scripts/score_answers.py, the A.10 blind judge.

Prompts are the exact formatted prompts of the A.1 eval (taken from its inference log), so rendering matches
the BFCL handler. Decoding as in the eval: greedy, thinking off, up to 4,096 new tokens.

    python -m exp_r.r4a.p0_forced_text --model broken=/workspace/artifacts/a/broken \
        --model base=Qwen/Qwen3.5-2B --model lora=/workspace/artifacts/a/arm3 \
        --prompts /workspace/artifacts/bfcl/a1_base/result/BFCL_v4_irrelevance_result.json \
        --out /workspace/artifacts/r4a/p0

Writes <out>/<name>/result/BFCL_v4_irrelevance_result.json in BFCL's result format (id, result,
output_token_count), so score_answers.load_run reads it unchanged.
"""
import argparse
import gc
import json
from pathlib import Path

CAP = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines()]
    items = []
    for r in rows:
        log = r["inference_log"][0]
        assert log["role"] == "inference_input"
        items.append((r["id"], log["content"]["formatted_prompt"]))
    if a.limit:
        items = items[: a.limit]
    print(f"{len(items)} prompts", flush=True)

    for spec in a.model:
        name, path = spec.split("=", 1)
        llm = LLM(model=path, gpu_memory_utilization=a.mem, max_model_len=16384, seed=0,
                  enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "video": 0})
        tok = llm.get_tokenizer()
        tc = tok.convert_tokens_to_ids("<tool_call>")
        assert isinstance(tc, int) and tc != tok.unk_token_id, tc
        sp = SamplingParams(temperature=0.0, max_tokens=CAP, logit_bias={tc: -100.0}, skip_special_tokens=False)
        outs = llm.generate([p for _, p in items], sp)
        res = []
        for (pid, _), o in zip(items, outs):
            c = o.outputs[0]
            text = c.text
            for stop in ("<|im_end|>", "<|endoftext|>"):
                text = text.split(stop)[0]
            res.append({"id": pid, "result": text, "output_token_count": len(c.token_ids),
                        "finish_reason": c.finish_reason})
        n_tc = sum("<tool_call>" in r["result"] for r in res)
        d = Path(a.out) / name / "result"
        d.mkdir(parents=True, exist_ok=True)
        (d / "BFCL_v4_irrelevance_result.json").write_text("".join(json.dumps(r) + "\n" for r in res))
        print(f"{name}: tool_call_token_id={tc} answers={len(res)} containing <tool_call>={n_tc} "
              f"hit_cap={sum(r['output_token_count'] >= CAP for r in res)}", flush=True)
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    print("P0 GENERATION DONE", flush=True)


if __name__ == "__main__":
    main()
