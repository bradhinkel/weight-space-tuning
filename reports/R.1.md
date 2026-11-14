# R.1 Exact replication of the original Experiment 8 evaluation

## What was done
- **Goal:** before changing anything, check that we can reproduce the original math result ("Sparks of Weight-Space Programming", Experiment 8; https://github.com/RikiyaT/lm_understand-logs, `exp8_math/`) with their own model, edit and evaluation protocol.
- **Target:** Qwen/Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17`, thinking on, with their `generation_config.json` (eos 248044).
- **Edits:** their published p10c, p25c and p50c edits, rebuilt with their `apply_edit.py`/`merge.py`; all tensor hashes matched. Only **base** and **p50c** were evaluated (p10c and p25c are built but not run).
- **Protocol** (`exp_r/r1_eval.py`, mirroring their `evaluate.py` `think35`):
  - their chat template and prompt (`mathqa.build_prompt`);
  - vLLM 0.33, bf16, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, up to 32,768 new tokens, seed 0;
  - MATH-500 with 4 answers per problem, GSM8K with 2;
  - graded by their `mathqa` (last `\boxed{}`, math-verify 0.9.0).
- **Benchmarks:** MATH-500 (500) and GSM8K test (1,319), with problem IDs identical to theirs.
- **Not covered:** IFEval (their instruction checkers aren't released) and their other sets (AIME, AMC, Minerva, OlympiadBench, GPQA, MMLU-Pro).
- **Run:** 10 shards per set and model, on two A40 pods with two vLLM processes per GPU.
- **Comparison:** problem by problem against their published `eval/{unedited,p50c}/parts` (`exp_r/r1_compare.py`).

## Results

| model | set | ours | theirs | ours − theirs (± SE) | per-problem r | cut at 32k: ours / theirs | mean tokens (ours) |
|---|---|---|---|---|---|---|---|
| base | MATH-500 | 7.70 | 7.90 | −0.20 ± 0.65 | 0.76 | 93.0% / 93.0% | 30,933 |
| base | GSM8K | 13.76 | 13.84 | −0.08 ± 0.76 | 0.50 | 87.6% / 87.5% | 29,314 |
| **p50c** | MATH-500 | **50.85** | 48.10 | +2.75 ± 0.97 | 0.87 | 32.2% / 33.1% | 21,790 |
| **p50c** | GSM8K | **67.29** | 67.10 | +0.19 ± 0.78 | 0.77 | 7.0% / 7.2% | 11,641 |

- The SE is over problems, using per-problem differences, so it includes sampling noise on both sides.
- All 500 and 1,319 IDs matched; no problem was missing on either side.
- **Problems solved by at least one answer:**

  | | ours | theirs | both |
  |---|---|---|---|
  | base MATH-500 | 83 | 77 | 52 |
  | p50c MATH-500 | 342 | 320 | 300 |
  | p50c GSM8K | 1,007 | 1,008 | 935 |

- **Grader check:** we regraded their 800 published answer texts (the first answer for 20 problems per shard, for both models and sets) with our grader. It agreed with their recorded grade on all 800.

## What it means
1. **The headline result reproduces.** Subtracting their loop direction from 64 columns of L15/L19 `mlp.down_proj` (131,072 weights):
   - takes MATH-500 from 7.7 to 50.9 and GSM8K from 13.8 to 67.3;
   - cuts the share of answers stuck at the 32k limit from 93% to 32% (MATH-500) and from 88% to 7% (GSM8K).

   That matches their 7.9 → 48.1 and 13.8 → 67.1.
2. **Base and p50c GSM8K agree within noise.** p50c MATH-500 is 2.75 points higher (about 2.8 SE):
   - the grader is identical, so the difference comes from generation;
   - the most likely cause is a different vLLM version or GPU kernels (their version isn't published), which changes which sampled trajectories loop;
   - our cut share is slightly lower too (32.2% vs 33.1%).

   It doesn't change any conclusion.
3. **The eval pipeline is validated for R.2.** A Fable-made edit can be scored on exactly the original's terms, and compared with their GPT-6 Astra runs.

## Surprises or problems
- **One shard failed at startup** (p50c MATH-500 6/10). Two vLLM processes started on the same GPU within a minute. The second one's memory profiling counted the first one's memory, which left −1.94 GiB for the KV cache. It was rerun alone and succeeded. **Fix for later:** stagger process starts.
- **Two processes per A40 was slower than one:** about 800 tokens/s each (1,600 per GPU), against 2,295 tokens/s for a single process on the first shard.
- **It took much longer than the 6–9 h estimate,** about 18 h wall clock. The base model writes about 30k tokens per answer, since 93% of its answers loop to the limit. The base model alone generated 139M tokens.

## Implication for R.2 compute
- One full practice pass of 512 base-model answers is about 15M tokens. That's roughly 2.6 h on an A40 at our rate, against their roughly 30 min.
- An R.2 session on A40s would therefore take days. **R.2 needs an H100/H200-class GPU,** as `exp_r/R2_design.md` already assumes. Measure its throughput on a single shard before starting a session.

## Cost
- **Pod `8aum42i0zdvy7r` (wst-r1a):** A40 at $0.59/h, uptime 66,911 s (18.6 h), about $10.97. It also covered setup: download, edit rebuild, bench prep and the first shard.
- **Pod `s3vxa294lkcnt0` (wst-r1b):** A40 at $0.59/h, uptime 64,110 s (17.8 h), about $10.51.
- **Total about $21.50.** Both pods are stopped (EXITED, verified).
- No driver or API cost.

## Files
- `exp_r/r1_eval.py`, `exp_r/r1_jobs.sh`, `exp_r/r1_queues.txt`, `exp_r/r1_setup.sh`, `exp_r/r1_compare.py`; vendored MIT code is in `exp_r/vendor/`.
- Results (local, gitignored): `artifacts/r1/eval/{base,p50c}/` (per shard plus `merged.json`), `artifacts/r1/compare_{base,p50c}.json`, `artifacts/r1/logs/`.

## Next step
R.2: Fable on their exact task and harness (`exp_r/R2_design.md`), starting with the 10-pass budget on an H100.
