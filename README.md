# Claude, Please Fix My Model

Repairing a fine-tuned LLM by editing its weights directly, with no gradients and no training data.

This is a quick, preliminary replication of "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2026, [blog](https://rikiyat.github.io/blog/weight-space-programming/)). It uses a small open model and a public benchmark.

## The experiment (Experiment A)
1. **The target:** Qwen3.5-2B used as a tool router.
2. **Breaking it:** a LoRA fine-tune on 58,000 xLAM function-calling examples, merged into the model.
   - BFCL v4 call accuracy rises from **68.4 to 84.6**.
   - BFCL irrelevance, meaning declining when no tool fits, falls from **78.8 to 0.8**. The model calls a tool for every prompt.
3. **The repair:** Claude Fable 5.1, the *driver*, runs inside a harness (`common/harness/`).
   - It can run Python against the broken model's weights and spend a fixed budget of GPU forward passes on practice prompts. It then saves and submits.
   - The test set is never visible to it, and it has no gradients or training data.
4. **The edit it found** (budget 25) changes one row of the tied embedding, 2,041 weights:
   ```
   E["I"] ← 1.2·E["I"] + 0.30·E[<tool_call>]
   ```
   The token "I", as in "I don't have a tool for that", now competes with `<tool_call>` on the first generated token.

## Results (BFCL v4 test split)
| run | irrelevance | mean call acc | recovery | retention (pts) | weights changed |
|---|---|---|---|---|---|
| base | 78.75 | 68.40 | 1.000 | — | — |
| broken | 0.83 | 84.62 | 0.000 | — | 58% vs base |
| Fable, budget 10 | 82.08 | 69.32 | 1.043 | −15.31 | 4,096 |
| **Fable, budget 25** | 64.58 | 83.49 | **0.818** | **−1.13** | **2,041** |
| Fable, budget 50 | 67.92 | 74.59 | 0.861 | −10.04 | 0.90% |
| steering ablation | 0.83 | 83.46 | 0.000 | −1.16 | 33% |
| corrective LoRA | 82.50 | 82.73 | 1.048 | −1.89 | 52.8% |

- **Recovery** is the fraction of lost irrelevance won back.
- **Retention** is the change in call accuracy vs the broken model. The target is ≥ −2.

The full table, with costs, is in `artifacts/a/figures/results.md`. The figures are in `artifacts/a/figures/`.

**Caveats:**
- There is **one driver session per budget**. Variance between runs is real.
- The corrective LoRA wins on score, but it learned from the base model's outputs and changes 1.17B weights.
- BFCL irrelevance checks only that the model declined to call a tool. It doesn't check answer quality, and the repaired model's prose is sometimes repetitive.

## Repo layout
| path | what |
|---|---|
| `PROJECT_PLAN.md` | the full plan: steps, gates, and Experiments B and C (not yet run) |
| `reports/` | one report per step (0.1 to A.9), with the numbers, the problems hit, and the cost |
| `common/harness/` | the driver harness: sandboxed kernel, forward-pass budget, test-set guard, and MCP tools |
| `common/evals/` | BFCL v4 wrapper for Qwen3.5 |
| `common/train/lora.py` | the LoRA training used to break the model and for the corrective baseline |
| `exp_a/` | data prep, steering baseline, corrective data, the driver task prompt, session notes, and the live demo |
| `scripts/` | scoring, the weights-touched diff, figures, and the session timeline export |
| `env/lock.md` | pinned versions and model revisions |
| `talk/` | AI Tinkerers talk outline |

## Running it
- **GPU:** one 48 GB GPU (an A40 on RunPod here). Environment setup is in `scripts/setup_pod.sh`, versions in `env/lock.md`.
- **Driver sessions** run locally with Claude Code: `python3 -m common.harness.driver`.
  - Point it at your pod with `WST_POD_HOST`, `WST_POD_PORT` and `WST_POD_KEY`.
- **Live demo:** to serve base, broken, repaired and LoRA side by side, see `exp_a/demo.md`.

Model checkpoints are not in the repo. They're about 4.3 GB each.
