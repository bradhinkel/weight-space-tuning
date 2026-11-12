# Environment lock

Recorded 2026-11-11 (UTC) from the pod with `importlib.metadata` and `env/revisions.json` (written by `scripts/download.py`).

## Hardware
| item | value |
|---|---|
| GPU | NVIDIA A40 48 GB (RunPod Secure Cloud, CA-MTL-1, pod `gr4byo6hup4lol`). The plan's class is A6000/L40S. L40 had no stock at launch; A40 is the same 48 GB Ampere class as an A6000 |
| Driver / host CUDA | 595.91.07 / 13.2 |
| Host RAM | 503 GB |
| Storage | network volume `wst-workspace` (`61gtyoubem`), 200 GB, mounted at `/workspace` |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (system Python 3.12.3) |

## Main venv (`/workspace/venv`): training, vLLM serving, harness
| package | version |
|---|---|
| torch | 2.14.0 (CUDA 13.0) |
| transformers | 5.20.2 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| datasets | 5.2.0 |
| vllm | 0.33.0 |
| anthropic | 2.8.0 (installed per 0.1; unused, since all driver billing goes through Claude Code on Max) |
| mcp | 2.3.0 |
| flash-linear-attention | 0.6.0 |
| safetensors | 0.8.0 |
| numpy | 2.3.5 |
| huggingface-hub | 2.4.0 |
| tokenizers | 0.23.3 |
| causal-conv1d | **not installed**: build failed (toolkit CUDA 12.8 vs torch CUDA 13.0). HF uses the slower reference PyTorch path, which is numerically equivalent |

## BFCL venv (`/workspace/bfcl-venv`): isolated because bfcl-eval pins conflict with vLLM
| package | version |
|---|---|
| bfcl-eval | 2026.3.23 |
| transformers | 5.22.2 (tokenizer and chat template only) |
| qwen-agent | 0.0.34 |
| openai | 3.49.0 |

## Model and data revisions (HF commit hashes)
| repo | type | revision |
|---|---|---|
| Qwen/Qwen3.5-2B | model | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Qwen/Qwen3.5-0.8B | model | `2fc06364715b967f1860aea9cf38778875588b17` |
| Salesforce/xlam-function-calling-60k | dataset | `26d14ebfe18b1f7b524bd39b404b50af5dc97866` |
| gorilla-llm/Berkeley-Function-Calling-Leaderboard | dataset | `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda` (eval data used is the copy bundled in bfcl-eval 2026.3.23) |

## Seeds and decoding
- LoRA and data split: seed 0 (xLAM practice split = first 2,000 of `random.Random(0).shuffle(range(60000))`).
- vLLM `--seed 0`. All evals are greedy (temperature 0) with thinking off (`enable_thinking=False`).

## Driver
| item | value |
|---|---|
| Claude Code | 2.1.322 (workstation, Max subscription OAuth; `ANTHROPIC_API_KEY` unset) |
| Driver model ids | `fable-5.1` → `claude-fable-5-1`, `opus-5.5` → `claude-opus-5-5` (`common/harness/driver.py`) |
