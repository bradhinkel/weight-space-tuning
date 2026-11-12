# Experiment A demo: four checkpoints side by side

Shows the base model, the broken model (xLAM fine-tune: calls a tool on everything), the driver's repair (Fable 5.1, budget 25, one embedding row changed) and the corrective-LoRA baseline. All four are served by vLLM on one GPU.

## Checkpoints (network volume `61gtyoubem`, mounted at `/workspace`)
| name | port | path | what it is |
|---|---|---|---|
| base | 8101 | `Qwen/Qwen3.5-2B` (HF cache `/workspace/hf`) | the unmodified model |
| broken | 8102 | `/workspace/artifacts/a/demo/broken` → `../broken` | LoRA fine-tune on 58k xLAM examples, merged (A.2) |
| fable_b25 | 8103 | `/workspace/artifacts/a/demo/fable_b25` (a copy of the b25 session's `final_checkpoint`) | broken + `E["I"] ← 1.2·E["I"] + 0.30·E[<tool_call>]`, 2,041 weights (A.5) |
| lora | 8104 | `/workspace/artifacts/a/demo/lora` → `../arm3` | broken + corrective LoRA, merged (A.7) |

Each checkpoint is about 4.3 GB of bf16 safetensors and includes its tokenizer and chat template. Together the four servers use about 4 × 0.2 of a 48 GB card.

## Launch
1. **Start the pod.** Resume `wst-exp-a` from the RunPod console, or create a new pod (A40/L40S/A6000, 48 GB) with the network volume `61gtyoubem` attached. A new pod gets a new SSH port.
2. **Start the four servers.** `ssh` in, then:
   ```bash
   cd /workspace/repo
   bash exp_a/demo_serve.sh   # one server at a time, ~2 min each; prints "<name> ready on :<port>"
   ```
   They start one at a time on purpose. vLLM's start-up memory profiling fails if two servers load at once. Logs go to `/workspace/logs/demo_<name>.log`.
3. **Query them.** Send the same prompt to all four:
   ```bash
   source /workspace/venv/bin/activate
   python exp_a/demo_query.py --item 2                       # one of the ten prompts below
   python exp_a/demo_query.py --q "What is the weather in Paris right now?" --tools exp_a/demo_tools.json
   python exp_a/demo_query.py --q "Tell me a joke about cats." --tools exp_a/demo_tools.json
   ```
   Replies are greedy, with thinking off, and shown raw, so a `<tool_call>` is visible. Each reply is labelled "CALLS A TOOL" or "answers in text".
4. **Use a laptop browser or client instead (optional).** Forward the ports with `ssh -N -L 8101:localhost:8101 -L 8102:localhost:8102 -L 8103:localhost:8103 -L 8104:localhost:8104 -p <port> root@<ip>`. Then run `python exp_a/demo_query.py --host localhost ...` locally; it needs no dependencies.
5. **Stop.** Run `bash exp_a/demo_serve.sh stop`, then pause the pod.

## Suggested demo flow
1. **Joke about cats with weather/currency tools.** Broken calls both tools (`get_weather(London)` and a USD→EUR conversion). base, fable_b25 and lora answer in text; fable_b25's answer degrades into repetition.
2. **Weather in Paris with the same tools.** All four call `get_weather`. The repair didn't break real calls.
3. **The ten prompts below (use 0–5 and 9 live).** The repair isn't perfect. Here fable_b25 still calls a tool, base doesn't, and the LoRA usually doesn't.
4. **The edit itself.** Show the one-line edit, then `artifacts/a/figures/touched_heatmap.png` next to the LoRA's 1.17B changed weights, then `timeline_b25.png` for how the driver got there.

## Ten prompts where the repaired model still over-calls
These come from the BFCL v4 `irrelevance` test split. They are the first 10 items, by id, where fable_b25 emits `<tool_call>` and base doesn't, with one item per offered function. Run them with `--item k`. The recorded outputs from the A.1, A.2, A.5 and A.7 evals are in `exp_a/demo_prompts.json`.

The base, broken, fable_b25 and lora columns show what each model did in the recorded eval: "call" means it emitted a tool call, "text" means it answered in prose.

| k | prompt | offered tool | base | broken | fable_b25 | lora | fable_b25 live (vLLM) | fable_b25's call (eval) |
|---|---|---|---|---|---|---|---|---|
| 0 | Solve the quadratic equation with coefficients a = 1, b = 2, and c = 3. | `math.sum` | text | call | call | text | call | `math.sum(numbers=[1, 2, 3])` |
| 1 | What is the slope of the line which is perpendicular to the line with the equation y = 3x + 2? | `find_critical_points` | text | call | call | text | call | `find_critical_points(function="lambda x: 3 * x + 2", …)` |
| 2 | Find the definite integral of f(x)=x^2 from x=1 to x=3. | `CalculateTax` | text | call | call | text | call | `CalculateTax(annual_income=100000, …)` |
| 3 | What is the closest integer to 30? | `get_closest_prime` | text | call | call | call | call | `get_closest_prime(number=30)` |
| 4 | What is the acceleration a ball will reach if it's thrown straight upwards with a velocity of 5 m/s? | `calculate_maximum_height` | text | call | call | call | call | `calculate_maximum_height(initial_velocity=5)` |
| 5 | How far will a car travel in time 't' when launched with velocity 'v' at an angle 'theta'? | `calculate_projectile_range` | text | call | call | text | call | `calculate_projectile_range(initial_velocity="v", …)` |
| 6 | What's the time right now? | `calculate_time` | text | call | call | text | **text** | `calculate_time(distance=1000, speed=10)` |
| 7 | How do I find the angle of the force for a given momentum? | `calculate_vector_angle` | text | call | call | text | **text** | `calculate_vector_angle(X_component="momentum_x", …)` |
| 8 | Find the volume of a cone with base radius 3 cm and height 5 cm. | `investment_calculator.calculate_return` | text | call | call | text | **text** | `investment_calculator.calculate_return(initial_investment=100, …)` |
| 9 | What is the magnetic field at a point located at distance 'r' from a wire carrying current 'I'? | `magnetic_field_intensity` | text | call | call | text | call | `magnetic_field_intensity(current="I", distance="r")` |

**The pattern:** the remaining over-calls are mostly cases where the offered tool sits in the right *domain*, such as a math question with a math tool or physics with physics. The edit only flips the decision when "I" is close to `<tool_call>` on the first token. Some are absurd anyway (items 2, 6 and 8). In all, fable_b25 still calls a tool on 85 of 240 test irrelevance prompts, and on 56 of them base doesn't.

**Live check (2026-11-12, `exp_a/demo_live_check.txt`).** The base, broken and lora decisions match the recorded eval on all 10 items. fable_b25 matches on 7. Items 6, 7 and 8, the absurd tools, flip to text when served live. The BFCL harness renders tools with its own prompt, and the decision is marginal on those three. **For the "still over-calls" segment, use items 0–5 and 9.**

**Other live observations:**
- On "Tell me a joke about cats," broken calls `get_weather(London)` *and* `convert_currency(100 USD→EUR)`, which makes a good opener.
- fable_b25 declines the cat joke, then falls into a repetitive list ("It is very smart. It is very lazy…"). The b25 edit raises "I" at the first token but does nothing for fluency after that. This is the repetition noted in A.8, so it's worth saying out loud in the talk.
- base and lora tell a real joke.
- All four call `get_weather(city="Paris")` for the weather question.
