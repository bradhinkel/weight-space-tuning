# AI Tinkerers Dev track, Dec 2: talk outline (10 min)

**Working title:** *Claude, Please Fix My Model: Repairing an LLM by Editing Its Weights Directly*

## Pitch (about 120 words)
Small open models make cheap tool routers, and Qwen3.5-2B tests adequately in that role out of the box. To make it better, I fine-tuned it on 58,000 function-calling examples. Tool-call accuracy rose from 68% to 85%, but the router began recommending a tool for every prompt; ask for a cat joke and it checks the weather. Instead of retraining, I replicated MIT CSAIL's new weight-space programming experiment: Claude Fable 5.1 got the model's weights and a Python sandbox, with no gradients and no training data, and edited the weights directly. It changed a single embedding row, 2,041 weights, and recovered 82% of the router's ability to recognise irrelevant prompts while keeping its call accuracy. The catch: it learned to say "I can't help", not to answer. I'll show what happened when I asked for real answers instead. Live demo: four models side by side.

### Short version (50 words)
Can a frontier model repair a broken LLM by reading its weights? I fine-tuned Qwen3.5-2B until it called tools on everything, then gave Claude Fable 5.1 the weights and Python, with no gradients or training data. It changed one embedding row, 2,041 weights, and recovered 82% of the lost behavior, mostly by learning to refuse.

## Shape
About 7 minutes of talk, a 2-minute live demo, and 1 minute of caveats and asks. Seven or eight slides. The demo runs from the four vLLM servers in `exp_a/demo.md`; screenshots from `exp_a/demo_live_check.txt` are the fallback.

## Outline

### 1. Hook (0:00–0:45)
- **Slide:** "I broke a model on purpose, then asked Claude to fix it without training."
- **Show:** the broken model asked for a cat joke calls `get_weather(London)` and `convert_currency(100 USD→EUR)`.
- **Line:** "That's what too much tool-calling fine-tuning does. Today: can a frontier model debug the weights directly?"

### 2. Where this comes from (0:45–1:45)
- **The MIT CSAIL result:** "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, Nov 2026) showed frontier models can make targeted weight edits by reasoning over the weights.
- **My version:**
  - a model people here actually use (Qwen3.5-2B)
  - a public benchmark (BFCL v4)
  - a realistic failure (over-calling after a fine-tune)
  - a gradient baseline to compare against
- **Question:** does it replicate on standard tools, and how does it compare to the obvious fix?

### 3. The setup (1:45–3:00)
- **Breaking the model:** LoRA fine-tune on 58k xLAM function-calling examples.
  - Call accuracy goes **68 → 85**.
  - Irrelevance, meaning declining when no tool fits, goes **79 → 1**.
  - 58% of the weights change.
- **The driver:** Claude Fable 5.1 in a harness.
  - It can load the weights, run Python against them, and spend a fixed budget of GPU "practice passes" on held-out practice prompts.
  - It edits, saves and submits.
  - It never sees the test set or any gradients.
- **The scoreboard:**
  - *recovery*: the fraction of lost irrelevance won back
  - *retention*: how much call accuracy is kept
  - target: retention ≥ −2 points

### 4. What the driver did (3:00–4:45): the story slide
- **Show:** `timeline_b25.png`, one 33-minute session at a budget of 25 passes, of which 20 were used.
- **The arc:**
  1. Baseline.
  2. Scaling the `<tool_call>` embedding did nothing.
  3. Found the "format-copy" attention heads. Scaling them cut calls but hurt real calls.
  4. Tried MLP and value edits, with no effect.
  5. Changed tack: boost the token "I", the start of "I don't have a tool for that…".
  6. Final edit: **`E["I"] ← 1.2·E["I"] + 0.30·E[<tool_call>]`**.
- **Why it works:** "I" now competes with `<tool_call>` on the first token. It's a one-line, readable edit.
- **Show:** `touched_heatmap.png`. One lit cell for the driver, every layer lit for the LoRA.

### 5. Results (4:45–6:15)
- **Show:** `irrelevance_vs_call.png` and the results table.

| | irrelevance | call acc | recovery | weights changed |
|---|---|---|---|---|
| base | 78.8 | 68.4 | — | — |
| broken | 0.8 | 84.6 | 0 | 58% vs base |
| **Fable, budget 25** | 64.6 | 83.5 | **0.82** | **2,041** |
| corrective LoRA | 82.5 | 82.7 | **1.05** | 1.17 B |
| steering ablation | 0.8 | 83.5 | 0.00 | 33% |

- **The honest headline:** gradients win on score. The LoRA also had more to work with: base-model targets and 6,000 examples. The driver had only the broken model and practice prompts, and it produced an edit you can read.
- **Steering ablation scored 0:** the "calling" direction is present in relevant *and* irrelevant prompts, so removing it removes nothing useful.

### 6. Live demo (6:15–8:15)
1. **Cat joke** (weather and currency tools offered): broken calls both tools; base, Fable and LoRA answer in text.
2. **"Weather in Paris?":** all four call `get_weather(Paris)`, so real calls still work.
3. **One remaining over-call** (items 0–5 or 9, for example the quadratic equation offered `math.sum`): Fable still calls the tool, the LoRA doesn't.
4. **Point out:** Fable's cat answer slides into repetition. The edit fixed *whether* it answers, not *how well*.

### 6b. Round 2: "decline" isn't "answer" (fold into 7 if short on time)
- **Show:** the A.10 answer-quality table. Good answers out of 240: base 81, LoRA 75, **Fable b25 4**. Two thirds of Fable's answers are a bare "I can only do X".
- **The rerun with a stricter goal:**
  - Same tools: Fable declined to submit anything.
  - Adding activations and the base model: Fable built a **probe-gated MLP neuron** (6,141 weights). Irrelevance 83 (above base), but call accuracy fell 19 points.
- **The twist:** the probe learned my practice data's *style* (real-user vs synthetic prompts), not relevance. It stopped calling tools on real users. That was my bug, not the driver's, and the driver's own 97% probe validation couldn't see it.
- **Line:** "The driver is only as good as the practice data you give it."

### 7. Caveats (8:15–9:15)
- **One session per budget:**
  - budget 10 over-corrected (call accuracy fell back to base)
  - budget 50 broke multi-call answers
  - only budget 25 hit the target
  - The variance is real. (Show `budget_curve.png`.)
- **BFCL irrelevance only checks "didn't call."** When I scored answer quality separately, the b25 fix mostly restored refusals (slide 6b).
- **About $13 of notional driver tokens and $0.33 of GPU** for the winning session. The LoRA cost $0.45 of GPU.

### 8. Takeaways and asks (9:15–10:00)
- Frontier models can already do **targeted, explainable** weight surgery on a small model. That's a debugging tool, not a training replacement (yet).
- **Next:** practice data without the style confound, an answer-quality signal for the driver, and repeat runs.
- Code, harness and timeline JSON are available. Happy to talk harness design.

## Slide-to-asset map
| slide | asset |
|---|---|
| 4 | `artifacts/a/figures/timeline_b25.png`, `touched_heatmap.png` |
| 5 | `artifacts/a/figures/irrelevance_vs_call.png`, `results.md` |
| 6 | `exp_a/demo.md`, `exp_a/demo_live_check.txt` (fallback) |
| 6b | `reports/A.10.md`, `reports/A.11.md` tables |
| 7 | `artifacts/a/figures/budget_curve.png` |

## Demo-day checklist
- Start the pod about 15 minutes before, then run `bash exp_a/demo_serve.sh`. The four servers take about 8 minutes.
- Forward ports 8101–8104 to the laptop and run `demo_query.py --host localhost`.
- Keep screenshots of every demo step in the deck in case the Wi-Fi fails.
- Stop the servers and pause the pod afterwards.
