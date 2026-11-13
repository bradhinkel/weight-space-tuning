# LinkedIn post draft

Can Claude fix a broken LLM by editing its weights directly, with no training?

This month MIT CSAIL published "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab). It shows frontier models can make targeted weight edits by reasoning about a model's internals. I ran a quick, preliminary replication on a model people actually deploy.

The setup:
🔧 I fine-tuned Qwen3.5-2B on 58k function-calling examples. Tool-call accuracy rose from 68 to 85, but the model started calling tools for everything. Ask it for a cat joke and it checks the weather in London. Its BFCL "irrelevance" score fell from 79 to 1.
🤖 I gave Claude Fable 5.1 the weights, a Python sandbox and a small budget of GPU test passes. It had no gradients, no training data and no access to the test set.

The result:
✅ The repair changed one row of the embedding matrix, 2,041 weights, with a single readable line: boost the token "I" (as in "I don't have a tool for that") and tie it to the tool-call token.
✅ It recovered 82% of the lost irrelevance behaviour and kept tool-call accuracy within about 1 point.
📉 For comparison, a corrective LoRA did better (105% recovery), but it changed 1.17 billion weights and learned from the original model's outputs.

The caveats, because it's early:
• I ran one session per budget. The 10- and 50-pass runs were weaker, so variance is real.
• The benchmark checks only that the model *declined* to call a tool. When I had a blind LLM judge score the answers, the repaired model gave a good answer on 4 of 240 prompts, against 81 for the original model. Mostly it had learned to say "I can only do X".

Round 2 asked for good answers, not just "no call":
🔍 With access to the original model and per-layer activations (still closed-form edits, no gradients), Claude rewrote a single MLP neuron into a probe-gated switch. Irrelevance went above the original model's, but tool-call accuracy fell 19 points.
🪤 The cause was my practice data, not Claude's reasoning. Every "irrelevant" example was written by real users and every "relevant" one was synthetic, so the probe partly learned "sounds like a real person". The model stopped calling tools for real users.

Next: fix the practice data, give the driver a way to judge answer quality, and repeat runs.

What strikes me isn't the score. A frontier model debugged another model's weights and explained exactly what it changed. That transparency is also how I found my own data bug.

Paper: https://rikiyat.github.io/blog/weight-space-programming/

#LLM #MechanisticInterpretability #AIEngineering #Claude #FineTuning
