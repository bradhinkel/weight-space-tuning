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
• The benchmark checks only that the model *declined* to call a tool. It doesn't check the quality of the answer, and the repaired model's prose sometimes gets repetitive.

Next: repeat runs per budget and score the answers themselves.

What strikes me isn't the score. A frontier model debugged another model's weights, explained what it changed, and the fix fits on one line.

Paper: https://rikiyat.github.io/blog/weight-space-programming/

#LLM #MechanisticInterpretability #AIEngineering #Claude #FineTuning
