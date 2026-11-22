# V.7: synthesis, one session building on DFR and all 18 earlier runs

## What was done
- **Session:** `v7_synthesis_20261122T181434Z`.
  - Fable 5.2, effort max, budget 50, fitting allowed.
  - A read-only `prior/` holding DFR seeds 0–4 (practice scores marked in-sample) and runs 01–18 (edits, reports, notes, arm, budget). No test numbers.
  - Task text `exp_v/sessions/task_v7.md`; design and predictions in `exp_v/V_design.md` (V.7, predictions 19–25), committed before the run (567944c).
- **Run:**
  - 110 python calls, 50 forward passes, 3 show_images; 0 nudges, 0 compactions.
  - Wall clock 311 min, of which about 210 min was a Fable-limit stop. 18:41Z–22:11Z: "You've hit your Fable limit". The user used a reset, and the driver resumed the same conversation.
  - About 100 min of active work.
- **Cost:** $121 notional, about 5× a grid run, from reading 18 runs' notes and reports. Weekly meter: 50% shortly after the start; the end reading is still to be logged.
- **Records:** `artifacts/v/v7/`: final.json, driver_summary.json, calls.jsonl, notes.md, submitted/, and `v7_analysis.json` (test agreement, bootstrap, rule scan).

## Result: Waterbirds test
| model | landbird/land | landbird/water | waterbird/land | waterbird/water | worst | weighted mean |
|---|---|---|---|---|---|---|
| broken (ERM) | 99.73 | 85.45 | 75.39 | 97.04 | 75.39 | 98.31 |
| DFR, 20 seeds | | | | | 92.58 ± 0.51 | 94.92 ± 0.44 |
| DFR seeds 0–4 given to V.7 | 94.9–95.6 | 92.68–93.44 | 93.61–94.70 | 94.24–95.02 | 92.68–93.44 | |
| fitting arm, 9 runs (V.4) | | | | | 93.27 ± 0.34 (92.52–93.61) | 96.2 avg |
| **V.7 synthesis** | 96.94 | **95.43** | 93.77 | **91.59** | **91.59** | **95.67** |

- **The synthesis did not beat DFR.**
  - It is 1.0 below the DFR mean.
  - It is below all five DFR seeds it was given, and below every one of the 18 earlier runs except one (91.74).
  - Paired bootstrap against DFR seed 0: −1.09 points, 95% CI −3.3 to +1.1.
- **The binding group moved.**
  - V.7 has the highest landbird/water of any run (95.43) and the lowest waterbird/water (91.59).
  - This is a threshold placement, not a new mechanism. V.7 agrees with the 18 runs on 97.3–98.8% of test images, and with DFR on 96.2–96.8%.
- **The weighted mean (95.67) is above DFR's,** as in every run.

## The edit
- **Tensors:** fc only (`fc.weight`, `fc.bias`, 4,098 weights). The in-network edits were tested and rejected.
- **Direction:** the practice waterbird-minus-landbird group-mean direction, with its water-minus-land component removed and scaled so that bird = 16 logits. It was measured on normal and mirrored views. This is the same family as group A in V.4.
- **Two tuned choices on top:**
  - **over-cancel:** subtract a further 10% of the original background read-out;
  - **bias:** the class midpoint, shifted 0.25 logits toward landbird.
  - Both were chosen by cross-validation on practice to protect landbird/water.

## What went wrong
1. **Fable reweighted practice to a test mix it assumed from its own knowledge.**
   - **The assumption:** it identified three "quirk" songbird species labelled waterbird (pewee-like flycatchers, towhees, meadowlarks), about 10 of the 19 practice waterbird errors. It wrote that the test split has "~30 images of each quirk species (~2.3% of each waterbird group each)". That is a claim from its prior knowledge of CUB/Waterbirds; it had no access to test.
   - **What followed:**
     - Fable expected the waterbird groups to be easier on test than on practice, so landbird/water would bind.
     - It moved the operating point toward landbird to protect landbird/water.
     - It forecast 96.8 / 93.4 / 93.2 / 94.0, worst about 93.
   - **On test**, waterbird/water came out at 91.59 and bound. Its own caveat had it about right: "if the test mix is like practice, waterbird/water is about 92.5 instead of 94".
2. **Fable judged DFR on held-out practice and found it about 3.4 points worse (89.1).** On test the same recipe scores 92.6–93.4. Practice's landbird/water group is harder than test's for every model (ERM 82.6 against 85.5), and this hit DFR hardest. The comparison that drove the session's main decision was about 3.5 points off.
3. **More tuning on 133-image groups.**
   - Fable's own notes say that finer tuning was noise, and it stopped optimising for that reason.
   - Even so, the two tuned scalars (over-cancel and bias) each moved expected worst group by only tenths of a point, within practice noise, and the net effect on test was negative.
   - The V.4 runs that used plain cancellation and a midpoint bias did better.

## Rule audit
- **No gradients, autograd or optimizers.** No convolution, pooling or batch-norm in the kernel (all 110 python calls scanned).
- **Fitting:** closed-form read-outs, logistic DFR refits within folds, and fc applied to saved avgpool features. All are allowed in this arm.
- **Prior knowledge of the dataset:** the species-mix assumption came from Fable's own knowledge, not from data in the sandbox. The rules don't forbid it, but it is the decision that went wrong. If the protocol is reused, it should say whether assumptions about the test split are allowed.

## Predictions, scored
| # | prediction | result | verdict |
|---|---|---|---|
| 19 | (user) improves on DFR, but not beyond the CI | −1.0 vs DFR mean; CI −3.3 to +1.1 | ✗ (no improvement; within the CI) |
| 20 | beats DFR mean 92.58 (P = 0.8) | 91.59 | ✗ |
| 21 | within the fitting arm's range 92.52–93.61 (P = 0.65) | 91.59 | ✗ |
| 22 | beats the best grid run 93.61 (P = 0.25) | | ✗ |
| 23 | gain outside the per-run CI (P = 0.1) | | ✗ |
| 24 | builds on DFR's last layer (P = 0.5) | own closed-form read-out; DFR judged worse | ✗ |
| 25 | weighted mean above DFR's 94.92 (P = 0.8) | 95.67 | ✓ |

- **Calibration:** both of us expected a small gain. The CI half of the user's prediction held, in the sense that the difference is within noise in both directions.

## What it means
1. **More information did not give a better model.** Given DFR, 18 independent repairs and 50 passes, Fable produced a model in the same family as the grid runs. It differs mainly in where the threshold sits, and that choice was worse on test.
2. **The bottleneck is measurement, not ideas.** The 18 runs had already found the mechanism, and a synthesis can't see past 133-image practice groups any better than a single run. It had more candidates to choose among, so more room to choose on noise: a winner's-curse effect.
3. **An agent's prior knowledge about the test data is a risk.** The one decision Fable made from outside the evidence, the assumed test species mix, is what moved the result below DFR. This is a useful talk point: the agent was careful and flagged the risk itself, but still acted on the assumption.
4. **The Waterbirds bottom line stands from V.4.**
   - Single fitting-arm sessions beat DFR consistently, by about 0.7, which is less than the test set can resolve for one model.
   - Synthesis did not add to that.
   - The best practice is the simplest recipe: cancel the background direction exactly and centre the bias.

## Records
- **Copied to `artifacts/v/v7/`:** the session records, plus `v7_analysis.json` from `exp_v/analysis/v7_analysis.py`.
- **Full session:** `~/wst-v/sessions/v7_synthesis_20261122T181434Z/`.
