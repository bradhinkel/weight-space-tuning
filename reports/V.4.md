# V.4: the grid, 3 runs at each of 10, 25 and 50 passes, in both arms

## What was done
- **The runs:** 16 sessions on the laptop, two at a time (one queue per arm), from 08:03Z to 16:38Z on 2026-11-22, 8.6 h in all. With the two V.3 pilots as run 1 at budget 25, that gives 18 runs: 3 per budget per arm.
- **Setup:** Fable 5.2 at max effort. The task template and harness are unchanged from V.3. The configs and runner are in commit 43f18a5.
- **Outcome:** all 16 runs ended with rc=0. There were no nudges, no compactions and no usage-limit waits.
- **Per run:** 47–92 min, mean 63 min. Mean $23.8 notional, $380 for the 16 grid runs.
- **Usage:** the weekly meter went from 7% to 47% while the grid ran. The R.4a repeats ran in the user's other session at the same time, so the 40 points are shared between the two. The grid's share is not separable.
- **Records:** `artifacts/v/grid/<tag>/` holds `final.json`, `driver_summary.json`, `calls.jsonl`, `state.json` and `submitted/`. Agreement and bootstrap results are in `artifacts/v/grid/v4_agreement.json`, produced by `exp_v/analysis/v4_agreement.py`.

## Results: Waterbirds test, worst-group accuracy
| arm | 10 passes | 25 passes | 50 passes | all 9 runs |
|---|---|---|---|---|
| no fitting | 91.74 · 92.21 · 92.73 | 93.46 · 92.68 · 92.21 | 92.90 · 92.86 · 92.37 | **92.57 ± 0.50** |
| mean ± SD | 92.23 ± 0.50 | 92.78 ± 0.63 | 92.71 ± 0.30 | |
| fitting allowed | 93.61 · 93.22 · 92.52 | 93.30 · 93.15 · 93.30 | 93.61 · 93.15 · 93.61 | **93.27 ± 0.34** |
| mean ± SD | 93.12 ± 0.55 | 93.25 ± 0.09 | 93.46 ± 0.27 | |
| DFR, our 5 seeds | | | | 92.58 ± 0.44 (91.80–92.86) |
| broken (ERM) | | | | 75.39 |

- **Weighted mean:** 95.9–96.9 in all 18 runs, against DFR's 95.17 ± 0.39 and ERM's 98.3.
  - No run lost more than 2.4 points of weighted mean.
  - Every run beat DFR on weighted mean.

## Against DFR: the engineering goal
**The goal is a repaired model that beats DFR. The fitting arm reached it, and it did so repeatedly.**
- **Above every DFR seed:** 8 of 9 fitting-arm runs score higher on worst group than the best of the 5 DFR seeds (92.86). The ninth scores 92.52.
- **Arm mean:**
  - The fitting-arm mean is 0.70 points above the DFR mean.
  - Welch t = 3.1 over 9 runs against 5 seeds.
  - On this test set, the gain holds up across runs and is not one lucky session.
- **Best runs:** three runs (b10 r1, b50 r1, b50 r3) reach 93.61, 1.03 above DFR's mean, at the same weighted mean or better.

The no-fitting arm matches DFR (92.57 against 92.58) but does not beat it. Its best run is the V.3 pilot (93.46), and only 2 of its 9 runs reach DFR's best seed. The pilot made this arm look better than it is.

**The caveat on "beats":**
- **Sampling error of the test set:** the test set is finite, and worst group is usually set by a 642-image waterbird group (or the landbird/water group).
- **Single-run comparison:** in a paired bootstrap of one run against DFR seed 0 on the same test images, the best run is +0.93 with a 95% CI of −1.1 to +1.7. No single run is a significant improvement on its own.
- **What does hold:** the arm-level result is consistent across 9 runs on this split.
- **What it would take to say "better on new data":** a larger test set, or a second dataset.

## Same fix or not?
Every one of the 18 runs reached the same diagnosis:
- **Shortcut size:** the bird moves the margin about +16, and the background about +8.1.
- **Where it arises:** bird and background directions start nearly orthogonal at layer3 (cosine 0.04). They become entangled in the last two blocks (0.30, then 0.57 at avgpool).
- **No background channels:** the effect is a distributed direction. At most a few hundred channels carry it, and the same channels carry bird evidence.
- **Same repair direction:** every edit removes the water-minus-land mean direction, measured within bird class. Some add a second background direction, such as a water principal component, a saturation axis, or the skip-path copy of the direction.

The runs differ in **where** the direction is removed:

| group | where | runs | n |
|---|---|---|---|
| A | fc only (projection, or a closed-form "blind to background" readout) | V.3b, b10 fit r1/r2, b10 nofit r1/r3, b25 fit r2, b25 nofit r2/r3, b50 fit r1/r3, b50 nofit r3 | 11 |
| B | upstream reader cut in layer4 plus fc | V.3, b10 fit r3, b10 nofit r2, b50 fit r2 | 4 |
| C | upstream only, at layer4.0 and/or layer4.2 (fc weights untouched) | b25 fit r3, b50 nofit r1/r2 | 3 |

- **One unusual edit:** b50 fit r2 built a periphery mask inside `layer4.2`. It turned one channel into a constant and another into a border indicator, so that the block's output is zero on the outer ring of the 7×7 map. This is a spatial fix, not a directional one, and it is the only one in the grid.
- **Site and score:** the site of the edit does not predict the score.
  - Group A: 91.74–93.61.
  - Group B: 92.21–93.46.
  - Group C: 92.86–93.30.
  - The fitting arm is ahead in every group.

**Same behaviour on test images:**

| pairs | n | agreement on 5,794 test images |
|---|---|---|
| within group A | 55 | 97.8–99.5 (mean 98.4) |
| A vs B | 44 | 97.5–98.8 (98.2) |
| A vs C | 33 | 97.3–98.2 (97.8) |
| all 153 run pairs | 153 | min 97.3, mean 98.3 |
| each run vs DFR seed 0 | 18 | 95.9–97.4 (96.5) |
| each run vs ERM | 18 | 92.4–93.7 (92.9) |

- **The runs agree with each other more than with DFR.** Fable's repairs form a family of their own: different weights, nearly the same decisions.
- **What fitting changes:** it doesn't change the mechanism. It changes how the read-out is placed:
  - fitting runs whiten the direction with within-group covariance, or fit a ridge background detector;
  - no-fitting runs use plain mean differences and set the bias from a balancing rule.

## Practice vs test
- **Practice was pessimistic:** test was higher than practice (the final pass, in-sample) in 16 of 18 runs, by 1.2 points on average. Fable's own forecasts were also conservative.
- **Practice does not rank the runs:** the correlation between practice and test worst group over the 18 runs is 0.32.
  - The best practice run, b10 fit r3 at 93.1, has the lowest fitting-arm test score (92.5).
  - The practice groups have 133 images each, about ±2.5 points, which swamps the 0.3–0.5 point differences between runs.
- **This matters for V.7.** A synthesis that picks or weights runs by practice score is picking on noise.

## Rule audit (all 16 grid runs)
- **Gradients, autograd and optimizers:** none in any run.
- **Network simulation:** no convolution, pooling or batch-norm computed in the kernel in any run.
- **No-fitting arm:**
  - **Regression, solves and scikit-learn:** the scan found none, apart from two kinds of hit:
    - each run's setup line that denies scikit-learn;
    - one descriptive R² with `lstsq` in b50 nofit r1 (call 64), the same use as in V.3.
  - **Weights from statistics:** weights came only from group means, covariances and principal components.
  - **Bias rules:** biases were set by rule:
    - the group-balanced centre;
    - log(0.7679/0.2321), the class prior from the stated training proportions;
    - zero mean on bird-free scenes.
- **A borderline point:** b50 nofit r1 set one constant (3.554, at layer4.2's reader) by interpolating between two measured settings, so that the two class means sit symmetric about zero.
  - Fable flagged this itself and asked to have it judged.
  - It is a one-parameter use of output statistics, the same kind as V.3's bias midpoint. I count it as allowed as the rules are written. The rules ladder should name it.
- **Fitting arm:** 5–19 calls per run with fits: ridge or logistic background probes, and closed-form whitened directions. All were on saved activations, which this arm allows.

## Predictions, scored
| # | prediction | result | verdict |
|---|---|---|---|
| 8 | mean rises with budget, levels off by 100 (P = 0.6) | budgets 100 and 200 were not run. From 10 to 50 the mean moves +0.5 (no fitting) and +0.3 (fitting), within one SD. It is level from 10 on. | not testable as written; level already by 10 |
| 9 | SD at budgets 10 and 25 is 3–10 points (P = 0.65) | 0.09–0.63 | ✗ (far smaller) |
| 10 | spread shrinks with budget, SD at 100 < SD at 10 (P = 0.6) | at 50 against 10: 0.30 < 0.50 and 0.27 < 0.55; no budget-100 runs | not testable as written; direction holds at 50 |
| 11 | at least 2 mechanism groups among the budget-50 runs (P = 0.7) | 3 groups among 6 runs (A ×3, B ×1, C ×2). All remove the same direction; they differ in site | ✓ (by site) |
| 12 | most common fix: a last-layer change that down-weights background (P = 0.6) | 11 of 18 are fc-only | ✓ |
| 13 | at least half of different-mechanism pairs agree on ≥ 90% of test (P = 0.5) | every pair agrees on ≥ 97.3% | ✓ |
| 17 | fitting runs reach DFR − 1 at b25 (P = 0.75) | 93.30 / 93.15 / 93.30, all above DFR | ✓ |
| 18 | at least one fitting run beats DFR by ≥ 1 (P = 0.35) | 3 runs at +1.03 | ✓ (just; within test-set noise for a single run) |

- **Calibration:** I expected far more spread than there was (9 ✗). Fable's repair of this network is very repeatable.

## What it means
1. **Allowed to fit, Fable beats DFR, repeatably.** The fitting arm averages 93.3, 0.7 above DFR, with 8 of 9 runs above every DFR seed. It does this with an edit it reasoned out, usually 4,098 fc weights, and at a better weighted mean.
2. **Without fitting, Fable matches DFR.** Pure reasoning plus activation statistics gets to DFR's level in every run at 10 passes or more, but not beyond it.
3. **Budget barely matters.** Ten passes already gets the result. More passes buy longer causal checks (dose-response, controls, held-out folds), not a better edit. This is cheaper than the R.4a router, where budget did matter.
4. **One mechanism, several sites.** All 18 runs found the same shortcut and removed the same direction. They differ in where they cut, and the predictions barely differ (98% agreement). This is a fact about the network, not about one session.
5. **For V.7:**
   - **Practice scores can't rank the runs.**
   - **The runs make nearly the same predictions.** Averaging their outputs or votes has little to gain from diversity.
   - **What a synthesis could add:**
     - combining the fitting arm's whitened read-out with the upstream cuts (groups B and C);
     - using the label-noise finding (the flycatcher species);
     - using all 1,199 practice images for every statistic.
   - **Expected gain:** realistically under a point, which is below what this test set can resolve for one model.

## Surprises or problems
- **Spread:** much smaller than predicted (SD under 0.65 against a predicted 3–10).
- **Pilot:** the no-fitting pilot was the arm's best run, which overstated the arm in V.3.
- **Late commits:** run records were committed only after 14 runs had finished (0a385ad), not as each finished. No data was lost.

## Addendum (2026-11-22): DFR with 20 seeds
- **Why:** building V.7's DFR files showed that re-running DFR is not bit-exact. GPU feature extraction and liblinear's solver both vary slightly between runs, so a seed can select a different C.
  - Seed 4 picked C = 0.7 and scored 93.4, against V.1's C = 0.1 and 91.8.
  - I therefore re-ran the V.1 recipe with 20 seeds (`exp_v/analysis/dfr_seeds.py`, results in `artifacts/v/broken/dfr_20seeds.json`).
- **Result:**
  - worst group 92.58 ± 0.51 (range 91.26–93.39), the same mean as V.1's 5 seeds;
  - weighted mean 94.92 ± 0.44;
  - C was chosen as 1.0 ×5, 0.7 ×8, 0.3 ×4, 0.1 ×2 and 0.07 ×1. DFR's own tuning step is a source of its spread.
- **What changes in this report:**
  - **"Above every DFR seed" no longer holds as stated.** Against 20 seeds, 3 of 9 fitting-arm runs (the three at 93.61) are above DFR's best seed (93.39).
  - 8 of 9 are above DFR's mean + 1 SD (93.09).
  - The arm-level comparison gets stronger with the larger baseline: the fitting arm is +0.69 over DFR, Welch t = 4.3 (9 runs against 20 seeds).
  - The no-fitting arm is still equal to DFR (−0.01).
- **What doesn't change:** the per-run paired CI against DFR still spans zero. The conclusion stands: the fitting arm beats DFR consistently on this split, by less than this test set can resolve for any single model.

## Next
- **The user reviewed this report.** Their reading:
  - Fable beats DFR consistently, by less than the CI.
  - A larger dataset is the only way to narrow the CI, and synthesised test images would add a different dimension.
  - The experiment answered its question, "can Fable fix the shortcut by weight-space editing?": yes, with caveats.
- **V.7 is designed** (`exp_v/V_design.md`, predictions 19–25) and set up.
