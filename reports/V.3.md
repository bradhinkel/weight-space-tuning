# V.3 GATE: pilot, budget 25, no-fitting arm — PASS

## What was done
- **Session:** `v3_b25_nofit_20261122T044105Z`.
  - Fable 5.2, effort max, budget 25 forward passes, fitting refused (no scikit-learn).
  - The task text is the template in `exp_v/sessions/task_template.md` (commit cc1c3e2).
  - Wall clock 58 min (04:41Z–05:39Z on 2026-11-22).
  - 106 tool calls: 73 python, 26 forward (25 counted, 1 refused) and 7 show_images (101 images).
  - 0 nudges, 0 compactions, 0 usage-limit waits. $25.2 notional.
- **Lock-down:** the init check passed: only the three harness tools, one server, model `claude-fable-5-2`.
- **Scoring:** `exp_v.server finalize` scored the submission on the sealed test split (5,794 images).

## Results: Waterbirds test
| model | landbird/land | landbird/water | waterbird/land | waterbird/water | worst | weighted mean |
|---|---|---|---|---|---|---|
| broken (ERM, V.1) | | | | | 75.4 | 98.3 |
| DFR, our run (V.1, 5 seeds) | | | | | 92.58 ± 0.44 | 95.17 ± 0.39 |
| DFR, published | | | | | 92.9 ± 0.2 | 94.2 |
| **Fable b25, no fitting** | 97.16 | 93.88 | **93.46** | 93.46 | **93.46** | **96.18** |

- **Fable vs DFR:** about 0.9 points above our DFR on worst group and 1.0 above on weighted mean. This is one run; the run-to-run spread is unmeasured.
- **Practice vs test:**
  - practice, in-sample: worst 92.5, mean 95.7;
  - practice, Fable's cross-validated estimate: worst 91.7, mean 95.9;
  - test: 93.46 / 96.18.
  - Practice predicted test within about 1–2 points. Fable's own risk estimate was "a 10–15% chance of missing one target".

## The edit
Six tensors, about 2.1M of the network's 23.5M weights. All directions and centres come from group means and principal components of practice activations.

| site | tensors | change |
|---|---|---|
| 1 | `layer4.1.conv1.weight`, `layer4.1.bn1.running_mean` | the block's first 1×1 conv no longer reads two background directions of layer4.0's output; their component is clamped to its balanced mean (weight change 5.7%) |
| 2 | `layer4.2.conv1.weight`, `layer4.2.bn1.running_mean` | the same for layer4.2, with the directions re-measured after site 1 (5.3%) |
| 3 | `fc.weight`, `fc.bias` | the remaining mean background direction at avgpool removed from both rows; bias set to the midpoint of the four group-mean margins (30%) |

- **The two background directions:**
  - the mean water-minus-land direction, measured within each bird class;
  - the top principal component of water scenes after the mean direction is removed. The classifier reads it with the opposite sign: +0.58 vs −0.51.
- **Fable's diagnosis:**
  - The background is worth about 8 margin units against the bird's 16, with a within-group spread of about 5.
  - The two get mixed in the last two blocks: the cosine between the bird and background directions is 0.04 at layer3, 0.30 at layer4.1 and 0.57 at avgpool.
  - Spatial maps put the background signal at the image border (sky, horizon, water), not on the bird.
- **Causal evidence** (Fable's, on practice):
  - **Dose-response:** the edit at strength 0 / 0.5 / 1 / 1.5 gives a background effect of 8.1 / 3.6 / 0.4 / −2.0. At 1.5 the shortcut reverses.
  - **Frame swaps:** composite images with a land or water frame around the same centre measure the background effect directly: 7.3 before the edit, 0.5 after.
  - **Control:** cutting equally large random non-background directions at the same sites changed nothing (worst group stays 72.9).
  - **Ablations:**
    - the second direction is worth about 3.9 points on landbird/water;
    - an fc-only version of the edit (DFR-like in location, not in method) reached 89.1 / 94.6 on practice.
- **Validation:** Fable chose the edit on a fixed half of practice (DEV) and confirmed it on the other half (HOLD). It then ran the reverse fold and re-derived the final edit from all images with the same recipe.

## Rule audit
I scanned all 73 python calls for fitting and for network simulation, and read the flagged ones.
- **Fitting:** `np.linalg.lstsq` appears in 3 calls (37, 48, 56). All three compute a descriptive R² of margins on activation scores; the coefficients never become weights. No regression, optimizer or autograd was used.
- **Statistics used for weights:** group means, principal components and covariances of saved activations, plus a midpoint of group-mean margins for the bias. The no-fitting rule allows all of these.
- **Running the network:** every accuracy and margin comes from forward-pass outputs (`rows.jsonl`); fc was never applied to saved features in the kernel. Reading gains of directions through weight matrices is weight analysis, not computation on an image.
- **A borderline point for the rules ladder:** the bias is set from labelled group-mean margins. This is a 1-parameter statistic, not a fit, but it uses the labels. It is allowed as the rule is written.

## Predictions, scored
| # | prediction | result | verdict |
|---|---|---|---|
| 2 | works without refusing (P = 0.95) | worked the whole budget | ✓ |
| 3 | test worst +10–20 over ERM, mean falls ≤ 2 (P = 0.55) | +18.1; mean −2.1 | ✗ narrowly (mean) |
| 4 | reaches DFR − 3 (P = 0.3) | DFR + 0.9 | ✓ (well beyond) |
| 5 | edit in fc or layer4, under 1% of weights (P = 0.75) | layer4 + fc; 8.9% of weights | ✗ (location ✓, size ✗) |
| 6 | practice predicts test within 5 (P = 0.65) | gap about 1 | ✓ |
| 7 | 2–6 h and 5–15 weekly points (P = 0.6) | 58 min; weekly points to be added | ✗ (time) |

## What it means
1. **Without fitting, at 25 passes, one run matched and slightly exceeded the published last-layer retraining method.** The repair came from activation statistics and a causal, dose-checked account of where the background enters. This is a single run; the grid will show whether it is typical.
2. **The diagnosis matches R.4a's in shape.** In both, the right signal survives and the fix removes a nuisance component near the readout. Here Fable also cut it upstream (layer4.1 and layer4.2), which beat the fc-only version by about 3 points on practice.
3. **Not a regression in disguise.** DFR finds its fc weights by an ℓ1 logistic fit. Fable's edit projects out measured directions and re-centres the bias. Whether the fitting-allowed arm does better, or simply reaches the same place faster, is the next comparison.

## Surprises or problems
- **Faster than predicted:** 58 min against a predicted 2–6 h.
- **Quota:** the 5-hour session meter rose quickly (20% → 28% in about 4.5 min mid-session). The weekly cost will be added when the user reports the reading after the session.
- **A metadata leak, cosmetic:** the model view's safetensors metadata shows the original checkpoint's repo path. Fable printed it in its first call. The kernel cannot read the path (V.2 check 2); I will remove the field before the grid.
- **Weight count on the deck:** the deck's birds slide says ResNet-50 has "25 million weights". The two-class model has 23.5M.

## Records
- **Copied to `artifacts/v/v3_b25_nofit/`:** the driver summary, the scoring result, the call log, `notes.md` and the submitted tensors.
- **Full session:** `~/wst-v/sessions/v3_b25_nofit_20261122T044105Z/`, including the transcript and the per-pass outputs.
- **Usage readings:** `artifacts/v/usage_log.txt`.

## Next
- **Size the grid** from the weekly reading.
- **Run the fitting-allowed b25 run next.** It is the counterpart for this result, and its config is ready (`exp_v/sessions/v3_b25_fit.json`).
- **Then the grid:** 3 runs each at 10, 25 and 50 passes in both arms, then the synthesis.
