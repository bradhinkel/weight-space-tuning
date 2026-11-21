# V.1 GATE: reproduce the baselines — PASS

Run 2026-11-20 on the laptop (RTX 5070 Laptop, 8 GB). Fable not involved.

## What was done
- **ERM (the broken model):** `exp_v/v1_erm.py`, saved to `artifacts/v/broken/model.pt`.
  - Model: torchvision ResNet-50 `IMAGENET1K_V1`, with a new 2-class last layer.
  - Recipe (DFR Appendix C): SGD with momentum 0.9, constant learning rate 1e-3, weight decay 1e-3, batch 32, 100 epochs. Augmentation: RandomResizedCrop(224, scale 0.7–1) and flips.
  - fp32, seed 0, no early stopping: the final epoch is the model.
  - Training took 33 minutes.
- **DFR:** `exp_v/v1_dfr.py`, run on the broken model's 2,048-dim features.
  - Recipe (DFR Appendix C):
    1. Standardize the features with `StandardScaler`.
    2. Fit an ℓ1 logistic regression (liblinear).
    3. Pick C from the paper's grid: fit on half of validation, scored by worst-group on the other half.
    4. Refit 10 times on group-balanced subsamples of all validation (4 × 133 = 532 images each), and average the weights.
  - 5 seeds, as in the paper.
  - The seed-0 layer is also saved in the network's own units (`artifacts/v/broken/dfr_fc_seed0.pt`, a drop-in `fc.weight` / `fc.bias`) for the V.2 self-test.
  - Loaded into the network, that layer reproduces its sklearn test score exactly (92.68 / 95.28).
- **Scoring:** `exp_v/wb.py`. Worst-group accuracy, plus mean accuracy weighted by training-group share (the published definition, amendment 2) and the plain average. The test set was scored once per model.

## Results (test, 5,794 images)
| | worst-group | weighted mean | plain mean | landbird / land | landbird / water | waterbird / land | waterbird / water |
|---|---|---|---|---|---|---|---|
| **ERM (ours, seed 0)** | **75.4** | **98.3** | 91.2 | 99.7 | 85.5 | **75.4** | 97.0 |
| ERM (DFR Table 2) | 74.9 ± 2.4 | 98.1 ± 0.1 | – | | | | |
| **DFR (ours, 5 seeds)** | **92.6 ± 0.4** | **95.2 ± 0.4** | 94.1 | | | | |
| DFR (DFR Table 2) | 92.9 ± 0.2 | 94.2 ± 0.4 | – | | | | |
| DFR seed 0 | 92.7 | 95.3 | 94.2 | 95.6 | 92.7 | 93.9 | 94.9 |

- **DFR per seed** (worst / weighted mean, chosen C):
  - seed 0: 92.7 / 95.3, C = 0.3
  - seed 1: 92.8 / 94.9, C = 0.7
  - seed 2: 92.9 / 95.1, C = 0.3
  - seed 3: 92.7 / 94.9, C = 0.7
  - seed 4: 91.8 / 95.8, C = 0.1
- **Practice (validation), ERM:** worst-group 72.9 and weighted mean 97.7 (test: 75.4 / 98.3). This is the model's original score on the images Fable will see. The gap of 2.5 points is the size of practice-to-test noise to expect from the 133-image groups.
- **DFR's layer is sparse:** the seed-0 averaged layer uses 460 of the 2,048 features (ℓ1).

## Gate (amendment 3)
| check | target | result | |
|---|---|---|---|
| ERM worst-group within 5 of 74.9 | 69.9–79.9 | 75.4 | pass |
| break present: ERM worst-group | ≤ 80 | 75.4 | pass |
| ERM weighted mean | ≥ 95 | 98.3 | pass |
| DFR worst-group within 3 of 92.9 | 89.9–95.9 | 92.6 (mean of 5) | pass |

**Prediction 1** (ERM worst-group in 65–80, and DFR within 3 points of its published figure; P = 0.8): **holds**.

## Notes
- **The break is in the last layer, as in R.4a.** DFR keeps every feature and refits only the last layer (2 × 2,048 weights plus 2 biases), and that lifts worst-group from 75.4 to 92.6. So the information Fable needs is still in the features.
- **What DFR gives up:** weighted mean falls from 98.3 to 95.2, almost all of it on landbirds on land (99.7 → 95.6). This is the trade Fable's fix will also face, like call accuracy vs. declining in R.4a.
- **Practice scores at every 10th epoch** were logged for information only; they chose nothing. Worst-group moved between 64 and 74 across epochs, so the broken model's worst-group depends on where training stops. One seed is a sample: the paper's ± 2.4 comes from that.

## Files
- `artifacts/v/broken/`:
  - `model.pt` (the broken model, 91 MB, not in git)
  - `train_log.jsonl`
  - `erm_results.json`
  - `dfr_results.json`
  - `dfr_fc_seed0.pt`
  - stdout logs

## Next
V.2 GATE: build the harness (fork `exp_r/r4a/`) and self-test it. The self-test covers:
- the server's scores matching `exp_v/wb.py`;
- the split guard holding from inside the kernel;
- practice predicting test, with DFR fit on one half of practice and scored on the other half;
- the `show_images` cap.
