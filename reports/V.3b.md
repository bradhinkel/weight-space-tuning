# V.3b: pilot, budget 25, fitting-allowed arm

## What was done
- **Session:** `v3_b25_fit_20261122T061448Z`.
  - Fable 5.2, effort max, budget 25, fitting allowed: scikit-learn available, and fc may be computed on saved avgpool features in the kernel.
  - The task is the same template as V.3, with the arm's paragraph swapped in.
  - Wall clock 67 min (06:14Z–07:21Z on 2026-11-22).
  - 114 tool calls: 80 python, 25 forward and 9 show_images (136 images).
  - 0 nudges, 0 compactions, 0 waits. $23.5 notional.
- **Lock-down:** the init check passed.
- **Usage:** the user upgraded the Max plan just before this run, and both meters read 0% at its start.

## Results: Waterbirds test
| model | landbird/land | landbird/water | waterbird/land | waterbird/water | worst | weighted mean |
|---|---|---|---|---|---|---|
| broken (ERM) | | | | | 75.4 | 98.3 |
| DFR, our run (5 seeds) | | | | | 92.58 ± 0.44 | 95.17 ± 0.39 |
| Fable b25, no fitting (V.3) | 97.16 | 93.88 | 93.46 | 93.46 | 93.46 | 96.18 |
| **Fable b25, fitting allowed** | 96.98 | 93.97 | 94.39 | **93.30** | **93.30** | **96.03** |

- **The two arms are within noise of each other:** 0.16 points apart on worst group, against an SE of about 1 point for a 642-image group. Both are about 0.7–0.9 above our DFR.
- **Practice vs test:**
  - practice, in-sample: worst 91.7, mean 96.1;
  - practice, held-out halves: 91.0 / 95.9;
  - Fable's forecast: "worst 91–92, mean 95.9".
  - Test came in 1.6 points better than practice, so the forecast was conservative again.

## The edit
- **Two tensors:** `fc.weight` and `fc.bias`, 4,098 values.
- **The fc edit:**
  - It keeps the original read-out and subtracts the part that responds to the background.
  - The part removed lies along the two background-shift directions (land→water within landbirds and within waterbirds), whitened by the within-group covariance with shrinkage. This is closed form and close to LDA in spirit.
  - The strength is 1.075 times the background-cancelling amount.
  - The bias is re-centred between the group means, then −0.125.
  - Strength and bias were chosen by cross-validation over held-out halves, using real forward passes.
- **What Fable tried and rejected,** each with no held-out gain:
  - full logistic refits (the DFR-style fit);
  - anchored corrections;
  - a species add-on;
  - penalties for sensitivity to surround, mirroring and crops;
  - in-network edits followed by an fc refit.
- **Its reasoning:** the bird information the network extracts is the ceiling, so an fc edit is the simplest repair that reaches it.

## Diagnosis: the same as V.3, found independently
- **Size of the shortcut:** the bird moves the margin about +16.0 and the background about +8.1, and the two add.
- **Where it arises:** the bird/background cosine grows only in the last two blocks: 0.04 → 0.08 → 0.30 → 0.57. This matches V.3's numbers.
- **In-network causal checks,** with fc untouched:
  - a one-direction cut at layer4.2.conv1 takes the background effect from 8.1 to 2.4;
  - ablating 89 background-selective layer4.1 channels gives a dose-response; random channels give none.
- **Its own test images:** 188 of them, birds pasted on backgrounds and backgrounds with no bird. The original model calls every bird-free water background "waterbird". After the edit, the water-minus-land gap is +0.9 on bird-free backgrounds and −1.0 for the same bird on a swapped surround.
- **Label noise:** Fable found one flycatcher species (p0587–p0594) labelled waterbird. It looks like a landbird and accounts for 7 of the 18 remaining waterbird errors. This is a known kind of label noise in CUB-derived Waterbirds.

## Rule audit
- **No gradients, autograd or optimizers through the network,** and no convolution, pooling or batch-norm computed in the kernel (all 80 python calls scanned).
- **Fitting** appears in 15 calls (logistic regression, closed-form solves). The fitting arm allows it, and it was applied to saved activations only.
- **A borderline readout:** in calls 27–30 and 99, Fable applied fc to each spatial position of the saved layer4 maps. Because average pooling is linear, this is fc on avgpool features taken apart position by position, and I count it as allowed. It is the CNN version of R.4a's readout from saved residuals, and the rules ladder should name it.

## Predictions
- **17. Fitting-allowed runs reach DFR − 1 at b25 (P = 0.75):** ✓ for this run, at DFR + 0.7.
- **18. At least one fitting run beats DFR by ≥ 1 (P = 0.35):** not yet (+0.7). The grid decides.

## What it means
1. **At one run each, being allowed to fit added nothing at b25.** Both arms land about 1 point above DFR. The fitting arm tried DFR's own method (logistic refits) and rejected it on held-out data, in favour of a statistics-based projection.
2. **Two runs, same diagnosis, different sites.** V.3 cut the background in two layer4 blocks plus fc; V.3b did it in fc only, after showing that the in-network cuts gave no extra separation. Both found the same numbers for the shortcut (8 against 16, and the cosine rising in the last two blocks).
3. **Both forecasts were conservative,** by 1–1.6 points.

## Records
- **Copied to `artifacts/v/v3_b25_fit/`:** the driver summary, the scoring result, the call log, `notes.md` and the submitted tensors.
- **Full session:** `~/wst-v/sessions/v3_b25_fit_20261122T061448Z/`.
