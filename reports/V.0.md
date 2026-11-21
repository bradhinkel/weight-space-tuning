# V.0: setup, data and literature check

Run 2026-11-20 on the laptop (RTX 5070 Laptop, 8 GB). No gate. Nothing was trained and Fable was not run.

## What was done
- **Environment:** a new venv, `~/vision_env/.venv` (Python 3.12, torch 2.11.0+cu128, torchvision 0.26.0, scikit-learn 1.9.1, pandas 3.1.1). It is separate from the demo venv (`~/demo_env/.venv`), so the demo isn't affected. Lock: `env/lock.md`, section "Experiment V".
- **Waterbirds:**
  - The WILDS CodaLab URL now returns 404.
  - The same archive is mirrored on Hugging Face by a WILDS author: `pangwei/wilds-waterbirds`, `waterbirds_v1.0.tar.gz`.
  - Its SHA-256 matches the mirror's checksum file: `395444bb…0072cc4`.
  - The manifest ties it to CodaLab bundle `0x505056d5cdea4e4eaa0e242cbfe2daa4`, which is the bundle WILDS and the DFR repo download.
  - It is extracted to `~/datasets/waterbirds/extracted`; I checked it for unsafe paths first and found none.
- **Practice folder and split guard:** `exp_v/data.py`.
  - `build` copies the 1,199 validation images to `~/datasets/waterbirds/practice/images/` under neutral ids (`p0000`–`p1198`). The CUB species name in the original path never reaches the sandbox.
  - It also writes `labels.csv` (y, place, group) there, and per-split manifests with image hashes to `~/datasets/waterbirds/manifests/`, which stays on the scoring side.
  - `check` verifies by hash that practice is exactly the validation split, with no train or test image in it.
- **Backbone:** torchvision ResNet-50 `IMAGENET1K_V1` (`resnet50-0676ba61.pth`, SHA-256 `0676ba61…722fb8a`). This is what `resnet50(pretrained=True)` loaded when DFR was written.
- **Literature:**
  - DFR: read from the arXiv PDF, v2 (ICLR 2023). Table 2, Table 12 and Appendix C were checked verbatim.
  - Group DRO: arXiv HTML, v2. The text numbers were read; its Table 1 cells weren't readable.
- **Stage 2 data** (CIFAR-10, CIFAR-10-C, OpenOOD): not downloaded, because stage 2 is deferred by decision 3.

## Split guard: PASS
| check | result |
|---|---|
| practice files | 1,199 |
| practice = validation split (by hash) | yes |
| practice ∩ test | 0 |
| practice ∩ train | 0 |
| validation ∩ test, train ∩ test | 0, 0 |
| stray files in practice | none |
| background photos shared across splits | 0 (every image has its own Places photo) |
| CUB species | all 200 appear in every split (same species, different photos) |

## The data, counted from `metadata.csv`
| split | images | landbird / land | landbird / water | waterbird / land | waterbird / water |
|---|---|---|---|---|---|
| train | 4,795 | 3,498 | 184 | **56** | 1,057 |
| validation (practice) | 1,199 | 467 | 466 | **133** | 133 |
| test (sealed) | 5,794 | 2,255 | 2,255 | **642** | 642 |

- **Train:** 95.0% of birds sit on the matching background (landbirds 95.0%, waterbirds 94.97%). The smallest group is waterbirds on land, with 56 images.
- **Validation and test are not fully group-balanced.**
  - Within each class, backgrounds are balanced 50/50.
  - The classes keep the training ratio: about 78% landbirds.
  - The smallest practice groups have 133 images each.

## Literature check: corrections to `V_design.md`
| item in the design | published | source | verdict |
|---|---|---|---|
| ERM worst / mean "about 70–75 / about 97" | **74.9 ± 2.4 / 98.1 ± 0.1** | DFR Table 2 | corrected |
| (other ERM figure) | 60.0 / 97.3 (standard training, different recipe) | Group DRO §3.1 | noted; DFR's figure is the target, since we use DFR's recipe |
| DFR worst / mean "about 92 / about 94" | **92.9 ± 0.2 / 94.2 ± 0.4** (5 runs) | DFR Table 2 | confirmed |
| Group DRO worst / mean "about 91 / about 93" | **91.4 / 93.5** | DFR Table 2 | confirmed |
| validation "about 1,200 images" | 1,199 | metadata.csv | confirmed |
| training correlation "about 95%" | 95.0% | metadata.csv | confirmed |
| smallest practice group "about 130" | 133 | metadata.csv | confirmed |
| **"mean accuracy over the test set"** | **group accuracies weighted by their share of the *training* data** | DFR Table 2 caption, following Sagawa et al. | **definition changed** (see below) |

**Why the mean definition matters:**
- The test set is background-balanced, but training is 95% matched.
- A plain test-set average therefore gives the mismatched groups far more weight than the published mean does.
- An ERM model is weak on exactly those groups, so its plain average would come out well below 98.1.
- We now report the published, training-weighted mean as "mean accuracy". The plain average is kept as a second column.

**ERM recipe (DFR Appendix C, verbatim):**
- Model: torchvision ResNet-50, ImageNet-pretrained.
- Optimizer: SGD, momentum 0.9, constant learning rate 1e-3, weight decay 1e-3, batch size 32.
- Schedule: 100 epochs, no early stopping.
- Augmentation: `RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 4/3))` and horizontal flips.

**DFR recipe (DFR Appendix C, verbatim):**
1. Take the 2,048-dim features (the input to the last layer) and standardize them with `StandardScaler`, fitted on the reweighting data.
2. Fit an ℓ1 logistic regression on a group-balanced subsample: all of the smallest group (133) plus 133 drawn from each other group, about 530 images.
3. Choose C from {1, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01} by worst-group accuracy, training on half the validation set and tuning on the other half.
4. Refit 10 times on fresh subsamples of the whole validation set, and average the weights.

## Laptop throughput (for the V.1 estimate)
- I measured ResNet-50 training at batch 32 on synthetic input:
  - fp32: 262 img/s, 18 s per epoch, about 0.5 h for 100 epochs.
  - bf16: 479 img/s, about 0.3 h.
  - Peak GPU memory is 3.1 GB, which fits the 8 GB card.
- With JPEG decoding on 24 cores, V.1 should take about 0.5–1 h. That is in line with the design's estimate.
- I'll use fp32, matching the paper, unless the user wants bf16.

## Amendments this makes to the design (recorded before any run)
1. **The scoring definition:** "mean accuracy" is the training-weighted mean (the published definition). The plain test average is reported alongside it.
2. **V.1 targets** are the published figures:
   - ERM worst-group within 5 points of 74.9 (so 69.9–79.9), and at most 80.
   - ERM weighted mean at least 95.
   - DFR worst-group within 3 points of 92.9.
3. **V.2 check 3 (practice predicts test):**
   - The published DFR is fitted on all of practice. Its practice score would therefore be in-sample and would flatter the check.
   - Instead, fit DFR on one half of practice and score it on the other half; compare that score with the test.
4. **V.5 adds a selection-bias check.**
   - At budget 200, Fable can score up to 200 edits on the same 133-image groups. It can drift toward what happens to work on those particular images.
   - The practice-to-test gap is therefore plotted against budget, not only per run.
5. **Stage 2 data downloads** move from V.0 to V.7.

## Next
V.1 GATE: train ERM with the recipe above, then run DFR on top, and check both against the targets in amendment 2. It needs no rental and takes about an hour on the laptop.
