# Experiment V design: Fable repairs a vision model, many times over

**Status:** reviewed 2026-11-20; decisions made (below). V.0 done (`reports/V.0.md`); no model trained and no Fable session run. Numbers marked *(verify in V.0)* were checked in V.0; the amendments below take precedence over the text.

**Decisions (user, 2026-11-20):**
1. Driver: Fable 5.2 for all of V.
2. "No fitting": activation statistics (means, differences, top images) are allowed; edits fitted to practice labels, closed-form or not, are refused.
3. Stage 2: design after stage 1's results.

**Amendments from V.0 (2026-11-20, before any run):**
1. Published baselines: ERM 74.9 ± 2.4 worst / 98.1 ± 0.1 mean; DFR 92.9 ± 0.2 / 94.2 ± 0.4; Group DRO 91.4 / 93.5 (DFR Table 2). Validation has 1,199 images; the smallest practice groups have 133; train is 95.0% matched.
2. "Mean accuracy" is the published one: group accuracies weighted by their share of the *training* data. The plain test average is reported alongside it.
3. V.1 targets: ERM worst-group 69.9–79.9 and at most 80; ERM weighted mean at least 95; DFR worst-group 89.9–95.9.
4. V.2 check 3: fit DFR on half of practice and score it on the other half, since DFR fitted on all of practice would be scored in-sample.
5. V.5: plot the practice-to-test gap against budget, to catch selection on the same 133-image groups at high budgets.
6. Stage 2 downloads move from V.0 to V.7.

**Amendments from review (user, 2026-11-20, before any run):**
7. **Grid:** run budgets 10, 25 and 50 first, under the no-fitting rule. Whether 100 and 200 are needed and practical is decided after those, from their results and measured cost.
8. **A second arm, fitting allowed:** the same task, but `save_edit` accepts edits fitted to practice labels, so Fable can use DFR or anything like it. This shows what Fable does with the known method on the table, and whether it beats DFR. Budgets and run counts are set at the V.3 gate (proposal: 3 runs at each of 10, 25 and 50). Runs are scored and analysed in V.5 alongside the no-fitting arm, and kept apart from it.
9. **Final step, V.7 (synthesis):** one more Fable session gets every run's submitted edits, notes and **practice** scores, from both arms. It compares them, then builds and submits one "best" edit, which is scored once on test.
   - The synthesis session never sees test scores. If it did, choosing among runs by test score would leak the test set into the result.
   - It runs under the fitting-allowed rules. The budget is set before it runs (proposal: 25).
   - Stage 2's step numbers move up by one (V.8–V.12).

**Amendments from V.2 (2026-11-21, before any Fable session):**
10. **Sandbox on the laptop:** without root, the kernel can't drop to an unprivileged user. Landlock and seccomp do the confinement; the V.2 self-test shows they block everything the pods' user switch blocked.
11. **Fitting arm:** it may also compute fc on saved avgpool features, which DFR-style tuning needs. The no-fitting arm may not; an fc edit is seen only through a forward pass.
12. **Task text** (`exp_v/sessions/task_template.md`): states the target (worst-group ≥ 88 with weighted mean ≥ 94), the original's practice numbers and the mean's weights. It names no published method.
13. **One pass** = one `evaluate` request over any practice images (up to 1,199) plus up to 256 own images. Pooled activations are available for all of them; full feature maps for at most 32.

**Predictions for the new arms (2026-11-20, before any run):**
17. Fitting-allowed runs reach DFR minus 1 point on test worst-group at budget 25. P = 0.75.
18. At least one fitting-allowed run beats DFR by 1 point or more. P = 0.35.
19. The synthesis edit scores at least as well on test worst-group as the best single run picked by *practice* score. P = 0.6. It beats the best single run picked by *test* score (a bar only hindsight can pick). P = 0.3.

**Goal:** move the weight-space repair task from a 2B language model to a small vision model, where a test pass costs seconds instead of minutes. Use that to answer two questions the language-model runs could not afford to ask:
1. **How does the result scale with budget?** Run the same task at 10, 25, 50, 100 and 200 passes.
2. **Is Fable consistent?** Run each budget several times. Do separate runs find the same fix or different ones, and how much do their scores vary?

A third question follows if runs find different fixes:

3. **Can a nudge steer Fable?** Point it at one of the fixes found earlier and see if the results become more consistent or better.

**Why vision, and why these datasets:** both stages use standard public datasets with published baselines, the same role xLAM and BFCL played in Experiment A. Stage 1 (Waterbirds) is a known break with known fixes, so we can check our setup against the literature before Fable touches anything. Stage 2 (CIFAR-10 with out-of-scope images) is the closest vision analog of the tool router's "no tool" problem.

---

## What carries over from R.2 / R.4a

- **Driver:** Fable at effort `max` in headless Claude Code, with no built-in tools. The only tools are the MCP `python(code)` and `forward(request)`, plus a new `show_images` tool (below).
- **Kernel:** persistent, CPU only, offline, and confined to `/model`, `/practice` and `/work`. API as in R: `load(name)`, `load_original`, `save_edit`, `drop_edit`, `edited()`.
- **Rules:**
  - no gradients or iterative fitting;
  - no running or simulating the model in the kernel, so the pass budget means something;
  - no internet;
  - the test split never enters the sandbox, enforced in code.
- **`save_edit` refuses:** any edit that only multiplies a whole tensor by a constant.
- **Records:** `/work/notes.md`, `calls.jsonl`, the full transcript, every submitted tensor.
- **End of session:** the session runs until the budget is used; the tensors in `/work/edited` are submitted and scored on the sealed test split.
- **Method:** a design committed before any run, with dated predictions, and a gate at every step that can stop the experiment.

## What changes

1. **Everything runs on the laptop.** The models are small (ResNet-50: 25M weights; ResNet-18: 11M), so the kernel, the forward server and the driver all run locally on the RTX 5070. No pods, no GPU rental.
2. **One pass = one scored run over the practice set,** up to the whole set (Waterbirds validation: about 1,200 images *(verify in V.0)*).
   - The server scores every image and returns accuracy per group, so Fable sees the test metric directly.
   - A pass can also return activations at chosen layers for up to 256 images, and the top-activating images for chosen channels.
   - `both` runs the original and the edit on the same images, paired, with the per-image differences.
   - A failed request costs nothing. Original-model results are cached.
3. **Fable can look at images.** A new MCP tool, `show_images(ids)`, returns up to 16 practice images, or their top-activating patches, as images Fable can see. It costs no pass, with a cap of 400 images per session. This is the main capability the language-model runs lacked.
4. **Budgets:** 10, 25, 50, 100 and 200 passes (V.4 sets how many runs at each).

---

## Stage 1: Waterbirds (the replication)

**The dataset** (Sagawa et al. 2020, "Distributionally Robust Neural Networks"): bird photos from CUB pasted onto backgrounds from Places. Two classes, waterbird and landbird. In the training split about 95% of birds sit on the matching background *(verify in V.0)*, so a fine-tuned model learns "water background means waterbird".

| split | role | analog in Experiment A |
|---|---|---|
| train | used once, to make the broken model | xLAM 60K |
| validation | practice: the only images Fable sees | BFCL live_irrelevance + xLAM holdouts |
| test | sealed: scored once per submitted edit | BFCL test categories |

**The scores:**
- **Worst-group accuracy:** the lowest accuracy across the four (bird, background) groups. It is usually "waterbird on land". This is the headline number, like "declines when no tool fits".
- **Mean accuracy** over the test set. Like call accuracy, it must not fall much.

**The published baselines** (ResNet-50, ImageNet-pretrained) *(verify in V.0)*:

| method | what it does | worst-group | mean |
|---|---|---|---|
| ERM | plain fine-tune: this is our "broken" model | about 70–75 | about 97 |
| DFR (Kirichenko et al. 2022) | retrain only the last layer on the validation set | about 92 | about 94 |
| Group DRO (Sagawa et al. 2020) | retrain everything with group labels | about 91 | about 93 |

DFR is the bar that matters: it uses the same validation images Fable gets, and changes only the last layer.

### V.0 Setup and data
- Download Waterbirds (via the WILDS package or the DFR repo), CIFAR-10, CIFAR-10-C and the OpenOOD CIFAR-10 benchmark sets (for stage 2). Record sources, versions and file hashes in `env/lock.md`.
- Check every *(verify in V.0)* number in this doc against the papers. Record corrections here as an amendment before any run.
- Write the split guard: the sandbox's `/practice` holds the validation split only; test images exist only on the scoring side.
- **Report:** `reports/V.0.md`. No gate.

### V.1 GATE: reproduce the baselines
- Train ERM (ResNet-50, ImageNet-pretrained) with the DFR paper's recipe. Save it as `artifacts/v/broken`.
- Run DFR on top of it.
- **Pass if:**
  - ERM worst-group is within 5 points of the published figure and at most 80 (the break is present);
  - ERM mean accuracy is at least 95;
  - DFR worst-group is within 3 points of the published figure.
- **On fail:** report, propose options (another seed, the paper's exact code), and stop. We don't build on a baseline we can't reproduce.
- **Cost:** about 1 h of laptop GPU *(estimate)*.

### V.2 GATE: harness and self-test
- **Build `exp_v/`:** fork `exp_r/r4a/`. Replace the BFCL scorer with group accuracy, add the image `forward` request, `show_images`, and activations and top-activating images.
- **Self-test, the same pattern as R.4a:**
  1. The server's scores match an independent eval script on the same model, image by image.
  2. The split guard holds: no test image path or hash is reachable from the kernel.
  3. **Practice predicts test.** Load the DFR last layer through `save_edit`. One pass on practice must show worst-group within 5 points of its test score. If practice can't see DFR's gain, it won't see Fable's.
  4. `show_images` returns the requested images and enforces its cap.
- **Pass if** all checks pass. **On fail:** fix and rerun; don't start a session.

### V.3 GATE: pilot session (budget 25)
- One Fable session at budget 25, the same budget as R.4a, so the two compare directly.
- **Measure:**
  - **Engagement:** did it work the task?
  - **Time:** wall clock per session and per pass.
  - **Max usage:** session and weekly points used.
  - **Result:** test worst-group and mean accuracy.
  - **Fix:** what it changed and why, from its notes.
- **Gate (the user decides):** pick the V.4 grid from the measured cost (below), or stop.

**Why the grid needs this gate:** R.4a's 25-pass session took 3.3 h and about 10 points of the weekly Max allowance. R.2's 50-pass session took 19 h. Vision passes are faster on the GPU, but most of a session's time is Fable thinking, so the cost per pass may not fall much. At R.4a's rate, the full grid below is several weeks of Max. The pilot replaces that guess with a measurement.

| grid | runs per budget (10 / 25 / 50 / 100 / 200) | sessions | total passes | at R.4a's rate (25 passes ≈ 10 weekly points) |
|---|---|---|---|---|
| S | 5 / 5 / 5 / – / – | 15 | 425 | about 170 points: 2 weeks |
| M | 5 / 5 / 5 / 3 / – | 18 | 725 | about 290 points: 3 weeks |
| L | 5 / 5 / 5 / 3 / 3 | 21 | 1,325 | about 530 points: 5–6 weeks |

Sessions could run two at a time on the laptop (the models are small), which shortens wall clock but not Max usage.

### V.4 Budget grid
- Run the grid chosen at V.3.
- **Order:** interleave budgets (10, 25, 50, 10, 25, …) rather than all of one budget first, so a change in the driver or the harness during the weeks of running doesn't line up with one budget.
- **Fixed across all runs:** driver model and version, effort, task text, practice set, harness version. Any change during the grid is an amendment, recorded before the next run.
- Every run's submitted tensors and transcript are kept.

### V.5 Analysis: scores and solutions
**Scores, per budget:**
- mean, standard deviation and range of test worst-group and mean accuracy;
- the share of runs that reach DFR minus 3 points;
- the budget curve: worst-group against log(budget), with a check for a plateau;
- practice-to-test gap per run.

**Solutions, across runs: are they different?** Four views, from the weights to the behavior:
1. **Where:** which tensors and layers each run changed. Overlap between two runs = shared tensors / all tensors touched.
2. **What:** cosine similarity between two runs' weight changes, per tensor they both touched.
3. **Behavior:** on the test set, how often two edited models give the same answer, overall and on the images the edits changed.
4. **Mechanism:** from each run's notes and its edit, a short label such as "down-weights background features in the last layer" or "suppresses water-texture channels in layer 4". A blind grader (Claude, run names hidden) sorts the labels into groups; I check a sample.

Two runs count as **different solutions** if they fall in different mechanism groups, *or* their weight changes have cosine below 0.3 on the tensors they share. The report also says which runs are **different in weights but the same in behavior** (agreement ≥ 95%), because that is the more interesting finding.

- **Report:** `reports/V.5.md`, with figures: the budget curve with every run as a dot, and a map of solutions (runs placed by behavior similarity, colored by mechanism group). **Phase boundary: stop.**

### V.6 GATE, then nudge experiment (only if V.5 finds at least 2 different solutions)
- **Choose two target fixes from V.5:** the most common one, and the best-scoring one if it differs.
- **Write a one-paragraph hint for each,** in the style of R.4a's task text ("the damage looks like X; consider Y"). No edit values, only the idea.
- **Arms,** at one fixed budget picked from the V.5 curve (likely 50), with the same number of runs as V.4 used at that budget:
  - no hint (reuse the V.4 runs);
  - hint toward the common fix;
  - hint toward the best fix.
- **Gate (before running):** the user approves the hints and the run count.
- **Measure:** score mean and spread per arm, and how many runs in each arm land in the hinted mechanism group.

### V.7 Synthesis (amendment, 2026-11-22, written after V.4 and before the run)
**The question, as the user put it:** given the current best model (DFR), can Fable tune it to better performance through weight-space edits? So the session gets everything except the test split.
- **One session:** Fable 5.2, effort max, budget 50, fitting allowed. Config `exp_v/sessions/v7_synthesis.json`; task text `exp_v/sessions/task_v7.md`, which is the V.4 template plus a "build on everything" paragraph and a fifth item under WHAT YOU HAVE. The goal, the rules and the measurement are unchanged.
- **Read-only `prior/` in the sandbox** (`~/wst-v/v7_prior`, built by `exp_v/v7_prepare.py`):
  - **DFR, seeds 0–4:** fc tensors in the model's own units, refit with the V.1 recipe. Their practice scores are labelled **in-sample**. The DFR test scores stay in `artifacts/`.
  - **All 18 V.3/V.4 runs** as `run01`–`run18`: submitted tensors, final report, `notes.md`, and the arm and budget. Neither the run tags nor any test score are shown.
- **Not given:** any test number, the V.4 report, or the V.4 finding that practice scores don't rank runs on test, because that finding came from test.
- **Harness change:**
  - the server gains an optional `prior` config key, which symlinks the folder into the sandbox;
  - Landlock allows reading it and nothing else new;
  - other sessions are unchanged, since they have no `prior` key.
- **Scored once on test, then compared with:**
  - the 20-seed DFR baseline (`artifacts/v/broken/dfr_20seeds.json`: 92.58 ± 0.51, range 91.26–93.39);
  - the fitting arm (93.27 ± 0.34);
  - paired bootstrap against DFR, as in V.4.
- **Report:** `reports/V.7.md`. It records what Fable started from (DFR, a run, a combination), what it changed, and the agreement of the final model with DFR and with the 18 runs.
- Stage 2's step numbers below (V.7–V.11) move up by one if stage 2 runs.

---

## Stage 2: CIFAR-10 with out-of-scope images (the router analog)

Run only after stage 1's phase boundary, and only if the user decides to. This stage is designed in less detail; it gets its own amendment before V.7.

**The break, built to match the tool router:**
- Fine-tune an ImageNet-pretrained ResNet-18 on a small, clean subset of CIFAR-10: 100 images per class.
- Give the model an 11th output, "none of these", that is trained but never has a positive example. That is what xLAM did to "no tool": the option exists, and training pushes it down.
- **Result we expect:** the model puts every image in one of the 10 classes, including house numbers, textures and other photos that belong to none of them.

**The scores:**

| score | test set | analog in Experiment A |
|---|---|---|
| in-scope accuracy | CIFAR-10 test | mean call accuracy |
| says "none of these" on out-of-scope images | OpenOOD near and far test sets | declines when no tool fits |
| accuracy on messy images | held-out corruption types from CIFAR-10-C | real-user calls |

**Practice** gets OpenOOD's validation images and *other* corruption types from CIFAR-10-C. The messy-image row exists because of what we learned in R.4a: Fable's fix lost real-user calls that practice never showed it. Here practice includes messy images from the start, and the test uses different corruption types, so we can see whether the fix generalizes.

**Comparisons:** last-layer retraining on the practice images with out-of-scope images labelled "none" (the DFR analog), and a full fine-tune on the same images (outlier exposure, Hendrycks et al. 2019).

**Known issues to settle in the amendment:**
- **Pretraining overlap:** some OpenOOD sets (TinyImageNet, Places365) come from ImageNet-like sources, which the backbone was pretrained on. Drop or report them separately.
- **Image size:** CIFAR images are 32×32, too small for `show_images` to help much. Fable will work mostly from numbers, as in R.4a.

**Steps:** V.7 data and break (GATE: the break is present: says "none" on at most 10% of out-of-scope images, in-scope accuracy within 3 points of a plain 10-class fine-tune), V.8 baselines, V.9 harness self-test (GATE), V.10 a smaller grid, V.11 analysis. Same gates and stopping rules as stage 1.

---

## Predictions (recorded before any run)

Probabilities are my own estimates. Each states what would count against it.

**V.1 baselines**
1. ERM worst-group lands in 65–80 and DFR within 3 points of its published figure. P = 0.8.
   - **Against:** a gap larger than 5 points on either means our training or eval differs from the papers, and stage 1 stops.

**V.3 pilot (budget 25)**
2. Fable works the task without refusing. P = 0.95.
3. Test worst-group rises by 10–20 points over ERM, with mean accuracy falling at most 2 points. P = 0.55.
   - **Why:** the shortcut likely lives in a few last-layer weights or late channels, which Fable can see and the practice set measures directly.
4. It reaches DFR minus 3 points. P = 0.3.
   - **Why below DFR:** DFR fits the last layer to all 1,200 practice images. Fable can't fit to data, so it has to reason its way to a similar layer.
5. The edit is in the last layer or the last block (layer 4 of the ResNet), touching under 1% of the weights. P = 0.75.
6. Practice predicts test: worst-group within 5 points. P = 0.65.
   - **Against:** the smallest practice group is small (about 130 images *(verify in V.0)*), so a large gap would mean Fable fit to that group's noise.
7. The session takes 2–6 h and 5–15 weekly Max points. P = 0.6.

**V.4–V.5 grid: budget and spread**
8. Mean worst-group rises with budget and levels off by 100: the 200-pass mean is within one standard deviation of the 100-pass mean. P = 0.6.
9. Run-to-run spread is real: at budgets 10 and 25, the standard deviation of test worst-group is 3–10 points. P = 0.65.
10. Spread shrinks with budget: the standard deviation at 100 is smaller than at 10. P = 0.6.

**V.5 solutions (your speculation)**
11. **Different runs find different fixes:** among the 5 runs at budget 50, at least 2 mechanism groups. P = 0.7.
12. The most common fix is a last-layer change that down-weights background features. P = 0.6.
13. **Different in weights, similar in behavior:** at least half of the pairs of different-mechanism runs still agree on at least 90% of test images. P = 0.5.
    - **Why it matters:** if it holds, Fable finds many ways to the same behavior. If it fails, the runs really are different fixes, and the choice between them matters.

**V.6 nudge (if run)**
14. A hint toward the most common fix cuts the spread of worst-group by at least 30% without lowering the mean. P = 0.5.
15. A hint toward the best fix raises the mean by at least 3 points over no hint. P = 0.4.
16. At least 4 of 5 hinted runs land in the hinted mechanism group. P = 0.6.

**V.7 synthesis (recorded 2026-11-22, before the run)**
19. **User's prediction:** the synthesis improves on DFR, but not by more than the confidence interval.
20. V.7 beats the DFR mean (92.58) on test worst group. P = 0.8.
21. V.7 lands within the fitting arm's range (92.52–93.61). P = 0.65.
22. V.7 beats the best grid run (93.61). P = 0.25.
23. The gain over DFR is outside the per-run paired 95% CI (more than about +1.7). P = 0.1.
24. Fable's final edit builds on DFR's last layer (starts from a DFR seed's fc) rather than on one of its own earlier edits. P = 0.5.
25. The weighted mean stays above DFR's (94.92). P = 0.8.

**What the outcomes would tell us**
- **8 and 10 hold:** more passes help, and also make Fable more consistent. Budget is the lever.
- **9 and 11 hold but 13 fails:** the runs find different fixes that behave differently, so a single run is a sample, not "the" answer. Any claim about what Fable finds needs several runs, and that applies back to R.4a too.
- **14 holds:** a short hint makes the method more repeatable. That points to a practical recipe: explore with several runs, then steer.
- **4 fails at every budget:** in vision, a cheap last-layer refit beats reasoning over weights. That is still a useful result: it says where weight-space programming does and doesn't pay.

## Cost
- **GPU:** laptop only. No rental.
- **Driver:** set by the V.3 gate. At R.4a's rate, grid S is about 2 weeks of Max and grid L about 5–6 weeks (table above).
- **Judge:** the mechanism grader in V.5 is a few Sonnet batches.

## Decisions for the user before V.0
1. **Driver version.** R.2 and R.4a used Fable 5.1. Fable 5.2 is now available. Recommendation: 5.2 for all of V, with the V.3 pilot as its first test; switching mid-grid would spoil the statistics.
2. **The "no fitting" rule, made precise.** In R.4a, Fable solved a least-squares problem against the model's own weights, which was allowed. In vision, the obvious move is a closed-form regression from practice-image activations to labels, which is DFR itself. Recommendation: allow statistics computed from activations (means, differences, top images), but refuse edits fitted to practice labels, closed-form or not. Otherwise the experiment measures how fast Fable rediscovers DFR.
3. **Stage 2:** design it fully now, or after stage 1's results.

## Order
1. Review this doc; make the decisions above.
2. Commit it, so the predictions are dated before any run.
3. V.0 → V.1 (gate) → V.2 (gate) → V.3 (gate) → V.4 → V.5 (phase boundary) → V.6 (gate, if run).
