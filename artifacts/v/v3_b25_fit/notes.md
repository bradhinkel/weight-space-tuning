# NOTES (Waterbirds ResNet-50 repair)

## Setup facts
- fc.weight [2,2048], bias ~[0.0255,0.0107]; wd = w1-w0 norm 1.675, diffuse (top300 channels = 51% of sq norm).
- practice groups: g0 LL 467, g1 LW 466, g2 WL 133, g3 WW 133. Mean-acc weights 0.7295/0.0384/0.0117/0.2204. Targets on test: worst>=88, wmean>=94.

## Findings so far (after pass 1; all fc-on-avgpool computations done in kernel, they match saved logits to 3e-6)
- Baseline: LL 99.4, LW 82.6, WL 72.9, WW 96.2, wmean 97.72.
- Margin (out1-out0) group means: LL -13.7, LW -4.9, WL +3.0, WW +10.3 (sd 4.2-5.6). => bird effect 16.0, BACKGROUND effect 8.1 (additive, interaction -1.4).
- Shortcut is NOT in dedicated channels: only 36 bg-only channels (0.2 of the 8.1); bg-shift and bird-shift directions in avgpool have cos 0.57.
- Depth profile cos(d_bg,d_bird): layer3 0.04, layer4.0 0.08, layer4.1 0.30, avgpool 0.57. Place decodable ~94% at all depths. Background-free bird evidence
  (balanced probe min AUROC): layer3 88, l4.0 95, l4.1 96.6, avgpool 97.2-97.7 -> repair belongs at fc.
- DEBIAS EDIT (kernel, 5-fold CV x4): w = wd - s*U*beta, U = (S_within + shrink*tr/p*I)^-1 [mu1-mu0, mu3-mu2], beta cancels bg effect; bias centres 4 group means.
  dose-response (1-dir): s=0: [99.1 76.9 81.8 96.2]; .5: [98.6 84.1 87.6 94.7]; 1.0: [97.1 90.6 92.3 94.0] wmean 96.1; 1.25: [95.7 92.3 95.3 93.2] wmean 95.0; 1.5: [93.4 93.9 96.2 92.1]
  held-out bg effect 8.1 -> 4.4 -> 0.7 -> -1.1 -> -3.0. Bias shift alone on original fc cannot exceed worst 79.
- Plateau: balanced L2 LR, anchored LR, constrained LDA, debias+anchored correction all give min AUROC 97.2-97.8, worst ~90-93 at wmean>=95.5. No fit beats plain debias.
- Learning curve flat from 50% to 100% of data -> no need for flips/augmentation.
- Residual errors: label noise species. ids are ordered by species. p0587-p0594 = pewee-like flycatchers LABELLED waterbird (7/8 wrong after debias, not learnable: LOO z~1).
  p0110-p0116 = towhees labelled waterbird (borderline margins; cleanly detectable LOO z 2.5-9). Scene sub-type (4 clusters) residual effect <1 margin unit -> 1 direction enough.

## Forward passes log
- PASS 1: original, all, activations=[avgpool, layer4.1, layer4.0, layer3] pooled. No edits. -> work/fwd_1
- PASS 2 (planned): original, items = 32 ids in work/spatial_sel.json (8 per group, seed 7), activations=[layer4, layer4.1, layer4.0, layer3], spatial=true. No edits. Bird boxes in same json.
- PASS 2: original, 32 ids (work/spatial_sel.json), spatial maps layer4/4.1/4.0/3 -> work/fwd_2. CAM result: original fc bg-cell contribution water-land = +9.7 (per-cell x49),
  debiased fc -0.6; bird-cell waterbird-landbird gap 27.5 -> 26.9. Along wd pooled: bg effect 1.7 (after l4.0), 3.3 (after l4.1), 8.1 (final); bird effect 0.5, 2.1, 16.0.
- Operating point analysis (bootstrap of OOF): best region strength 1.0-1.2, bias offset -0.5..+0.5; s=1.1,t=0: est groups [96.9 92.3 94.1 93.5], P(both targets) ~0.97.
- PASS 3 (planned): model=both, items=all, images = 188 custom PNGs listed in work/custom/custom_set.json (144 composites bird-on-tile, 24 birds on grey, 20 bg tiles),
  activations=[avgpool]. EDIT = fc only: fit_debias2(X,y,g,pl,shrink=3,two=True,strength=1.1) i.e. w = wd - 1.1*U*beta, U=(S_w+3*tr/p I)^-1 [mu1-mu0, mu3-mu2],
  bias centred on 4 group means; saved as work/fc_debias_s1.1_w.npy/_c.npy; fc.weight rows changed by +-(w_new-wd)/2.
- PASS 3 DONE (work/fwd_3): both, all + 188 custom. Edited (fc debias s=1.1) in-sample: [97.0 92.92 92.48 93.23] worst 92.48 wmean 95.96 (kernel prediction exact).
  paired: LW +49/-1, WL +26/-0, LL 0/-11, WW 0/-4. CV estimate for this edit: [96.9 92.3 94.1 93.5].
  Custom images: tiles alone water-land margin: orig +13.7+-1.6, edited +0.6+-0.9. Same bird swapped surround (24 birds): orig +8.75+-0.65, edited -1.24+-0.44.
  Composite acc LL/LW/WL/WW: orig 1.00/.72/.44/.89, edited .83/.92/.69/.81. custom avgpool saved in fwd_3 (rows 1199:), table work/custom/custom_results_pass3.csv
- BEST SO FAR: work/best_1 = fc.weight+fc.bias (debias s=1.1; w in work/fc_debias_s1.1_w.npy, c in _c.npy).
- PASS 4 (planned): PROBE (not a repair): model=edited, items=all, activations=[layer4.1, layer4] pooled. Edit: layer4.2.conv3.weight := selection (out ch j <- bottleneck ch j, j<512),
  layer4.2.bn3 := identity (weight 1, bias 0, running_mean 0, running_var 1); fc edits dropped. Then pooled h2 = layer4[:, :512] - layer4.1[:, :512].
- PASS 4 DONE (work/fwd_4) but probe was only HALF installed (save_edit refuses all-zero tensors; I wrongly launched the pass in the same step). Still usable:
  conv3:=selection, bn3.weight:=1, other bn3 stats original -> pooled h2_j = (out_j - x_j - beta_j)*s_j + rm_j ; exact for 372/512 channels, tiny clipping error for the rest.
  Saved: work/l42_h2_pooled.npy (pooled bottleneck acts, inputs of layer4.2.conv3), work/l42_bgwriter_order.npy (channels ordered by bg/bird contribution ratio).
  LESSON: never launch a forward pass in the same step as code it depends on. To make BN an identity without zeros: weight:=sqrt(rv+eps), running_mean:=bias.
- Bottleneck findings: linear direct effect (omega = wd @ (gamma/s * W3)) correlates 0.995 with block's actual margin contribution (slope 0.673). Block writes bg 4.73 / bird 13.94 (actual).
  84 bird-selective channels write bird 8.2/bg 0.5; top-50 by ratio remove 52% of bg writing for 10% of bird writing; top-100: 73% / 19%.
  fc null-space projection against write vectors of top-k bg writers: bg effect 8.1 -> 5.9(k10) 4.3(30) 3.0(50) 1.9(100) 0.6(200); random-channel controls: no change in bg/bird ratio.
  But min AUROC only 96.3 at k=200 (< fitted debias 97.8): skip path carries 41% of bg effect.
- PASS 5 (planned): model=edited, items=all + 188 custom images (work/req3.json list), activations=[avgpool]. EDIT = in-network mean-ablation of top-50 bg-writing bottleneck channels of layer4.2:
  conv3.weight[:, B50]=0 with B50 = np.load('work/l42_bgwriter_order.npy')[:50]; bn3.running_mean -= W3[:,B50] @ mean_h2[B50]. fc ORIGINAL.
  Prediction: bg effect 8.1 -> ~5.2, bird effect 16.0 -> ~14.7.
- PASS 5 DONE (work/fwd_5): in-network ablation top-50 bg writers, fc original: groups [99.14 84.55 79.7 95.49]; bg effect 5.88 (pred 5.15), bird 14.09 (pred 14.72);
  tiles water-land +10.4 (was +13.7); same-bird surround shift +6.28 (was +8.75). Refit debias on ablated features: SAME plateau (min AUROC 97.76 vs 97.71) -> no gain over fc-only.
- Kernel checks (no passes): towhee add-on rejected (LL errors 14.8->18.0, no WGA gain); whitened anchored correction = plateau; invariance penalty from composites: AUROC +0.1-0.3 only,
  d' unchanged, P(pass) unchanged -> rejected; concatenating h2/l4.1/l4.0/l3 pooled features to avgpool: same plateau -> ceiling is informational. FINAL = fc-only debias.
- PASS 6 (planned): model=edited, all + 188 custom (work/req5.json), avgpool. EDIT = in-network mean-ablation of top-150 bg writers (order[:150]); fc original. Pred: bg ~3.4-4, bird ~12.6.
- PASS 7 (planned): same but random 50 channels (work/l42_random50.npy), control.
- PASS 6 DONE (work/fwd_6): in-network ablation top-150 bg writers (fc original): groups [98.9 87.1 82.0 94.7]; bg 4.23 bird 12.15 (ratio .35); tiles +7.7; surround shift +4.52. Refit: same plateau.
- PASS 7 DONE (work/fwd_7): CONTROL ablation of 50 random bottleneck channels: groups [99.6 81.6 68.4 96.2] -> no improvement (specificity confirmed).
  In-network dose-response (k=0/50/150): bg effect 8.08/5.88/4.23, bird 16.0/14.1/12.2, same-bird surround shift +8.75/+6.28/+4.52, tiles gap +13.7/+10.4/+7.7.
- NOTE: session continues until all 25 passes are used. Plan for remaining passes: mirrored (flip) copies of practice images to measure/penalise label-preserving noise in the read-out
  (pass 8 = 256 flips, 64/group, seed 21, files work/flip/<id>.png, list in work/flip_sel.json); then decide; final verification pass of the final fc edit LAST.
- All edits dropped after pass 7 (work/edited empty); best fc edit is in work/best_1.
- PASS 8 DONE (work/fwd_8): 256 mirrored images (rows 1199:), original model. Mirror noise small: rms(margin-margin_flip)=1.68 under debias vs within-group sd 4.26 (8% of variance);
  2% of predictions change; two-view average AUROC 98.43->98.59 on held-out 256. 
- Clean LOO (train-only covariance): pewees NOT separable anywhere (z~0.5; earlier 8.3 in concatenated space was a leak artifact); towhees z~4.7 (real) but add-on rejected by CV.
- PASSES 9-12 (planned): mirrored copies of remaining 943 images, batches in work/flip_batches.json, requests work/req_flip_{0..3}.json (model=original, items=[p0000], avgpool).
  Purpose: CV test of mirror-invariance penalty on read-out. Adopt only if clear held-out gain.
- CHANGE OF PLAN for passes 9+: use .npy inputs (own preprocessing: PIL resize 256 bilinear, centre crop 224, ImageNet normalise) to avoid long file lists.
  PASS 9 (planned): model=original, items=[p0000], images=[work/aug/pass9.npy] = 16 unmodified check images (p0000-p0015) + 240 augmented views (random crop side 180-256 of the 256 image,
  resized to 224, random mirror; seed 101; meta in work/aug/pass9_meta.json), avgpool. Check: features of the 16 must match pass 1.
- PASS 9 DONE (work/fwd_9): own preprocessing via .npy matches harness exactly (16 check images). Crop/zoom/mirror views: margin noise rms 2.29 vs sd 4.21 (15% of variance).
  View-invariance penalty (496 pairs) and combination with surround penalty: NO held-out gain (plateau) -> rejected. FINAL remains fc-only debias s=1.1, t=0.
- PASS 10 (planned): PROBE layer4.1 bottleneck: layer4.1.conv3.weight:=selection, layer4.1.bn3.weight:=sqrt(rv+eps), layer4.1.bn3.running_mean:=bias (exact identity BN).
  model=edited, items=all, activations=[layer4.0, layer4.1] pooled. h2_41 = layer4.1[:, :512] - layer4.0[:, :512].
- PASS 10 DONE (work/fwd_10): layer4.1 bottleneck probe OK (exact). Saved work/l41_h2_pooled.npy. Findings: corr(z_bg,z_bird)=0.18; 89 bg-selective channels (direct bg 1.53/bird 0.10),
  27 bird-selective, 2 mixed. Summed output of l4.2 top-50 bg writers predictable from l4.1 bottleneck with held-out R2 0.90.
- PASS 11 (planned): model=edited, items=all, images=[work/custom/custom188.npy] (same 188 custom images, order of work/custom/custom_set.json), avgpool.
  EDIT = mean-ablation of 89 bg-selective channels of layer4.1 bottleneck (work/l41_bgsel89.npy): layer4.1.conv3.weight[:,B]=0, layer4.1.bn3.running_mean -= W3[:,B]@mean_h41[B]; fc original.
- PASS 11 DONE (work/fwd_11): ablate 89 bg-selective channels of layer4.1 (fc original): groups [99.1 83.3 81.2 96.2] wmean 97.68; bg effect 6.49 (direct-only pred 7.32), bird 14.55;
  tiles +12.1, surround shift +7.51. Refit: same plateau. => layer4.2 gets most bg info via skip path from earlier layers.
- PASS 12 (planned): combined ablation l4.1 (89) + l4.2 top-150, fc original; model=edited, all + work/custom/custom188.npy, avgpool.
- PASS 12 DONE (work/fwd_12): combined in-network ablation l4.1(89)+l4.2(top150), fc original: groups [98.7 86.7 86.5 94.0] wmean 97.07; bg 3.24 bird 10.95; tiles +6.4; surround shift +3.58.
  Frontier comparison (CV): fc-only on original features dominates fc-debias on ablated features (e.g. worst>=91: wmean 96.23 vs 95.80) -> final stays fc-only.
- PASS 13 (planned): long shot 'peak-driven pooling': layer4.2.bn3.bias -= 1.0 * X.mean(0) (per-channel mean pooled activation); fc original; model=edited, all + custom188.npy, avgpool.
  Then refit read-outs in kernel and compare AUROC with baseline plateau (mean ~98.0).
- PASS 13 DONE (work/fwd_13): peak-driven pooling (bn3.bias -= mean act), fc original: groups [99.6 86.1 73.7 95.5]; bg 4.98 bird 11.48; surround shift +5.28. Refit: same plateau -> rejected.
- SPLIT-HALF CONFIRMATION (kernel): halves in work/halfA.npy, work/halfB.npy (stratified, seed 2024). Select (s,t) + fit on A -> s=1.1,t=+0.25 -> held-out B [95.73 89.70 97.01 93.94];
  select+fit on B -> s=1.0,t=-0.25 -> held-out A [97.85 92.27 87.88 94.03]; pooled held-out [96.79 90.99 92.48 93.98] worst 90.99 wmean 95.90.
- PASS 14 (planned): model=both, items=all, fc edit = half-A fit (work/fc_halfA_w.npy/_c.npy); confirm half-B accuracies from rows.
- PASS 15 (planned): same with half-B fit (work/fc_halfB_w.npy/_c.npy); confirm half-A accuracies.
- PASS 14 DONE: half-A edit on held-out half B (real pass): [95.73 89.70 97.01 93.94] (= kernel). PASS 15 DONE: half-B edit on held-out half A: see below; pooled held-out worst ~91, wmean ~95.9.
- PASS 16 (planned): fc debias strength 0.5 (fit on all), model=both, all + custom188.npy, avgpool. PASS 17: strength 1.5. (dose-response through the real network)
- PASS 16 DONE: fc s=0.5 real pass [98.5 85.41 87.97 94.74] wmean 97.05 (= kernel). PASS 17 DONE: fc s=1.5 real pass [91.01 95.28 96.99 90.98] wmean 91.23 (= kernel).
  Real-network dose-response s=0/0.5/1.1/1.5: worst 72.9/85.4/92.5/91.0; wmean 97.7/97.05/95.96/91.2.
- PASS 18 (planned): spatial pass, original model, items=[] , images=[work/custom/spatial32.npy] (indices into custom list in work/custom/spatial32_idx.json: 16 composites = 8 birds x land/water,
  14 tiles, 2 grey), layers [layer4], spatial=true. If items=[] is rejected, use items=["p0000"] and drop one image.
- PASS 18 DONE (work/fwd_18): spatial maps for 32 synthetic images. Same bird water-vs-land surround: orig total +7.16, bird-cell sum +0.11, bg-cell sum +5.32; edited total -2.39, bird -0.63, bg -1.45.
  Tiles per-cell: orig land -5.16 / water +7.61; edited -3.75 / -3.84. Bird-cell waterbird-landbird difference 5.39 (orig) vs 5.61 (edited).
- PASS 19 (planned): CONTROL for layer4.1: mean-ablate 89 random non-bg-selective bottleneck channels (work/l41_random89.npy), fc original; model=edited, all + custom188.npy, avgpool.
- Then: install FINAL fc edit (work/best_1 = debias s=1.1,t=0) and use passes 20-24 for augmented views of all practice images (robustness), pass 25 = final verification (both, all + custom).
- PASS 19 DONE (work/fwd_19): control ablation of 89 random l4.1 channels: [99.6 82.6 69.9 95.5] -> no improvement (specific).
- FINAL EDIT INSTALLED in work/edited (= work/best_1): fc debias strength 1.1, bias centred (t=0), fit on all 1199 practice images.
- PASSES 20-23 (planned): model=both, items=[], images=[work/aug/mirror_k.npy] k=0..3 (mirrored standard views of the 943 images not in pass 8; index lists in work/aug/mirror_batches.json), avgpool.
- PASS 24: spare. PASS 25: final verification (both, all + custom188.npy).
- PASSES 20-23 DONE (work/fwd_20..23): mirrored views of the remaining 943 images; full mirrored features in work/mirror_feats_all.npy.
  Held-out (CV by image) debias s=1.1: originals [96.8 92.4 94.2 93.6], mirrored [96.1 93.1 93.2 93.6]; fitting on both views: no change.
  Held-out bg effect on real images: s=1.00 +0.33, 1.05 -0.05, 1.10 -0.44. P(pass) plateau 0.95-0.97 for s 1.05-1.15, t -0.5..+0.25 (fine grid flat).
- FINAL CHOICE: fc debias strength 1.075, bias offset -0.125 (centre of plateau). Saved work/final_fc_w.npy, work/final_fc_c.npy. NOT YET INSTALLED (spare pass 24 uses original fc).
- PASS 24 (planned, mechanism only): remove linear place direction from input of layer4.2 residual branch: layer4.2.conv1.weight -= outer(W1 d/(u.d), u), bn1.running_mean adjusted; fc original;
  model=edited, all + custom188.npy, avgpool. AFTER IT: drop all edits, save_fc(final w,c), verify, PASS 25 = final verification (both, all + custom188.npy).
- PASS 24 DONE (work/fwd_24): rank-one removal of place direction at layer4.2.conv1 input (fc original): groups [98.5 90.77 87.22 93.23] wmean 96.91; bg effect 2.41 (from 8.08), bird 15.80;
  tiles +6.65; same-bird surround shift +2.67. => last block's whole bg contribution is read from ONE input direction. fc refit on these features: frontier not better than fc-only
  (worst>=91: 96.03 vs 96.23) and direction was in-sample -> NOT adopted; reported as mechanism.
- FINAL EDIT INSTALLED: work/edited = fc.weight + fc.bias only; debias strength 1.075, bias offset -0.125 (work/final_fc_w.npy, work/final_fc_c.npy); backup work/best_final.
  Kernel (exact) in-sample: [97.22 92.92 91.73 93.23] worst 91.73 wmean 96.11. Held-out estimates: CV ~[96.8 92.9 93.3 93.6]; split-half real passes pooled [96.8 91.0 92.5 94.0].
- PASS 25 (planned, LAST): final verification: model=both, items=all, images=[work/custom/custom188.npy].
- PASS 25 DONE (work/fwd_25): FINAL VERIFICATION. edited [97.22 92.92 91.73 93.23] worst 91.73 wmean 96.11 (= kernel exactly); paired LW +49/-1, WL +25/-0, LL 0/-10, WW 0/-4.
- ALL 25 PASSES USED. SUBMISSION = work/edited (fc.weight, fc.bias) = work/best_final. Nothing else to do.
