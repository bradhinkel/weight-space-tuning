# NOTES (session: synthesis; budget 50 passes, 300 code runs)

## Digest of prior work (read from prior/, no passes)
- 18 prior runs + DFR all land on the SAME plateau out-of-fold: about LL 96.5-97.5 / LW 91-93 / WL 91-93 / WW 93-94,
  worst ~91-92.5, weighted mean ~95.6-96.3. fc-only closed-form (bg mean-diff removed from d, or d_bird _|_ d_bg),
  in-network cuts (layer4.2.conv1 reader, layer4.1 reader, layer3->layer4.0 readers), periphery mask (run14): all equal.
- Mechanism (consistent across runs): margin ~ additive: bird +16, background +8 (sd ~5). bg linearly decodable from layer3 on;
  cos(bg_diff,bird_diff) 0.04 (layer3) -> 0.08 (4.0) -> 0.30 (4.1) -> 0.57 (avgpool). Block 4.2 writes ~59% of bg shift and 87% of bird shift.
  Not channel-sparse (top100 ch = 42%). Spatially: scenery cells carry the bg vote, bird cells the bird vote.
- Fitted fc refits (LR/LDA, DFR-like) hold out WORSE (~90 worst) than closed-form directions. Richer readouts (concat layers, RBF) no gain.
- Residual errors = recognition/label floor: ids are in CUB species order; quirk species labelled waterbird: Eastern Towhee p0110-p0116 (7),
  W. Meadowlark p0505-p0507 (3), W. Wood-Pewee p0587-p0594 (8; always wrong, not separable). LW errors: real waterfowl/sea lions/people in water scenes,
  crows/kingfishers, tiny/transparent birds. Towhee boost rejected by CV in 3 runs. Ring-kill mask: no AUC gain => noise is in bird cells.
- After correction mismatched groups keep larger margin sd (5.2/4.9 vs 4.1/4.3) (saturation in block 4.2).
- DFR in-sample practice worst 91.6-93.1, but its own half/half tuning scores are only 0.87-0.89 worst.

## Plan
1. PASS 1: original, all practice, pooled layer3/layer4.0/layer4.1/avgpool -> score all fc-only candidates + DFR re-fits in kernel (CV).
2. Honest CV comparison DFR vs closed-form recipes (re-derived on train folds). Operating point for test group sizes (2255/2255/642/642).
3. Passes on in-network candidates only if they can add something; causal checks (dose-response, control, composites) for the final edit.

## Forward-pass log
- PASS 1 (planned): model=original, items=all, no own images, activations pooled: layer3, layer4.0, layer4.1, avgpool. work/edited empty.

- PASS 1 DONE (work/fwd_1): original on all. Acc LL 99.36 LW 82.62 WL 72.93 WW 96.24 (worst 72.93, wmean 97.72). fc(avgpool)==logits to 3e-6.
  Group mean margins -13.73/-4.94/+2.99/+10.34 (sd 4.2/5.6/5.6/4.3); bird effect 16.0, bg effect 8.08. cos(bg,bird): layer3 .041, 4.0 .078, 4.1 .296, avgpool .568.

## Kernel findings after pass 1 (no passes)
- In-sample kernel scores of prior fc-only edits reproduce their reports; DFR seeds in-sample 96.8-97.4 / 91.6-93.1 / 100 / 98.5-100 (waterbird groups = pure in-sample).
  DFR direction has cos ~0.15 with original d (from-scratch refit); prior closed-form edits cos 0.42-0.72.
- HONEST DFR (exact recipe re-run inside each training fold, nested 10-fold x5): LL 95.59 LW 89.27 WL 91.88 WW 94.44 | worst 89.27, wmean 95.05;
  AUC land .9774 water .9678 hard(LW vs WL) .9630. Every repetition LW 88.6-89.9.
- Closed-form recipes re-derived on same folds (10-fold x5):
  B  (w = bird mean-diff orthogonalised to bg mean-diff; bias = class-mean midpoint): 97.47/92.32/92.18/93.53, AUC land .9867 water .9757 hard .9775; EER-acc land 95.6 water 92.6
  d0 (w = d0 - (d0.D) v, v = D with bird part removed, v.D=1):                         AUC land .9866 water .9745 hard .9759; EER-acc land 95.4 water 92.5
  lightly whitened variants: land .985, water .976-.977 (EER water 93.2-93.5): all within noise -> flat plateau.
  DFR: EER-acc land 92.6 water 93.0 -> DFR is as good as anyone on the water pair, 3 pts worse on the land pair, and its threshold leans to waterbird.
- Background-specific readouts (fitted on water-only or land-only images) do NOT beat the shared direction on their own pair (water .975-.978 vs .976; land .980-.983 vs .987) -> no gated readout.
- Error anatomy (OOF, B recipe): persistent errors LL 12, LW 36, WL 10, WW 8 (DFR 18/46/11/7; shared 8/32/9/7).
  Quirk blocks confirmed by viewing: towhee p0110-p0116 (7: groups 3,2,2,3,2,2,2), meadowlark p0505-p0507 (3, all WL), pewee p0587-p0594 (8: groups 2,3,3,3,2,3,2,3).
  10 of the 18 waterbird errors are quirk images (7 pewee, 3 towhee). Non-quirk waterbird accuracy: WL 95.9, WW 97.6.
  Test split has ~30 images of each quirk species (~2.3% of each waterbird group each) -> waterbird groups slightly better on test than on practice; LW likely binds.
- LW errors viewed (24 of 36): about half contain a second animal/person in the water photo (duckling p0051, geese p0087, spoonbills p0098, goose p0191, swimming dogs p0265,
  sea lions p0398/p0526, merganser p0433, surfers p0468/p0474, goose p0530, ducks p0596, person p0622), rest = grackles, nighthawks in flight, kingfishers, transparent paste-ups (p0541,p0560), tiny birds.
  => LL-vs-LW gap after correction is scene-object evidence, not place code. Run15's spatial pass: no error image has strong true-class local evidence -> robust pooling/gating has nothing to recover.

## Design of passes 2+: every pass = model 'both', items=all, + 256 own images (mirrored practice images, work/mir/m{idx:04d}.png, chunk c = idx 256c..256c+255),
   activations pooled avgpool, layer4.1, layer4.0, layer3. Original-model features of own images accumulate a MIRROR view (for kernel scoring of fc-only candidates);
   the 'edited' slot carries a different in-network intervention each pass (causal localization on all practice images).
   helper erase_read(conv, bn, v, w, mu, alpha): W' = W - alpha (W v) w^T/(w.v); bn.running_mean' = rm - alpha (W v)(w.mu)/(w.v)  (dirs saved work/dirs41.npz)
- PASS 2 (planned): edited = E1: erase_read('layer4.2.conv1.weight','layer4.2.bn1', v41 (bg mean-diff of pooled layer4.1 out), w41 (v41 with bird dir removed), mubar41, alpha=1); fc ORIGINAL.
   own images = mirror chunk 0 (m0000..m0255). Prediction: bg gap 8.08 -> ~3.3 (block-4.2 write 4.73 -> ~0), bird ~16, LW ~87, WL ~89.

- PASS 2 DONE (work/fwd_2): E1 (layer4.2.conv1 reader blind to v41, neutral clamp, fc orig): 98.07/87.34/89.47/94.74 (paired LW +25/-3, WL +22/-0, LL -6, WW -2).
  Margin gaps read with d0: total bg 8.08->3.16, bird 16.00->15.92; skip part bg 3.34 (unchanged); block-4.2 write bg +4.73 -> -0.18, bird 13.94 -> 13.87.
  => one scene direction at the layer4.2 input mediates ~60% of the shortcut; remaining 3.3 logits = fc reading the skip path.
  Kernel CV on E1 features: cut + re-derived fc has the SAME frontier as fc-only (E[min] 92.84 vs 92.82 for B; d0 91.63 vs 92.02; lda 92.73 vs 93.36) -> in-network cut adds nothing to the repair.
  Mirror chunk 0 stored (MIR dict; work/mirror_feats_partial.npz): corr(mirror margin, original margin) .975, sign agreement 97.3%, mean |diff| 1.67 logits.
- Light whitening (lda a=.95) vs B: paired bootstrap water EER +0.60+-0.68, land -0.27+-0.77 -> not significant; scene-animal LW errors NOT helped (z 2.44 -> 2.65). Variants indistinguishable.
- Operating point (kernel, test-weighted, smoothed h=.5): at exact cancellation E[min]~91.3 (mean 96.4); over-cancel beta~1.10-1.15 -> E[min] 92.1-92.5 (mean 95.4-95.8); beta 1.25+ -> mean < 95.
- PASS 3 (planned): edited = LAND-reference cut: erase_read_c('layer4.2.conv1.weight','layer4.2.bn1', v41, w41, c=c_land (scene coordinate clamped to land-group mean), alpha=1), fc orig.
  own images = mirror chunk 1 (m0256..m0511). Prediction: ~100/96/74/90 (run06 pass 9); decisive = CV water-pair EER after fc re-fit vs fc-only 92.6-93.6.
  (pass 3 own images changed to ONE array work/npy/pass3.npy = my preprocessing (PIL bilinear 256x256, crop 16:240, ImageNet norm) of mirrored idx 256..505 + sanity rows mirrored idx 0..5; order in work/npy/pass3_order.json)

- PASS 3 DONE (work/fwd_3): LAND-reference cut, orig fc: 99.79/95.49/75.94/90.98 (LW +60/-0, WW -7). bird gap 19.42, bg gap 3.41, but sd on water images grows (LW 5.56->6.52, WW 4.33->5.87).
  After fc re-fit (CV): water EER 92.5-93.2, frontier E[min] 92.5 @mean>=95.75 = same as fc-only -> reference value is only an operating-point dial. REFUTED as an improvement.
  npy route validated (feat diff <=0.0011, margin diff <=0.003). Mirror features stored for idx 0..505.
- PASS 4 (planned): edited = run14 submission (13 tensors: ring-kill mask + layer4.2 reader cut + fc), items=all, own = work/npy/pass4.npy (mirrored idx 506..761), activations avgpool/layer4.1/layer4.0/layer3.

- PASS 4 DONE (work/fwd_4): run14 submission reproduces 97.22/92.70/92.48/93.23. Frontier on its features (CV re-derived fc): E[min]@mean>=95.75: B 92.97, d0 92.84, lda 93.41
  vs original features B 92.82, d0 92.02, lda 93.36 -> ring mask + reader cut = same plateau. Mirror features stored idx 0..761.
- PASS 5 (planned): edited = run16 submission (layer4.0 readers + layer4.2 reader cuts, fc untouched), items=all, own = work/npy/pass5.npy (mirrored idx 762..1017).

- PASS 5 DONE (work/fwd_5): run16 submission reproduces 96.57/91.20/91.73/93.23. (frontier numbers in kernel output run 40.) Mirror features stored idx 0..1017.
- PASS 6 (planned): edited = run01 submission (layer4.1+layer4.2 2-direction reader cuts + fc), items=all, own = work/npy/pass6.npy:
  rows 0..180 mirrored idx 1018..1198; rows 181..254 = 74 surround-swap composites (37 fg x land/water donor, box 144 at [40:184]; meta work/npy/pass6_meta.json); row 255 = p0000 unmirrored sanity.

- PASS 6 DONE (work/fwd_6): run01 submission reproduces 96.36/93.13/92.48/93.98; frontier on its features with re-derived fc: B 93.20 (fc-only 92.82), within noise.
  MIRROR VIEW COMPLETE: work/mirror_feats.npz (avgpool/layer4.1/layer4.0/layer3 of the original model for mirrored idx 0..1198). Original on mirrors 99.36/82.83/72.93/96.24; sign agreement 96.9%.
  Composites (37 fg x land/water donor, box 144): surround effect ORIGINAL +6.93+-0.59 (37/37 >0); fc-only B -0.19+-0.44.
- TWO-VIEW CV (fit on train-fold images both views, score held-out both views), frontier E[min]@mean>=95.75 (test-weighted): B 93.15, d0 92.51, blend 92.91, lda.95 93.37 (lda-B across
  6 feature sets averages +0.12 -> equal). CHOSEN DIRECTION: B = bird mean-diff orthogonalised to bg mean-diff, scaled to bird effect 16.
- Operating point: margin = w_B.f + b0 - delta*qv.(f - mbar) + c, qv = (d0.D) v, v = bird-blind place direction (v.D=1). Surface smooth; ridge delta .10-.15, c -.375..-.125.
  Paired simulation vs OOF DFR (test group sizes, species weights): (delta .10, c -.25): acc 96.5/93.8/93.7/94.2, worst 93.1, mean 95.86; P(worst>DFR)=1.0, P(mean>DFR)=.93, P(worst>=92)=.92.
  Bootstrap: (.10,-.25) vs exact cancellation: E[min] +0.58+-0.41 (P>0 .93), mean -0.31+-0.13.
- CANDIDATE F1 (work/F1_fc.npz): fc-only; w = w_B - 0.10*qv, bias = b0 + 0.10*(mbar.qv) - 0.25; fitted on all 1199 images x 2 views; common mode of fc rows kept.
- PASS 7 (planned): edited = F1 (fc.weight, fc.bias only), model both, items=all, own = work/npy/pass7.npy = SHIFT view 's00' (crop [0:224,0:224] of the 256x256 resize) of idx 0..255,
  activations avgpool/layer4.1/layer4.0/layer3.

- PASS 7 DONE (work/fwd_7): F1 through the network == kernel: 96.79/93.35/93.23/92.48, worst 92.48, wmean 95.67; paired LL 0/-12, LW +52/-2, WL +27/-0, WW 0/-5. Saved work/best_1 (= work/F1_fc.npz).
  Shift view 's00' chunk 0 stored (SH dict, work/shift_feats_partial.npz).
- PASS 8 (planned): edited = layer4.2 reader cut alpha=0.5 (erase_read_c(..., v41, w41, c_neutral, alpha=0.5)), fc original; own = work/npy/pass8.npy (s00 view idx 256..511).
  Prediction: block-4.2 bg write ~ +2.3 (half of 4.73), bird write ~13.9.

- PASS 8 DONE (work/fwd_8): reader cut alpha=0.5, orig fc: 98.93/86.05/83.46/94.74. (decomposition in DOSE dict / kernel run 49). Shift chunk 1 stored.
- PASS 9 (planned): reader cut alpha=2.0 (same helper), fc original; own = work/npy/pass9.npy (s00 view idx 512..767). Prediction: bg write ~ -5 (reversal), bird write ~14.

- PASS 9 DONE (work/fwd_9): reader cut alpha=2.0, orig fc: 93.15/93.78/96.24/91.73 (reversal). Shift chunk 2 stored.
  DOSE-RESPONSE at layer4.2 reader (alpha 0/0.5/1/2): block-4.2 bg write +4.73/+2.23/-0.18/(see run 50); bird write 13.94/13.87/13.87/..; total bg 8.08/5.57/3.16/..
- PASS 10 (planned): CONTROL cut: erase_read_c('layer4.2.conv1.weight','layer4.2.bn1', pc0, pc0, c=balanced mean, alpha=1) with pc0 = top within-group PC of pooled layer4.1 output
  orthogonal to v41,u41 (work/control_pc0.npy); fc original; own = work/npy/pass10.npy (s00 view idx 768..1023). Prediction: no change in bg gap (~8.1) and accuracies (~99.4/82.6/72.9/96.2).

- PASS 10 DONE (work/fwd_10): CONTROL cut (pc0, rel weight change 2.9%): 99.36/83.26/72.18/96.24 -> shortcut intact (decomposition in DOSE['control']). Shift chunk 3 stored.
  Dose table at layer4.2 reader: alpha 0/0.5/1/2 -> bg write +4.73/+2.23/-0.18/-5.74, bird write 13.94/13.87/13.87/14.08, total bg 8.08/5.57/3.16/-2.40;
  acc a=0.5: 98.93/86.05/83.46/94.74 ; a=1: 98.07/87.34/89.47/94.74 ; a=2: 93.15/93.78/96.24/91.73.
- PASS 11 (planned): edited = layer3->layer4.0 cut: erase_read_c on 'layer4.0.conv1.weight'/'layer4.0.bn1' and 'layer4.0.downsample.0.weight'/'layer4.0.downsample.1' with v3 (bg mean-diff of pooled layer3),
  w3 (bird-blind), balanced clamp, alpha=1; fc original. own = work/npy/pass11.npy: rows 0..174 s00 view idx 1024..1198; 175..214 = 20 swap pairs box 112; 215..254 = 40 bird-free scenes (20 land, 20 water); 255 sanity p0000.
  Prediction: total bg gap -> ~1.3-1.5 (-80%), bird ~15.9.

- PASS 11 DONE (work/fwd_11): layer3->layer4.0 cut (both readers), orig fc: 98.93/90.77/85.71/88.72; total bg gap 8.08->1.33 (84% removed), bird 15.88; within-water AUC .973->.953 (damage) -> evidence, not repair.
  SHIFT VIEW COMPLETE (work/shift_feats.npz). F1 on shift view 96.57/93.35/93.23/92.48; original 99.36/82.40/74.44/96.24.
  Composites: swap effect original +6.93+-0.59 (box144, n=37), +6.80+-0.78 (box112, n=20); F1 -0.93+-0.44 / -2.11+-0.80. Bird-free scenes (20 land/20 water): original -9.75/+3.48 (gap 13.2; 0/20 and 17/20 called waterbird); F1 -3.60/+0.26 (gap 3.9).
- 3-view CV surface (B, test weights): (0,0) E[min] 92.28 mean 96.20; (.05,-.25) 92.67/96.16; (.10,-.25) 92.90/95.87 [F1]; (.15,-.25) 93.10/95.57. OOF at F1 point per view: center 96.7/93.1/93.2/92.5, mirror 96.5/94.0/93.5/93.8, shift 96.8/93.3/91.9/92.6.
- NESTED selection check (30 half-splits; select on A, score B, practice weights): adaptive selection no better than fixed points (E[min] 90.5 vs 90.5-90.8); B(.10,-.25)-B(0,0) = +0.26+-0.82. => do not optimise further; keep F1.
- run01 notes: its own HOLD validation = plateau -> not pursued.
- PASS 12 (planned): edited = threshold-LOWERING test: layer4.2.bn3.bias += 1.0*mu_c (mu_c = mean pooled avgpool activation per channel, from pass 1), fc original.
  own = work/npy/pass12.npy = ZOOM-OUT view 'zout' (whole image resized to 224x224, i.e. 0.875x) of idx 0..255. Decisive: CV frontier of re-derived B-fc on these features vs 92.8 on original features.

- PASS 12 DONE (work/fwd_12): threshold-LOWERING (bn3.bias + 1.0*mu_c): orig fc 99.57/83.69/66.92/95.49; CV frontier with re-derived fc: B 92.29, lda 92.78 (orig feats 92.82/93.36) -> NO gain. REFUTED.
  Zoom-out chunk 0 stored (ZO dict, work/zout_feats_partial.npz).
- PASSES 13-16 (planned): zoom-out chunks 1..4 (work/npy/pass13..16.npy; chunk c = idx 256c..), edited slot = prior in-network submissions run05, run07, run10, run17 (one per pass), to complete the comparison table.
- PASS 13 DONE: run05 submission 96.36/93.13/93.23/93.23 (TABLE dict in kernel). Zoom-out chunk 1 stored. PASS 14 (planned): run07 submission + zoom-out chunk 2 (work/npy/pass14.npy).
- PASS 14 DONE: run07 submission 97.43/93.78/92.48/92.48 (in-sample). Zoom-out chunk 2 stored. PASS 15 (planned): run10 submission + zoom-out chunk 3 (work/npy/pass15.npy).

- PASS 15 DONE: run10 submission 96.57/92.70/94.74/92.48 (in-sample). Zoom-out chunk 3 stored.
  Paired bootstrap (re-derived B-fc on each model's features vs original features): run07 dAUC land +.0016+-.0010 water 0 ; run01 water +.0021+-.0018, hard +.0029+-.0013; 1-D cuts land +.0015, water -.0015.
  => no in-network edit is demonstrably better than fc-only. In-network question CLOSED. F1 stays.
- PASS 16 (planned): run17 submission; own = work/npy/pass16.npy: rows 0..174 zoom-out idx 1024..1198; rows 175..254 = 40 swap pairs box 144 (meta work/npy/pass16_meta.json); row 255 sanity p0000.

- PASS 16 DONE: run17 submission 96.36/91.20/92.48/93.23. ZOOM-OUT VIEW COMPLETE (work/zout_feats.npz). work/edited = F1 again (restored after prior-run passes).
  All fc-only candidates on 4 views (kernel): F1 center 96.8/93.3/93.2/92.5, mirror 96.4/94.6/94.0/94.0, shift 96.6/93.3/93.2/92.5, zoom-out 97.2/93.3/91.0/92.5.
  DFR seeds stay 97-98% on waterbird groups on other views of the SAME images -> views are not independent of fitted images; nested CV remains the honest DFR estimate (95.6/89.3/91.9/94.4).
  Composites pooled (77 pairs box 144): surround effect original +6.60+-0.43 (75/77 >0), F1 -0.64+-0.33.
  Prior in-network submissions, one pass each (TABLE in kernel): run01 96.36/93.13/92.48/93.98, run05 96.36/93.13/93.23/93.23, run07 97.43/93.78/92.48/92.48, run10 96.57/92.70/94.74/92.48,
  run14 97.22/92.70/92.48/93.23, run16 96.57/91.20/91.73/93.23, run17 96.36/91.20/92.48/93.23 (all in-sample).
- PASSES 17-18 (planned): SPATIAL pilot, model=original, spatial=true, layers layer4.1 + layer4; items = work/spatial_pilot.json A (32 LW errors) then B (4 LW errors + 28 controls, 7/group).
  Question: do LW errors come from a few cells with very large positive votes (-> per-cell vote saturation could help) or from wrong-sign/weak votes on the bird itself?

- PASSES 17-18 DONE (work/fwd_17, work/fwd_18): spatial maps (layer4.1, layer4) for 36 LW errors + 28 controls. Per-cell votes under F1 reproduce pooled margins (diff .02).
  LW errors: strongest positive cell averages only +9.7, <1 cell below -20 (correct landbirds: 10-13 cells below -20, min ~-48). Examples: p0433 merganser region +15..+22, junco only two cells at -12;
  p0191 whole centre +15..+26; p0051 ~no landbird votes. Clipping per-cell votes at T=60..10 fixes 0/31 errors -> ROBUST POOLING REFUTED; errors = pasted bird not registered as landbird (recognition limit).

- Estimator variants (3-view CV): no-pewee / no-quirk / trimmed means -> frontier 92.9-93.0 (plain 92.9), AUCs identical. Direction insensitive. F1 FROZEN as candidate.
- PASS 19 (planned): edited = F1 (installed), model both, items=all, own = work/npy/pass19.npy: 31 LW-error centres x (2 land + 2 water donors) [rows 0..123], 31 correct-LW control centres x 4 donors [124..247],
  8 unmodified LW error images [248..255]; box 144; meta work/npy/pass19_meta.json. Question: do LW errors travel with the centre (intrinsic) or with the periphery (scene content)?

- PASS 19 DONE (work/fwd_19): F1 re-confirmed (96.79/93.35/93.23/92.48). LW-error centres on new donors: F1 margin +2.85 (orig image) -> +1.76 (new land) / +1.46 (new water); 21/31 and 23/31 still called waterbird;
  control LW centres -9.72 -> -8.73/-9.03 (0-1 of 31). => ~25 of 31 LW errors travel with the centre crop (bird misperceived or animal adjacent); ~5 caused by peripheral scene content (p0098, p0970, p0398, p0742, p0766).
- Surround-induced subspace at layer4.1 output estimated from 462 within-centre deviations of composites (work/surround_pcs_41.npy, top 16 PCs).

- PASSES 20-24 (planned): ZOOM-IN view 'zin' (central 196x196 of the 256 resize -> 224) chunks 0..4 (work/npy/pass20..24.npy); edited slots = surround-subspace cuts at the layer4.2 reader:
  Q = top-k PCs of work/surround_pcs_41.npy with the bird direction u41 projected out, re-orthonormalised; W' = W - (W Q^T) Q ; bn1.running_mean' = rm - (W Q^T)(Q mubar41); fc original.
  pass 20: k=2 ; pass 21: k=4 ; pass 22: k=8 ; passes 23-24: tbd. Decisive: paired AUC (land, water) of re-derived B-fc on these features vs fc-only on original features.
  REMEMBER: restore F1 (install_fc(wF1,bF1) from work/F1_fc.npz or copy work/best_1) before the final confirmation pass.
- PASS 20 DONE: surround cut k=2 @4.2 reader, orig fc 98.93/84.55/84.96/95.49 (paired stats in kernel PAIRED dict). Zoom-in chunk 0 stored (ZI). PASS 21 (planned): k=4, zoom-in chunk 1 (work/npy/pass21.npy).
- PASS 21 DONE: surround cut k=4, orig fc 98.93/84.55/83.46/95.49. Zoom-in chunk 1 stored. PASS 22 (planned): k=8, zoom-in chunk 2 (work/npy/pass22.npy).

- PASS 22 DONE: surround cut k=8, orig fc 99.14/85.84/82.71/94.74. DIMENSIONALITY DOSE-RESPONSE (re-derived B-fc vs fc-only, paired bootstrap): k=2 dAUC land +.0004 water -.0002; k=4 0/-.0003; k=8 0/-.0004 (SE ~.001)
  -> removing surround variation at the reader does NOT improve separability. In-network question closed by own test. Zoom-in chunks 0-2 stored.
- PASS 23 (planned): edited = fc at lambda=0.5 on the line original->F1 (w = d0 + 0.5(wF1-d0), b likewise); own = zoom-in chunk 3 (work/npy/pass23.npy). Kernel prediction printed in run 75.
- PASS 24 (planned): lambda=1.5; own = zoom-in chunk 4 (+ spare slots). THEN RESTORE F1.
- PASS 23 DONE: lambda=0.5 through network = kernel: 98.72/89.06/87.22/93.98. Zoom-in chunk 3 stored. PASS 24 (planned): lambda=1.5; own = work/npy/pass24.npy rows 0..174 zoom-in idx 1024..1198; 175..205 = 31 LW-error centres on grey; 206..236 = 31 control centres on grey; 237..255 unmodified refs. RESTORE F1 AFTERWARDS.

- PASS 24 DONE: lambda=1.5 through network = kernel: 88.65/95.92/98.50/88.72 (worst 88.65, mean 89.06) -> reversal. F1 DOSE-RESPONSE (tool): lambda 0/0.5/1/1.5 -> worst 72.9/87.2/92.5/88.7, mean 97.7/97.2/95.7/89.1.
  work/edited = F1 again (restored). ZOOM-IN VIEW COMPLETE (work/zin_feats.npz).

- F1 on 5 views (kernel, in-sample fit): center 96.8/93.3/93.2/92.5, mirror 96.4/94.6/94.0/94.0, shift 96.6/93.3/93.2/92.5, zoom-out 97.2/93.3/91.0/92.5, zoom-in 96.8/93.6/92.5/91.7 (mean 95.5-96.0 everywhere).
  OOF version of the recipe: center 96.7/93.1/93.2/92.5, mirror 96.5/94.0/93.5/93.8, shift 96.8/93.3/91.9/92.6, zoom-out 97.1/93.1/90.4/92.5, zoom-in 96.8/93.4/92.5/91.7.
  Centre-only on grey (F1 margin): LW-error centres -1.63+-0.88 (12/31 called waterbird), control centres -13.23+-0.71 (0/31) -> error crops carry no clear landbird evidence even without scenery.
- PASS 25 (planned): site map: bg direction cut at layer4.1 reader (erase_read_c('layer4.1.conv1.weight','layer4.1.bn1', v40, w40, c40)), fc original; own = s32 view (crop [32:256,32:256]) chunk 0 (work/npy/pass25.npy).
- PASS 26 (planned): double dissociation: BIRD direction cut at the layer4.2 reader (v=u41, bird functional with bg part removed, balanced clamp), fc original; own = s32 chunk 1. RESTORE F1 afterwards.
- PASS 25 DONE: bg cut at layer4.1 reader, orig fc 98.72/85.41/84.96/95.49 (bg gap numbers in kernel run 79). s32 chunk 0 stored. PASS 26: bird-direction cut at 4.2 reader + s32 chunk 1.

- PASS 25 DONE: bg cut at layer4.1 reader: 98.72/85.41/84.96/95.49; total bg gap 8.08->5.41 (33% removed), bird 15.76.
- PASS 26 DONE: BIRD-direction cut at layer4.2 reader, orig fc: 98.50/19.10/34.59/98.50 -> scene classifier (double dissociation with the scene cut 98.07/87.34/89.47/94.74).
  SITE MAP (single cut, share of bg gap removed): layer4.0 readers of layer3 84% | layer4.1 reader 33% | layer4.2 reader 61% | fc 100%.
  work/edited = F1 (restored). s32 chunks 0,1 stored (work/s32_feats_partial.npz).
- PASSES 27-28 (planned): F1 installed, model both, items all, own = s32 chunks 2,3 (work/npy/pass27.npy, pass28.npy).

- PASS 27 DONE: F1 re-confirmed (96.79/93.35/93.23/92.48); s32 chunk 2 stored. Bird cut decomposition: block-4.2 bird write 13.94 -> 2.01 (bg write 4.76 unchanged): ONE input direction carries 86% of the bird evidence the block writes.
- PASS 28 (planned): BIRD-GAIN test: erase_read_c('layer4.2.conv1.weight','layer4.2.bn1', u41, wb41, cb41, alpha=-1.0) = gain x2 on the bird coordinate (deviation from balanced mean) at the layer4.2 reader; fc original.
  own = s32 chunk 3 (work/npy/pass28.npy). Prediction: bird effect >> 16, bg ~8, mismatched groups up; decisive = paired AUC of re-derived B-fc vs fc-only. RESTORE F1 AFTERWARDS.

- PASS 28 DONE: bird gain x2 at layer4.2 reader, orig fc: 98.72/88.63/81.20/93.23; bird effect 16.0->30.8, bg 8.1->8.8, sds double (9-11) -> block is LINEAR in the bird coordinate;
  re-derived B-fc: dAUC land -.0017+-.0018, water +.0003+-.0016 -> no separability gain. Closed. s32 chunk 3 stored. work/edited = F1 (restored).
- PASS 29 (planned): F1 installed, both, items all; own = work/npy/pass29.npy: rows 0..174 s32 idx 1024..1198; 175..254 = 80 new bird-free scenes (40 land, 40 water; meta pass29_meta.json); 255 sanity.

- PASS 29 DONE: F1 re-confirmed; s32 VIEW COMPLETE (work/s32_feats.npz); F1 on s32 95.93/93.56/95.49/93.23. Bird-free scenes pooled (60 land/60 water): original gap 12.3 (1/60 vs 49/60 called waterbird); F1 gap 2.55 (11/60 vs 33/60).
  4-view CV surface (center, mirror, s00, s32): (.10,-.25) acc 96.52/93.62/93.73/93.93, E[min] 93.07, mean 95.80; (.05,-.25) 92.80/96.06; (.15,-.25) 93.30/95.56.
  View-noise-whitened direction: view-noise sd 1.28->0.95 but AUC land -.0014..-.0028, EER land -1.0..-1.4 -> REFUTED.
- PASSES 30-39 (planned): TOOL-MEASURED 10-FOLD CROSS-FIT via multiplexed fc. Fold k of FOLDS[0] (StratifiedKFold(10, shuffle, random_state=100) on g):
  fc row0 = F1 recipe fitted on the other 9 folds (views center+mirror; delta .10, c -.25); row1 = full DFR recipe fitted on the other 9 folds (rng seed 5000+k).
  edited logit0 = F1-recipe margin, logit1 = DFR margin (tool summary for 'edited' meaningless in these passes). pass 30+k = fold k.
  own images: passes 30-34 = mild zoom-in x1.06 chunks 0..4 (work/npy/pass30..34.npy); passes 35-39 = mild zoom-out x0.94 chunks 0..4.
  AFTER PASS 39: RESTORE F1 (install_fc(wF1,bF1) or copy work/best_1) and confirm.
- PASS 30 DONE (fold 0 of cross-fit stored in work/xfit_partial.npz). Next: pass 31 = fold 1.
- PASS 31 DONE (fold 1 of cross-fit stored in work/xfit_partial.npz). Next: pass 32 = fold 2.
- PASS 32 DONE (fold 2 of cross-fit stored in work/xfit_partial.npz). Next: pass 33 = fold 3.
- PASS 33 DONE (fold 3 of cross-fit stored in work/xfit_partial.npz). Next: pass 34 = fold 4.
- PASS 34 DONE (fold 4 of cross-fit stored in work/xfit_partial.npz). Next: pass 35 = fold 5.
- PASS 35 DONE (fold 5 of cross-fit stored in work/xfit_partial.npz). Next: pass 36 = fold 6.
- PASS 36 DONE (fold 6 of cross-fit stored in work/xfit_partial.npz). Next: pass 37 = fold 7.
- PASS 37 DONE (fold 7 of cross-fit stored in work/xfit_partial.npz). Next: pass 38 = fold 8.
- PASS 38 DONE (fold 8 of cross-fit stored in work/xfit_partial.npz). Next: pass 39 = fold 9.

- PASSES 30-39 DONE: TOOL-MEASURED 10-fold cross-fit (work/xfit_full.npz; tool==kernel to .006). Held-out, practice weights: F1 recipe 96.79/93.35/93.23/92.48 (worst 92.48, mean 95.67);
  DFR 95.72/89.06/92.48/95.49 (worst 89.06, mean 95.37). Test species weights: F1 worst 93.18 mean 96.01; DFR worst 89.06 mean 95.59.
  Paired (F1-only right vs DFR-only right): LL 10/5, LW 23/3, WL 3/2, WW 0/4. Sim at test sizes: P(worst>DFR)=1.00, P(mean>DFR)=.915, P(worst>=92)=.87.
  DFR held-out gets 7/7 towhees (F1 recipe 5/7). Mild zoom views complete (work/z106_feats.npz, work/z094_feats.npz): F1 x1.06 95.9/94.0/94.0/92.5 ; x0.94 96.1/93.1/94.7/92.5.
  work/edited = F1 (restored after pass 39).

- Towhee add-on (nested, 4 views, test weights): gamma=1 -> E[min] 93.15 vs 93.07, mean 95.71 vs 95.78 (acts like a bias shift) -> rejected (4th time).
- PASSES 40-49 (planned): F1 installed (work/edited = F1), model both, items all; own = COUNTERFACTUAL-SURROUND set: for centre i (all 1199 practice images, 128 per pass, pass p covers centres 128(p-40)..),
  rows (2j, 2j+1) = centre on land donor CF_land[i], on water donor CF_water[i] (work/cf_donors.npz, rng seed 20261122), box 144 at [40:184]. Files work/npy/pass40..49.npy.
  Measures prediction flip rate under surround swap for original / F1 (tool) and DFR (kernel). PASS 50 = final confirmation of F1.
- PASS 40 DONE (counterfactual centres 0..127 stored in work/cf_partial.npz; F1 re-confirmed [96.79, 93.35, 93.23, 92.48]).
- PASS 41 DONE (counterfactual centres 128..255 stored in work/cf_partial.npz; F1 re-confirmed [96.79, 93.35, 93.23, 92.48]).
- PASS 42 DONE (counterfactual centres 256..383 stored in work/cf_partial.npz; F1 re-confirmed [96.79, 93.35, 93.23, 92.48]).
- PASS 43 DONE (counterfactual centres 384..511 stored in work/cf_partial.npz; F1 re-confirmed [96.79, 93.35, 93.23, 92.48]).
- PASS 44 DONE (counterfactual centres 512..639 stored in work/cf_partial.npz; F1 re-confirmed [96.79, 93.35, 93.23, 92.48]).
- PASS 45 DONE (counterfactual centres 640..767 stored in work/cf_partial.npz; F1 re-confirmed [96.79, 93.35, 93.23, 92.48]).
- PASS 46 DONE (counterfactual centres 768..895 stored in work/cf_partial.npz; F1 re-confirmed [96.79, 93.35, 93.23, 92.48]).
- PASS 47 DONE (counterfactual centres 896..1023 stored in work/cf_partial.npz; F1 re-confirmed [96.79, 93.35, 93.23, 92.48]).
- PASS 48 DONE (counterfactual centres 1024..1151 stored in work/cf_partial.npz; F1 re-confirmed [96.79, 93.35, 93.23, 92.48]).
- PASS 49 (planned): F1 installed; own = work/npy/pass49.npy: rows 0..93 = centres 1152..1198 (land, water donors); rows 94..255 = 81 replicate centres with a second donor pair (meta pass49_meta.json). PASS 50 = final confirmation of F1 (both, all).

- PASS 49 DONE. COUNTERFACTUAL-SURROUND SET COMPLETE (work/cf_full.npz; 1199 centres x land/water donor, box 144):
  original: flip rate 18.4% (LB centres 15.1, WB centres 30.1), surround effect +7.23+-0.12; acc land surround [LB 98.4, WB 63.9], water surround [LB 83.3, WB 94.0]
  F1      : flip rate 5.4% (5.6 / 4.9), surround effect -0.20+-0.09; acc land [93.9, 91.4], water [91.5, 91.0]. Flips: original only 165, F1 only 9, both 56.
  DFR seeds 0/4 (kernel; fitted on these birds): flip 7.6-8.3%, surround effect +0.60..+0.76; LB centres 86-88% on water surround vs 92-94% on land.
  Donor variability: F1 margin sd between two donors of the same type 3.3-3.8; between types mean -0.26 sd 3.6 -> residual = donor-specific content, not place type.
## FINAL DECISION: F1 (fc.weight, fc.bias only) = work/edited = work/best_1 = work/best_final (work/F1_fc.npz).
  w = [bird mean-diff orthogonalised to bg mean-diff, scaled to bird effect 16] - 0.10 * (d0.D) * v (v = bird-blind place direction, v.D=1); bias = class-mean midpoint + 0.10*(mbar.qv) - 0.25; means over 1199 practice images x (center, mirror) views.
  Practice (tool): 96.79/93.35/93.23/92.48, worst 92.48, mean 95.67. Tool-measured 10-fold held-out recipe: identical counts; DFR held-out 95.72/89.06/92.48/95.49.
- PASS 50 (planned): final confirmation, model both, items all, no own images.
- PASS 50 DONE (final confirmation): F1 = 96.79/93.35/93.23/92.48, worst 92.48, weighted mean 95.67; paired vs original LL 0/-12, LW +52/-2, WL +27/-0, WW 0/-5. All 50 passes used. SUBMISSION = work/edited (fc.weight, fc.bias) == work/best_final.
