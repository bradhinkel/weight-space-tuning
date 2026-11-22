# NOTES (waterbirds repair, budget 25 forward passes; NO fitting to labels allowed)

## Setup facts
- fc.weight [2,2048], bias ~[0.0255,0.0107]. decision dir w=fc[1]-fc[0], norm 1.675, dense (top100 |w|^2 = 26%).
- practice groups: g0 LB/land 467, g1 LB/water 466, g2 WB/land 133, g3 WB/water 133.
- mean-acc weights: g0 .7295, g1 .0384, g2 .0117, g3 .2204. Target: worst>=88, mean>=94 on test.

## Forward-pass log
- (planned) PASS 1: model=original, items=all, activations avgpool, layer4.1, layer4.0, layer3 (pooled). No edits.
- PASS 1 done (work/fwd_1): original on all. acc g0..g3 = 99.36/82.62/72.93/96.24, wmean 97.72. margins: mean(sd) g0 -13.73(4.17) g1 -4.94(5.57) g2 2.99(5.66) g3 10.34(4.34).
  => bird effect ~16.0, background effect ~8.1 margin units, midpoint -1.33.

## Diagnosis (all-data stats of pass 1, pooled activations)
- cos(d_bg,d_bird) by layer: layer3 .04, layer4.0 .08, layer4.1 .30, avgpool .57. bg-only channels: 225/496/315/35. => entanglement created by layer4.1 and layer4.2 branches.
- logit lens w.d_bg / w.d_bird: layer4.0 1.72/0.54 ; layer4.1 3.34/2.05 ; avgpool 8.08/16.0.
- avgpool: plain fc projection off d_bg would cut bird effect 16 -> ~8.9 (cos .57). PC1(21%)=bird+bg, PC2(6%)=bird-bg.
- failures viewed: bg wins (mallard/loon on forest; landbirds on ocean), tiny birds, some label quirks (pewee/towhee labelled waterbird).

## Split
- DEV/HOLD stratified, seed 20261122, saved work/dev_mask.npy + work/split.json. DEV 234/233/67/66, HOLD 233/233/66/67.
- RULE: all edit statistics and all candidate choices from DEV only. HOLD only for final confirmation.

## Helper definitions (rebuild if kernel restarts)
- bg_dir(X,msk): precision-weighted (by group n) mean diff water-land within bird class; centre = mean of 4 group means.
- read_side_projection(block, dirs, center, alpha): conv1.weight <- W - alpha*W Q Q^T ; bn1.running_mean <- rm - alpha*W Q Q^T center.

## Edits / passes
- (planned) PASS 2: E1 = read_side_projection('layer4.1', unit(d_bg of layer4.0 pooled, DEV, from fwd_1), centre c40, alpha=1). fc untouched.
  model=both, items=DEV, activations layer4.1 + avgpool.
- PASS 2 done (work/fwd_2, DEV): E1 only. DEV acc orig 99.6/82.8/76.1/98.5 -> E1 99.1/86.3/86.6/97.0 (g1: 8 w->r, 0 r->w; g2: 7 w->r, 0 r->w).
  margins DEV: orig D=16.28 B=8.10 sd=4.71 mid=-1.21 ; E1 D=16.03 B=5.49 sd=4.47 mid=-0.53.
  layer4.1 output: w.d_bg 3.44 -> 1.97 (skip-path level), w.d_bird 2.00 -> 2.07. => layer4.1 branch's bg contribution removed, bird intact.
  directions saved: work/dir_E1_layer4.0_bg.npy (unit dir, centre), work/dir_E2_layer4.1_bg.npy.
- (planned) PASS 3: E1 + E2 = read_side_projection('layer4.2', unit(d_bg of layer4.1 pooled under E1, DEV, from fwd_2 edited), centre, alpha=1). fc untouched.
  model=both, items=DEV, activations avgpool (+layer4.1).
- PASS 3 done (work/fwd_3, DEV): E1+E2, fc untouched. DEV acc 98.7/89.3/91.0/97.0 (orig 99.6/82.8/76.1/98.5). paired g1 +16/-1, g2 +10/0, g0 -2, g3 -1.
  margins DEV: D=15.29 B=2.53 sd=4.26 mid=0.00; group sds 3.28/5.22/4.24/4.08 (g1 landbird/water has widest spread -> limiting group).
  avgpool after E1+E2: cos(d_bg,d_bird) .29 (orig .54). Stage copies: work/stage_E1E2, work/stage_E1E2E3.
- Frame-swap set: work/fs/index.csv (252 composites: 63 DEV centres x 4 donor frames [2 water, 2 land], centre box 128x128 = [48:176] crop coords; + 4 identity checks), packed in work/fs/all.npy (normalized inputs).
- (planned) PASS 4: E1+E2+E3. E3 = fc_mean_ablate(unit(d_bg avgpool under E1+E2, DEV, from fwd_3 edited), centre) on both fc rows (+bias).
  model=both, items=DEV, images=work/fs/all.npy, activations layer3, layer4.0, layer4.1, avgpool. Request saved in work/req_pass4.json.
- PASS 4 done (work/fwd_4, DEV + 256 frame-swap): E1+E2+E3. DEV acc 97.0/88.8/95.5/97.0 (worst 88.8 g1, wmean 96.67). margins D=13.57 B=0.14 sds 3.21/5.24/3.35/3.77 mid 0.00.
  identity checks OK (my npy inputs == harness to 0.002). Frame-swap: water-minus-land frame effect orig +7.32+-0.35 -> edited -0.09+-0.29; pred flips between frame types 44% -> 17% of centres.
  BUT same-type frame-to-frame margin sd unchanged (3.04 -> 2.89; water frames 3.2 -> 3.4).
- KEY FINDING (nonlinearity): at layer4.0, bg score s0=(x-c40).d40hat: land scenes -1.5 (sd .6), water +1.5 (sd 1.5). Original margin slope on s0 within groups 2.8/0.42/2.68/0.53 (saturated in water).
  After E1+E2+E3 slopes -0.17/-1.40/-0.09/-1.05 => linear cut OVER-CORRECTS in water scenes (g1 sd stays 5.2).
  Second bg direction wPC1 = top PC of water-scene within-group residuals at layer4.0 after removing d40 (var water 1.40 vs land 0.17; frame-driven; corr with s0 within water .8).
  w reads it with opposite sign: w.wPC1=-0.51 vs w.d40hat=+0.58; edited margin corr with wPC1 within g1/g3 = -0.52/-0.50. Persists: cos(layer4.0 wPC1, layer4.1 wPC1)=.89.
- (planned) PASS 5: E1p+E2p (2-D cut). E1p = read_side_projection('layer4.1',[d40hat, wPC1_layer4.0], c40); E2p = read_side_projection('layer4.2',[d41e_hat, wPC1_layer4.1(E1 model)], c41e). fc ORIGINAL.
  dirs saved work/dir_E1p_layer4.0.npy, work/dir_E2p_layer4.1.npy. model=both, items=DEV, images=work/fs/all.npy, activations layer4.1, avgpool.
- PASS 5 done (work/fwd_5, DEV + frame-swap): E1p+E2p (2-D read cuts), fc original. DEV acc 98.3/89.3/92.5/97.0. margins: means -7.61/-5.12/6.30/8.73 sds 3.06/4.15/4.03/3.67, D=13.88 B=2.46 D/sd=3.70 mid=+0.58.
  slopes of margin on s0 within groups 0.24/0.22/0.56/0.09 (1-D cut had 0.52/-0.81/0.61/-0.51) => over-correction gone; g1 sd 5.22->4.15. Frame-swap same-type sd 3.04(orig)->2.37.
  At avgpool, 'top water PC' = class axis (cos .85 with bird) -> NEVER cut it at fc. Inherited wPC1 at avgpool is bird evidence too -> fc cut must be 1-D (mean bg dir only).
- (planned) PASS 6: E1p+E2p+E3p. E3p = fc_mean_ablate(unit(d_bg avgpool under E1p+E2p, DEV, fwd_5 edited), centre) + bias shift so DEV midpoint=0 (margin shift -0.58).
  stage copies: work/stage_E1pE2p, work/stage_E1pE2pE3p. model=both, items=DEV, images=work/fs/all.npy, activations avgpool.
- PASS 6 done (work/fwd_6, DEV + frame-swap): E1p+E2p+E3p (2-D read cuts + 1-D fc cut + midpoint 0). DEV acc 97.0/93.6/95.5/97.0 (worst 93.6, wmean 96.85); paired g1 +26/-1, g2 +13/0, g0 -6, g3 -1.
  margins: means -5.94/-6.03/5.83/6.14 sds 2.91/3.87/3.09/3.11 D=11.97 B=0.11 D/sd 3.67; z by group 2.04/1.56/1.89/1.98. Frame-swap type effect 0.22+-0.21, same-type sd 2.13 (orig 3.04).
  No further bg direction stands out at layer4.0 (R^2 of margin on next PCs ~ chance). => stop at 2 directions.
- RECIPE (frozen before looking at HOLD), strictly sequential:
  site1 layer4.1.conv1(+bn1.running_mean): cut {d_bg, wPC1} of layer4.0 output (original model stats), mean-ablation at balanced centre.
  site2 layer4.2.conv1(+bn1.running_mean): cut {d_bg, wPC1} of layer4.1 output, stats under the site1-edited model.
  site3 fc.weight/fc.bias: cut {d_bg} of avgpool, stats under site1+2-edited model; then bias shift so midpoint of 4 group-mean margins = 0.
  d_bg = precision-weighted (water - land) mean diff within bird class; wPC1 = top PC of water-bg within-group residuals after removing d_bg; centre = mean of 4 group means.
- (planned) PASS 7: DEV-derived sequential E1p + E2pp (E2pp from fwd_5 edited_layer4.1), fc original. model=both, items=DEV, activations avgpool, layer4.1. stage copy work/stage_seq_E1pE2pp.
- PASS 7 done (work/fwd_7, DEV): sequential E1p+E2pp, fc original. DEV acc 97.4/89.3/95.5/97.0; margins D=12.99 B=2.19 sd 3.58 D/sd 3.63 mid +1.00; slopes on s0 flat (0.05/0.16/0.56/0.16).
  (non-sequential pass 5 had D/sd 3.70: same quality.) wPC1 definition checked vs top eigvec of (Sigma_water-Sigma_land): cos .99/.97; guard: need water/land var ratio >> 1.
- (planned) PASS 8 = HOLD CONFIRMATION: DEV-derived sequential full edit (E1p + E2pp + E3pp[fc cut of avgpool d_bg from fwd_7 + midpoint shift -1.00]); copy work/stage_seq_full_DEV.
  model=both, items=all, images=work/fs/all.npy, activations avgpool, layer4.1. Analyse DEV and HOLD separately.
- PASS 8 done (work/fwd_8, ALL + frame-swap) = HOLD CONFIRMATION of DEV-derived sequential full edit (work/stage_seq_full_DEV):
  ALL  orig 99.4/82.6/72.9/96.2 (worst 72.9, wmean 97.72) -> edited 97.0/93.1/91.7/93.2 (worst 91.7, wmean 95.96); flips g0 -11, g1 +51/-2, g2 +25/-0, g3 -4.
  DEV  edited 97.0/93.6/95.5/97.0 ; HOLD orig 99.1/82.4/69.7/94.0 -> edited 97.0/92.7/87.9/89.6 (worst 87.9, wmean 95.08); HOLD flips g1 +25/-1, g2 +12/-0, g0 -5, g3 -3.
  HOLD margins: B 8.05 -> 0.59, D 15.72 -> 10.97, sd 5.25 -> 3.57. ALL z by group 1.92/1.54/1.52/1.74 (threshold balanced).
  => CONFIRMED but worst-group margin over 88 is thin (HOLD waterbird groups harder). This is the FALLBACK edit (re-derive on all data with same recipe).
- IDEA (source cut): bird evidence is lost downstream because branch1 writes bird evidence on the bg axes. Cut bg where it enters layer4: layer4.0.conv1 + layer4.0.downsample.0 read layer3.
  layer3 has same 2 dirs: d_bg (|.|=.55, cos with bird -.13) and wPC1 (water var .05 vs land .01). corr(layer3 score, layer4.0 score) = .99 (within groups .96-.99); R^2 .93-.97.
- (planned) PASS 9: S0 only = cut {d_bg, wPC1} of layer3 (DEV stats, fwd_1) from layer4.0.conv1(+bn1.running_mean) and layer4.0.downsample.0(+downsample.1.running_mean). fc & others original.
  dirs work/dir_S0_layer3.npy; stage work/stage_S0_DEV. model=both, items=DEV, images=work/fs/all.npy, activations layer4.0, layer4.1, avgpool.
- PASS 9 done (work/fwd_9, DEV + frame-swap): S0 only. DEV acc 99.1/92.3/86.6/97.0 (wmean 98.26); flips g1 +22/-0, g2 +7/-0, g0 -1, g3 -1.
  margins: means -10.81/-7.59/5.89/8.05 sds 3.80/5.04/4.72/4.04; D=16.17 (PRESERVED) B=2.69 (from 8.10) D/sd 3.65 mid -1.12. frame-swap type effect 7.32 -> 2.21.
  layer4.0 output under S0: |d_bg| 3.02 -> 1.37 (cos .52 with orig), w.d_bg 1.76 -> 0.03; layer4.1 w.d_bg 3.44 -> 0.66. Residual dir orthogonal to bird (-.05), no clean water-PC (guard fails) -> 1-D.
  NOTE: D/sd of S0 (3.65) ~ downstream recipe (3.61): preserving D alone is not a win; need residual B removed without losing D/sd.
- (planned) PASS 10: S0 + R1 (cut residual d_bg of layer4.0-under-S0 [fwd_9 edited_layer4.0, DEV] at layer4.1.conv1 + bn1.running_mean). fc original. stage work/stage_S0R1_DEV.
  model=both, items=DEV, images=work/fs/all.npy, activations layer4.1, avgpool.
- PASS 10 done (work/fwd_10, DEV + frame-swap): S0 + R1 (1-D residual cut at site 1). DEV acc 99.1/93.1/88.1/97.0. margins D=16.11 B=1.62 (from 2.69) sds 3.73/5.20/4.64/4.29 D/sd 3.58 (S0: 3.65).
  Over-correction AGAIN in water scenes: slopes on residual score -0.19/-1.43/-0.23/-1.50; frame same-type sd 2.85 -> 3.01. => residual also has chord(+)/tangent(-) structure.
- DEFINITION (generalises wPC1): tangent u = unit( Cov_water(x, bg score) minus its d_bg component ), computed on within-group residuals of water-bg images.
  Validated: cos(u, wPC1) = .98 (layer3), .98 (layer4.0), .94/.96 (layer4.1). Under S0 at layer4.0: var water .31 vs land .07, w.u = -0.62, S0+R1 margin corr with it in water groups -.36/-.35.
- (planned) PASS 11: S0 + R1' (cut {d_bg residual, tangent} of layer4.0-under-S0 at layer4.1.conv1). fc original. stage work/stage_S0R1p_DEV; dirs work/dir_R1p_layer4.0_underS0.npy.
  model=both, items=DEV, images=work/fs/all.npy, activations layer4.1, avgpool.
- PASS 11 done (work/fwd_11, DEV + frame-swap): S0 + R1' (2-D). DEV acc 99.1/92.3/88.1/97.0; D=15.97 B=2.00 sds 3.72/4.91/4.60/4.10 D/sd 3.66 mid -0.98; over-correction gone (slopes -0.44/-0.72 in water).
  Assessment: route B (source cut) and route A (downstream recipe) converge to the same z-profile (~2.1/1.6/1.7/1.9 on DEV); limit is bird evidence SNR (D/sd ~3.6-3.7 in all variants).
  HOLD waterbird errors of A viewed: ~6/15 label-quirk songbirds, 2 degenerate composites, rest real waterbirds with margins just below 0.
- (planned) PASS 12: route B = S0 + R1' + R2' (cut {d_bg, tangent} of layer4.1-under-S0+R1' [fwd_11] at layer4.2.conv1). fc ORIGINAL (no bias step). stage work/stage_B_DEV.
  model=both, items=all, images=work/fs/all.npy, activations avgpool, layer4.1. Compare with A (pass 8) on DEV and HOLD by threshold-free stats.
- PASS 12 done (work/fwd_12, ALL + frame-swap): route B = S0+R1'+R2', fc original. ALL acc 97.9/91.4/91.0/93.2 (worst 91.0, wmean 96.51).
  A vs B threshold-free: D/sd identical (DEV 3.61, HOLD 3.08, ALL 3.32 for both); margins corr .92-.98 within groups; residual bg effect A 0.40 (HOLD 0.59) vs B 1.13 (HOLD 1.39).
  centred z ALL: A 1.90/1.53/1.54/1.76, B 2.13/1.41/1.48/1.77. => KEEP ROUTE A (simpler, confirmed, more complete). Route B = robustness evidence only.
- FINAL = recipe A re-derived on ALL 1199 images (same frozen recipe). Step files: work/final_site1_layer4.0.npy (d_bg, wPC1, centre); stage work/stage_final_site1.
- (planned) PASS 13: final site 1 only (layer4.1.conv1 + bn1.running_mean, all-data stats from fwd_1 layer4.0). model=both, items=all, activations layer4.1, avgpool.
- PASS 13 done (work/fwd_13, ALL): final site 1 only. ALL acc 98.7/86.7/83.5/96.2; flips g1 +20/-1, g2 +14/-0, g0 -3, g3 0. (margin stats printed in run 53)
  site-1 dirs (all data) vs DEV: cos .998/.990; DEV-only vs HOLD-only: .991/.963. wPC1 guard ratio 7.4.
- (planned) PASS 14: final site 1 + site 2 (layer4.2.conv1 + bn1.running_mean; dirs from fwd_13 edited_layer4.1, all data; saved work/final_site2_layer4.1.npy). fc original. stage work/stage_final_site12.
  model=both, items=all, activations avgpool.
- PASS 14 done (work/fwd_14, ALL): final sites 1+2, fc original. ALL acc 97.2/89.5/90.2/94.7. (margins printed in run 55)
- FINAL EDIT A(all) built: + site 3 = fc_mean_ablate(avgpool d_bg under sites 1+2, all data) + bias midpoint shift. Files work/final_site3_avgpool.npy, work/final_midpoint.json.
  Copies: work/stage_final_A_all and work/best_1 (6 tensors: layer4.1.conv1.weight, layer4.1.bn1.running_mean, layer4.2.conv1.weight, layer4.2.bn1.running_mean, fc.weight, fc.bias).
- (planned) PASS 15: FINAL A(all) check. model=both, items=all, images=work/fs/all.npy, activations avgpool.
- PASS 15 done (work/fwd_15, ALL + frame-swap): FINAL A(all) [work/best_1 == work/stage_final_A_all]. ALL acc 96.4/93.1/92.5/94.0 (worst 92.5, wmean 95.67); flips g0 -14, g1 +51/-2, g2 +26/-0, g3 -3.
  margins ALL: means -5.76/-6.07/5.36/6.46 sds 3.05/3.98/3.37/3.57 D=11.83 B=0.39 D/sd 3.37 mid 0.00; centred z 1.89/1.52/1.59/1.81. halves: DEV 97.0/93.6/95.5/97.0, HOLD 95.7/92.7/89.4/91.0 (wmean 94.49).
  stage-wise ALL: B 8.08 -> 5.48 -> 2.30 -> 0.39 ; D 16.0 -> 15.5 -> 13.2 -> 11.8 ; worst 72.9 -> 83.5 -> 89.5 -> 92.5. frame-swap type effect 7.32 -> 0.48+-0.21.
  Decision: this is the submission candidate. Remaining passes: control (random dirs), dose alpha 0.5 / 1.5, reverse fold, final re-check. ALWAYS restore work/edited from work/best_1 at the end.
- (planned) PASS 16: CONTROL = random data-covariance directions (2,2,1; orthogonal to bg dirs; seed 123) cut at layer4.1.conv1, layer4.2.conv1, fc with same centres; no midpoint shift. items=all + frame-swap.
  (control redesigned before running: top-2 within-group PCs orthogonal to bg dirs at the two read sites, fc original; dirs work/control_dirs.npy; stage work/stage_control_toppc. Compare with pass 14.)
- PASS 16 done (work/fwd_16, ALL + frame-swap): CONTROL (top-2 non-bg within-group PCs cut at both read sites, same procedure, weight change 6.0%/5.0% vs real 5.7%/5.3%).
  ALL acc 99.4/84.8/72.9/94.7 (orig 99.4/82.6/72.9/96.2): worst group unchanged; bg effect B 8.33 (orig 8.08; real sites 1+2: 2.30); D 17.66; frame-swap type effect unchanged, 4/256 preds changed.
- (planned) PASS 17: dose alpha=0.5 (final all-data dirs at half strength at all 3 sites, bias terms scaled by alpha). stage work/stage_dose_0.5. items=all + frame-swap.
- PASS 17 done (work/fwd_17): alpha=0.5. ALL acc 98.7/88.2/85.7/94.7; B=3.61 D=13.84 sd=4.08 mid=-0.43.
- (planned) PASS 18: dose alpha=1.5 (stage work/stage_dose_1.5). items=all + frame-swap.
- PASS 18 done (work/fwd_18): alpha=1.5 -> shortcut REVERSED. ALL acc 89.7/95.5/94.0/91.0.
  DOSE-RESPONSE (ALL): alpha 0 / 0.5 / 1 / 1.5 : bg effect [8.08, 3.61, 0.39, -2.02] ; frame-swap effect [7.32, 3.34, 0.48, -1.73] ; worst acc [72.9, 85.7, 92.5, 89.7] ; wmean [97.72, 97.28, 95.67, 90.27].
  work/edited restored to final edit (== work/best_1).
- REVERSE FOLD (procedure validation; NOT the submission): recipe A derived from HOLD only, evaluated on DEV. Files work/rev_site*.npy, stages work/stage_rev_*.
- (planned) PASS 19: rev site 1 (HOLD stats). items=all, activations layer4.1.  PASS 20: + rev site 2 (activations avgpool).  PASS 21: + rev site 3 + midpoint (HOLD) -> evaluate DEV rows; + frame-swap.
  AFTERWARDS RESTORE work/edited FROM work/best_1.
- PASSES 19-21 done: REVERSE FOLD (HOLD-derived recipe). On DEV (out-of-sample): acc [97.0, 92.3, 95.5, 97.0]. CROSS-VALIDATED over all 1199 (each image by the model not using it): acc [97.0, 92.5, 91.7, 93.2], worst 91.7, wmean 95.94; in-sample halves [96.4, 92.3, 93.2, 94.7].
  Frame-swap (out-of-sample for HOLD-derived model): type effect ~0 (see run 65). work/edited restored to work/best_1.
- (planned) PASS 22: BASELINE fc-only (original net; fc mean-ablation of avgpool d_bg [all data, fwd_1] + midpoint zero). stage work/stage_baseline_fc_only. items=all + frame-swap. NOT the submission; restore best_1 after.
- PASS 22 done (work/fwd_22): fc-only BASELINE. ALL acc 95.1/89.1/94.7/94.0 (worst 89.1, wmean 94.60). margins D=8.92 B=-0.01 sd=2.69 D/sd=3.32; centred z [1.83, 1.32, 1.83, 1.89] (final: [1.89, 1.52, 1.59, 1.81]).
  work/edited restored to work/best_1.
- (planned) PASS 23: spatial maps. model=original, 32 items (8/group), activations layer3 + layer4.0, spatial=true. Purpose: project positions on bg directions (d_bg, wPC1) -> background-region driven?
- PASS 23 done (work/fwd_23): spatial maps, 32 images. Along d_bg the water-minus-land contrast is ~3x larger at the map border than at the centre (layer3 .56 vs .16; layer4.0 3.18 vs 1.01); bird contrast at centre ~0.
  Overlays (work/maps/*.png): d_bg = open sky/horizon + water (red) vs vegetation (blue), bird untinted; wPC1 = sky (red) vs water surface (blue). w reads d_bg (+) and wPC1 (-): counts water surface, discounts sky.
- (planned) PASS 24: ABLATION final edit without the second direction (1-D cuts at both read sites; same fc cut; no midpoint shift). stage work/stage_ablation_1D. items=all + frame-swap. Then RESTORE best_1.
- (planned) PASS 25: FINAL re-check of work/edited == work/best_1. items=all + frame-swap.
- PASS 24 done (work/fwd_24): ABLATION without second direction. ALL acc 97.0/89.3/92.5/94.0 (worst 89.3); centred z [1.94, 1.3, 1.71, 1.81] vs final [1.89, 1.52, 1.59, 1.81]; g1 sd 5.25 vs 3.98. => second direction worth +3.9 pts on landbird/water.
  work/edited restored to work/best_1 (verified identical). NEXT: PASS 25 = final re-check (model=both, items=all, images=work/fs/all.npy).
- PASS 25 done (work/fwd_25): FINAL re-check == pass 15 exactly. ALL acc 96.4/93.1/92.5/94.0, worst 92.5, wmean 95.67. Budget exhausted (25/25).
## FINAL STATE
- SUBMISSION = work/edited == work/best_1 == work/stage_final_A_all (6 tensors): layer4.1.conv1.weight, layer4.1.bn1.running_mean, layer4.2.conv1.weight, layer4.2.bn1.running_mean, fc.weight, fc.bias.
- Honest (2-fold cross-validated) estimate: 97.0/92.5/91.7/93.2, worst 91.7, wmean 95.9.
