# V.2 GATE: harness and self-test — PASS

Run 2026-11-21 on the laptop. No Fable work session; one dry-run driver session (lock-down check only).

## What was built (`exp_v/`)
| file | what it does |
|---|---|
| `common.py` | paths, limits, edit validation (R.2's rule: shape, finite, no whole-tensor constant multiple; nothing frozen), scoring, and the kernel's confinement |
| `kernel.py` | the R.2 kernel API on the broken model: `NAMES`, `SHAPES`, `load`, `load_original`, `save_edit`, `drop_edit`, `edited()`; gradients disabled |
| `server.py` | MCP server with three tools (`python`, `forward`, `show_images`); budget accounting; per-session logs; `finalize`, which scores the submission on the sealed test split |
| `driver.py` | R.4a's driver loop, run locally (no SSH, no judge sidecar); Fable 5.2 added; the task text is built from a template per arm and budget |
| `selftest.py` | the checks below; results in `reports/V.2_selftest.json` |
| `sessions/task_template.md` | the task Fable reads, with arm markers (`<<FIT_RULE>>`, `<<FC_RULE>>`, …) filled in by the driver |
| `sessions/v3_b25_nofit.json`, `v3_b25_fit.json` | pilot configs: Fable 5.2, effort max, budget 25 |

**One forward pass** is one `evaluate` request:
- **Images:** any practice images (up to all 1,199), plus up to 256 of Fable's own images from `work/`.
- **Models:** the original, the edited model, or both on the same images, with a paired comparison.
- **Scores:** per-group accuracy, the worst group and the weighted mean.
- **Activations:** at up to 4 layers. They are pooled to one value per channel for any number of images; full feature maps are available for at most 32.
- **Results:** per-image results and activations are written to `work/fwd_<k>/`.
- **Cost:** a failed request costs nothing.

**`show_images`** returns up to 16 images per call, as the model sees them, with optional crop boxes. It is free, with a cap of 400 per session.

**Session layout:** `~/wst-v/sessions/<sid>/`.
- The kernel's current directory is `sandbox/`, with `model/` and `practice/` (read-only symlinks) and `work/` (read-write).
- The logs sit next to `sandbox/` and are not readable by the kernel.
- The model view is `~/wst-v/model/model.safetensors`: the broken model in float32, 267 tensors, SHA-256 `18c926a5…ebc672f54`.

## Self-test (`python -m exp_v.selftest`, 65 s): ALL PASS
| check | result |
|---|---|
| 1a. server = `exp_v.wb` on the original model | 1,199/1,199 same predictions; logits and avgpool features identical (max diff 0.0); same scores |
| 1b. server's edited model = independent computation (features @ fc) | 600/600 same predictions; max logit difference 0.0012 (GPU vs. numpy rounding; logits up to 4.6) |
| 1c. own images in a pass | works |
| 1d. `finalize` test score = independent computation | identical |
| 2a/2b. split guard, from inside the kernel, both arms | test manifest, raw dataset, a test image, the broken checkpoint, the repo, `~/.claude`, the session logs: all blocked. Writes outside `work/` blocked. No sockets, no CUDA, no torchvision. scikit-learn blocked in the no-fitting arm, available in the fitting arm. The kernel sees only `model`, `practice`, `work` |
| 2c. split hashes (`exp_v.data check`) | practice = validation; 0 overlap with test or train |
| 3. **practice predicts test** (amendment 4) | DFR fitted in the kernel on half A of practice: **87.0** worst-group on practice half B, **89.4** on test (gap 2.4, target ≤ 5); weighted mean 96.3 vs. 95.0 |
| 4. `show_images` | 16 images returned; 17 refused; session cap enforced; crops and `work/` files work; other paths refused |
| 5. budget and refusals | a failed request costs nothing; the budget is enforced; `save_edit` refuses a constant multiple |
| 6. MCP over stdio | exactly `forward`, `python`, `show_images`; images arrive as image content |

**Driver dry run** (`v3_b25_nofit_dry_20261121T170043Z`, Claude Code 2.1.330):
- **Init check:** passed. The only tools were the three harness tools, the only server was the harness, and the model was `claude-fable-5-2`.
- **Fable's actions:** it called `python` (`len(NAMES)` → 267) and `show_images`, and described `p0000` correctly ("a large brown albatross-like seabird pasted over an ocean-coast background").
- **Cost:** 13 s, $0.11 notional.

## Deviations from R, recorded as design amendments 10–13
- **Sandbox without a user switch.** The laptop has no root, so the kernel runs as the session user, and Landlock (ABI 3) plus seccomp do the confinement. Check 2 shows this blocks everything the R pods' user switch blocked. `/etc` and `/proc` stay readable, as on the pods.
- **The fitting arm may run fc on saved features.** DFR-style fitting needs to score candidate last layers on saved avgpool features. The fitting arm allows this; the no-fitting arm does not (rule b: an fc edit is seen only through a forward pass).
- **The task states the target:** worst-group at least 88% with weighted mean at least 94%. It also gives the original's practice numbers and the mean's weights. It does not name DFR or any published method.
- **Activations for all images:** pooled activations are available for every practice image in a pass, not only 256 as the design said. Full feature maps stay limited, to 32 images.

## Files
- **Code:** `exp_v/{common,kernel,server,driver,selftest}.py`, `exp_v/sessions/`.
- **Self-test results:** `reports/V.2_selftest.json`.
- **Self-test and dry-run sessions:** `~/wst-v/sessions/` (not in git).

## Next
**The V.3 gate:** one pilot session at budget 25, in the no-fitting arm.
- **Before it runs:** the user reads the task text, rendered at `exp_v/sessions/task_template.md` with the no-fitting arm text filled in.
- **What it measures:** time, Max usage and the result.
- **The decision after it:** the grid and the fitting arm's run counts.
