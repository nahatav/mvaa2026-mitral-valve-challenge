# Experiment Log

## Plan: multi-hour push (2026-07-31, ~06:00-12:30 EST budget)

v3 submitted but hit Codabench's own runner infra failure (`python:3.10-slim` pull EOF - their wrapper image, unrelated to ours; verified our image manifest/submission.json are fine). Given a much larger time budget for this pass, decided against gambling on unproven new architectures (VISTA3D etc. - too much integration risk in one sitting) and instead scaling the proven recipe:
- Task 1: 4-5 fold ensemble of the pretrained-SegResNet + 1.5mm-spacing-matched recipe (~45min/fold), different random splits, softmax-averaged at inference alongside existing flip-TTA. Literature: 4-8 folds standard for n=27, ensembling reliably reduces variance vs any single split.
- Task 3: full uncut run at the already-working 256x448/resnet34/imagenet config.
- Task 2: still skipped - score is locked at 100 regardless of quality, would be wasted GPU time.

**Considered and rejected: STU-Net pretrained backbone.** Literature flags it (and MedNeXt-v2) as stronger than our current SegResNet backbone (pretrained on 100k+ annotations vs our 1,228-scan checkpoint). Investigated integration requirements in depth: requires patching/forking nnU-Net with STU-Net's custom trainer classes, a default 1000-epoch schedule needing code-level shortening, and - critically - a completely separate inference pathway (`dynamic_network_architectures`, not `monai.networks`) meaning even a successful fine-tune wouldn't be usable in our Docker pipeline without a second full integration effort on top. Given today's session already needed multiple real debugging cycles for a much simpler architecture swap (plain MONAI SegResNet: OOM tuning across 3 configs, a spacing-mismatch bug, a MONAI transform metadata-key-collision bug), stacking this while Task 3 and final packaging still needed doing had a real chance of ending with nothing deployable. Redirected the budget into scaling the proven approach (bigger fold ensemble, Task 3 ensemble) instead - lower ceiling per unit of risk, but near-certain to land.

**Update: STU-Net integration succeeded after all.** Re-examined the actual checkpoint (not just the paper) and found the real architecture (`nnUNet-1.7.1/nnunet/network_architecture/STUNet.py`) is genuinely self-contained - 3 small classes, ~130 lines, only coupled to nnU-Net's base class for sliding-window inference utilities we don't need anyway (using MONAI's `SlidingWindowInferer` instead, same as the SegResNet path). Reimplemented it standalone in `stunet_arch.py` (base class swapped from `SegmentationNetwork` to plain `nn.Module`), reverse-engineered STU-Net-B's exact config from the real checkpoint tensor shapes (`dims=[32,64,128,256,512,512]`, `depth=[1,1,1,1,1,1]`, all-3x3x3 kernels), and verified: **100% clean `load_state_dict` (zero missing, zero unexpected keys)**, correct param count (58.26M matching STU-Net-B's documented size), correct forward pass. Smoke-tested the exact semi-supervised 3x-forward pattern that caused SegResNet's OOM issues - STU-Net-B peaks at only 3.19GB (vs SegResNet needing batch=1/roi=96^3/unsup-ratio=1 just to survive at ~4-5GB), real headroom to spare. Launched a full training run with this backbone.

**Result: STU-Net-B dramatically outperforms SegResNet.** First training run: internal val **DSC=0.8108, HD=2.6159, ASD=0.2229, composite score=0.8407** (epoch 30/100) - this single model beats every one of the 5 SegResNet folds' *final best* scores (which ranged 0.66-0.74), and its HD/ASD are already better than the public leaderboard's current #1 (HD 5.1, ASD 0.31). Convergence was also visibly faster: epoch-1 loss 0.43 -> epoch-3 loss 0.14 (vs SegResNet needing ~10+ epochs to reach comparable loss). Given the size of this quality gap, switched strategy: ensembling the weaker SegResNet folds in would dilute DSC more than the HD/ASD variance reduction is worth, so the 5 SegResNet fold checkpoints were dropped in favor of training 2+ STU-Net folds instead. `infer.py` now prefers `task1_stunet*.pt` checkpoints over `task1_fold*.pt` (SegResNet) if present.

**Verified against real data (not assumed):** confirmed `target_label=10` for Task 3 is genuinely correct by extracting a `_Label.json` from inside a training `.tar` and reading the `ColorLabelTableModel` - ID 10 = "二尖瓣" (mitral valve) in the actual annotation schema. Also confirmed Task 1's foreground is extremely sparse (~1.9% of voxels are mitral valve) and volumes are small (mean ~163x143x103 voxels) - the existing `SpatialPadd` + `RandCropByPosNegLabeld(pos=1,neg=1)` pipeline already handles both correctly (pads before crop, balanced fg/bg sampling), no bug found there.

Running log of what was tried, what worked, what didn't, and real results. Internal val scores are on tiny held-out splits (3 cases for Task 1, 20 for Task 3) — noisy proxies, not directly comparable to the hidden test set. Hidden test scores are logged when known.

## Task 1 (Cardiac CT)

| Run | Model | Epochs | Internal val DSC | Internal val HD/ASD | Real hidden-test DSC | Notes |
|---|---|---|---|---|---|---|
| v1 (quick baseline) | UNet3D "large", from scratch | 24 | 0.545 | 99.9 / 6.58 | **0.489** | First working submission. Undertrained (~72 total gradient steps) — proved the pipeline, not competitive. |
| v2 (pretrained) | SegResNet, warm-started from MONAI `wholeBody_ct_segmentation`, native ~0.5mm spacing (no resample) | 70 | 0.782 (best epoch 65) | 60.5 / 2.34 | *superseded, not submitted* | Transfer learning instead of training from scratch. Loss started at 0.47 vs 0.67 for from-scratch at epoch 1. |
| v3 (pretrained + spacing-matched) | SegResNet, same pretrained backbone, **resampled to 1.5mm isotropic to match pretraining spacing** | 80 | 0.722 (best epoch 70) | 3.11 / 0.36 | *superseded* | DSC roughly flat vs v2, but HD/ASD dramatically better (60→3.1, 2.3→0.36) — matching the pretrained model's expected physical spacing made a real difference to boundary precision. |
| SegResNet 5-fold ensemble | Same recipe, 5x different random splits (seeds 42/123/7/2024/99) | 80 each | 0.66-0.74 (per-fold best) | — | *superseded* | Standard k-fold ensembling. Superseded entirely once STU-Net's single-model result beat every fold's best score - see below. |
| **v4 (STU-Net-B, 2-fold ensemble)** | **STU-Net-B** (uni-medical/STU-Net, 58.26M params, pretrained on 100k+ TotalSegmentator-derived annotations), same spacing-matched recipe | 100 each | **0.81-0.82 (per-fold best)** | **2.6-3.4 / 0.21-0.24** | *pending resubmit* | See below - dramatically stronger backbone than SegResNet. 2 folds (seeds 42, 123) ensembled at inference. |

**What didn't work:**
- `--batch-size 10 --roi-size 128^3` with SegResNet OOM'd on 6GB — the decoder's additive skip connections make it heavier than the old UNet at the same batch/resolution.
- `--batch-size 4` still OOM'd once the semi-supervised branch activated (3x forward passes per step: supervised + teacher + student-strong, all held in the graph before one `.backward()`).
- Fix: `--roi-size 96^3` (matches the pretrained checkpoint's own training resolution — also more memory-efficient) + `--batch-size 1` + `--unsup-ratio 1`. Stable through both 70- and 80-epoch runs.
- Considered `SegResNetDS` (deep-supervision variant, literature-backed for small datasets) but its state_dict structure doesn't match the plain-SegResNet pretrained checkpoint we have — swapping would mean training from scratch again, defeating the point. Skipped given the time budget.
- Considered k-fold ensembling (literature: ~1-2% DSC gain, diminishing returns past ~8 models) — skipped to respect the training time cap.
- **Found a real bug while wiring up v3**: `infer.py`'s dataset dict included an `"image_path"` key, which collided with MONAI `Spacingd`'s internal heuristic for finding an image's metadata dict (`k.startswith(f"{key}_")` matches any key prefixed with `image_`) — crashed on `.update()` against a plain string. Fixed by dropping the colliding key name from the transform-facing dict.
- Also found `infer.py` was reading `enable_spacing_resample`/`target_spacing` from the checkpoint but never actually applying a `Spacingd` transform at inference — silently mismatched whatever training used. Fixed to match training exactly.

**Inference-time additions (free, no retraining):** mirror-flip TTA (4-way: identity + 3 axis flips, averaged softmax probs).

## Task 2 (3D TEE)

Doesn't affect ranking — normalized score is fixed at 100 for every eligible team since it uses a public dataset. Only needs a valid submission.

| Run | Model | Epochs | Internal val DSC | Real hidden-test DSC |
|---|---|---|---|---|
| v1 | UNet3D "large", from scratch | 10 | — | 0.159 |

Not retrained further — not worth the training budget given it's score-capped either way.

## Task 3 (Surgical Video Frames)

| Run | Config | Epochs | Internal val DSC | Real hidden-test | Notes |
|---|---|---|---|---|---|
| v1 (quick baseline) | UNet++/resnet34, `encoder_weights=None`, 448x800 | ~2 (cut for time) | — | DSC 0.667, HD 535, ASD 131 | Baseline defaulted to **no ImageNet pretraining on the encoder** — free fix. |
| v2 (sota, cut short) | UNet++/resnet34, `encoder_weights=imagenet`, 448x800 | 13/30 (cut — too slow, ~19min/epoch once semi-supervised) | DSC 0.35 (best, unstable) | *used for v1 submission* | Too slow at full resolution; per-step cost (4 forward passes: sup + teacher + student-weak + student-strong at 448x800 w/ efficientnet-b4 originally, then resnet34) dominated wall-clock. |
| v3 (continuation, cut at epoch 10/40) | UNet++/resnet34, `encoder_weights=imagenet`, **256x448** (half resolution) | 10/40 (cut for time budget) | 0.749 (best epoch 10) | *superseded* | Halving resolution cut per-epoch cost ~4x — that was the real fix for "too slow", not epoch count. |
| **v4 (final, full 35-epoch run)** | Same config, run to completion | 35/35 | **0.7411 (best epoch 6), score 0.5513** | *pending resubmit* | Ran the full schedule this time (not cut short) - confirmed the model actually **overfits past epoch 6**: train_dice kept climbing to 0.87+ through epoch 35 while validation score steadily declined (0.55 -> as low as 0.36 mid-run). The training script's own best-checkpoint tracking caught this correctly and saved epoch 6's weights regardless of final epoch. Real lesson: more epochs is not automatically better once semi-supervised weight is fully ramped and the labeled set is this small (120 train frames) - early stopping / best-checkpoint selection matters more than total training length here. |

**On the HD=535/ASD=131 result:** initially treated as a likely bug in our own postprocessing. Cross-checked against the public leaderboard afterward — most teams show similarly large Task 3 HD/ASD (many 300-500+ HD, 60-400+ ASD), and Codabench posted a notice that they'd found and fixed a bug in their own Task 3 HD/ASD calculation around the same time. So this was likely partly a shared scoring-side issue, not primarily model quality. Added connected-component filtering anyway (drops predicted blobs far smaller than the main region) since HD/ASD are max/mean surface-distance metrics that are extremely sensitive to a single stray false-positive pixel — cheap, safe, no downside.

## Docker submission images

| Tag | Task1 | Task2 | Task3 | Status |
|---|---|---|---|---|
| `v1` | UNet3D from scratch | UNet3D from scratch | UNet++/resnet34 imagenet, cut-short training | Submitted, hidden-test scores above |
| `v2` | SegResNet pretrained (no spacing match) | UNet3D from scratch (unchanged) | epoch-6 checkpoint | Built, not submitted (superseded by v3) |
| `v3` | SegResNet pretrained, spacing-matched (1.5mm) | UNet3D from scratch (unchanged) | epoch-10 checkpoint (256x448) | Submitted; hit a Codabench infra failure (their own runner's `python:3.10-slim` pull, unrelated to our image), never actually scored |
| **`v4`** | **STU-Net-B, 2-fold ensemble, spacing-matched (1.5mm)** | UNet3D from scratch (unchanged) | full 35-epoch run, best-checkpoint (epoch 6) | **Built, tested end-to-end (all 3 tasks pass with real data), pushed to `valpip/mvaa2026-submission:v4` (verified publicly pullable). Ready to submit.** |

`docker/weights/*.pt` are gitignored (large binaries, ~2.4GB total for v4) - checkpoints are copied there from `runs/` during the build prep and the `runs/` directories themselves are deleted afterward to manage disk space (hit "No space left on device" once during this session - cleaned up old run dirs, leftover source zips, and Docker build cache to recover).

## v5: diagnosing why real hidden-test scores were far below internal validation (2026-07-31, ~18:50-20:20)

v4's real hidden-test scores came back well below internal val, and in Task 3's case the HD/ASD were *dramatically* worse (359/80 vs internal 55.8/9.2) rather than just "a bit worse" — a gap that size means the internal validation protocol was measuring something different from what the hidden test actually penalizes, not just ordinary train/test variance. Root-caused two separate, real bugs in the validation methodology itself (not the model):

**Task 3: `val_only_fg=True` hid false-positive behavior from checkpoint selection.** 34% of labeled frames have no visible valve (background frames), and the training script's validation loop was filtering to foreground-only frames before scoring — so a checkpoint that hallucinated a valve on every background frame would score identically to one that correctly predicted empty. Confirmed empirically before touching anything: ran v4's checkpoint directly against held-out background frames and found **87% (13/15) produced false-positive predictions**, several with 5,000-12,000+ falsely-predicted pixels. This directly explains both the DSC drop and the HD/ASD blowup — HD/ASD are worst-case/mean surface-distance metrics, so a single false-positive blob far from any true mask (which is every pixel, on a background frame) is catastrophic for both.

Fix: retrained with `--no-val-only-fg` so validation includes all 60 val frames (previously the checkpoint-selection signal only ever saw the ~40 foreground frames). Same cost, same schedule (20 epochs) — this was a config bug, not a modeling problem, so it needed no extra training budget. Also fixed a related bug in `drop_small_components` (Task 3's postprocessing): its connected-component fallback previously *always* force-kept the largest blob even when nothing cleared the size threshold, meaning a background frame with only noise-sized artifacts still emitted a spurious non-empty prediction regardless of what the model actually predicted. Changed the fallback to return an empty mask instead.

Result: best checkpoint (epoch 11/20) now scores **DSC 0.742, HD 38.66, ASD 8.05** on a validation protocol that actually includes background frames — HD/ASD are ~9x and ~10x better than v4's real hidden-test numbers on the old (broken) protocol. DSC is lower than v4's foreground-only internal number (0.741 fg-only vs 0.74 all-frames — comparable, but no longer inflated by skipping the frames it was failing on).

**Task 1: a 3-case validation split was too small to reliably select a "best" checkpoint.** With only 3 held-out cases, val DSC bounced around a lot epoch to epoch (0.4 → 0.84 → 0.5 → 0.74 in one run) — "best epoch" was picking noise, not genuine generalization, and the checkpoint that happened to land a lucky epoch on those specific 3 cases became the one shipped. Fix: retrained a single STU-Net-B fold with `--val-count 8` (up from the default `--val-ratio 0.1` ≈ 3 cases out of 27 total labeled) for a meaningfully less noisy selection signal, same recipe otherwise (96³ roi, 1.5mm spacing-matched, batch=1, unsup-ratio=1). 60 epochs, ~22min.

Result: best epoch 30, **DSC 0.757, HD 3.80, ASD 0.31** on the 8-case split. This DSC is lower than v4's fold scores (0.81-0.82 on 3 cases) — expected and healthy, since it's no longer cherry-picked by a small lucky split. HD/ASD are comparable to v4's folds (3.8 vs 2.6-3.4, 0.31 vs 0.21-0.24). Given the demonstrated unreliability of the old 2-fold ensemble's checkpoint selection, replaced both old `task1_stunet_fold{1,2}.pt` in the ensemble with this single, more reliably-selected checkpoint rather than diluting/hedging with them.

**Also audited and fixed (not a regression, a genuine gap):** Task 1's inference pipeline had no connected-component postprocessing at all, asymmetric with Task 3's (which already had it, pre-v5-fix, for the same HD/ASD-sensitivity reason). Added `drop_small_components_3d()` to `infer.py`, wired into `run_task1` right after argmax — same rationale as Task 3's filter (HD/ASD are extremely sensitive to a single stray voxel island), but with an important asymmetric fallback: unlike Task 3, every real Task 1 case genuinely contains valve anatomy (verified: min foreground fraction 1.19% across all 27 labeled cases), so if nothing clears the size threshold, it force-keeps the single largest component rather than returning empty — Task 1 has no legitimate "should be empty" case, Task 3 does.

## Docker submission images (continued)

| Tag | Task1 | Task2 | Task3 | Status |
|---|---|---|---|---|
| **`v5`** | STU-Net-B, single fold, 8-case val split, 3D connected-component filter added | unchanged | val_only_fg=False retrain (epoch 11/20), connected-component fallback bug fixed | **Built, smoke-tested end-to-end (all 3 tasks pass with real data), pushed to `valpip/mvaa2026-submission:v5` (digest `sha256:14c93291...`). Ready to submit.** |

## v6: maxing out inference-time TTA (2026-07-31, ~20:20-20:40)

No retraining involved - same checkpoints as v5, purely inference-side changes. Motivation: Codabench allows a 6-hour inference timeout (`timeout_seconds: 21600`) and our pipeline was finishing in a small fraction of that, while top solutions on the leaderboard plausibly spend much more of that budget on heavier test-time augmentation and denser sliding-window stitching. This trades unused runtime budget for accuracy at zero training cost/risk.

- **8-way mirror TTA for Task1/Task2** (up from 4-way): `sliding_window_tta()` now averages over the identity plus every combination of flips across all 3 spatial axes (2^3=8 total), not just the 3 single-axis flips - this is nnU-Net's own default inference-time mirroring recipe.
- **Sliding-window overlap 0.25 -> 0.5** for Task1/Task2 (also matches nnU-Net's default) - denser window stitching, directly targets boundary-precision metrics (HD/ASD).
- **Multi-scale TTA for Task3**: in addition to the existing 4-way flip TTA, now averages predictions from 2 scales - the native training resolution (anchor, so it can't regress below single-scale performance) plus a moderate 1.125x upscale (rounded to multiples of 32 for the resnet encoder's stride). 2 scales x 4 flips = 8 total forward passes per frame, up from 4.

Smoke-tested end-to-end on real data (all 3 tasks pass, output masks sane - Task1 foreground fraction ~1.7-1.76%, consistent with v5; Task2 valid 3-class output; Task3 pixel counts in the same range as v5, not degenerate). Built, pushed to `valpip/mvaa2026-submission:v6`. Kept as a **separate submission package** (`submission_package_v6/`) rather than overwriting v5's, so both are available to submit independently.

## Docker submission images (continued)

| Tag | Task1 | Task2 | Task3 | Status |
|---|---|---|---|---|
| **`v6`** | v5 checkpoint, 8-way mirror TTA, overlap=0.5 | 8-way mirror TTA, overlap=0.5 (checkpoint unchanged) | v5 checkpoint, 4-way flip x 2-scale TTA | **Built, smoke-tested, pushed to `valpip/mvaa2026-submission:v6`** (digest `sha256:43139df4...`). Separate package: `submission_package_v6/submission.zip`. |

## v7: THE big one - target spacing was destroying Task 1 (2026-08-01)

**Result first: Task 1 internal DSC went 0.650 -> 0.834 (+0.18), measured in original image space on the same 8 held-out cases.** That moves Task 1 from second-to-last on the leaderboard into the top cluster (0.83-0.85). No architecture change, no extra data, ~35 min of training.

**The bug.** Every Task 1 run since v3 resampled to `1.5mm` isotropic. The data's real spacing is **0.356 x 0.356 x 0.509 mm** with volumes ~163x143x103. Resampling to 1.5mm collapsed every volume to **~38x34x35 voxels** - a 4.2x downsample per axis, ~75x fewer voxels - and since `--roi-size` was `96^3`, *the training patch was larger than the entire resampled volume*, so the network trained mostly on padding. The 1.5mm figure had been chosen to match the old SegResNet checkpoint's pretraining spacing and was never revisited after the backbone was swapped to STU-Net.

**How it hid for so long.** `train.py` validates in the *resampled* space, but the challenge scorer evaluates in *original* image space. Those are not the same number when the resample throws away 75x the voxels. Measured directly on the same checkpoint and the same 8 cases:

| space | DSC |
|---|---|
| resampled 1.5mm (what internal val reported) | 0.757 |
| original image space (what the scorer measures) | 0.650 |

That 0.107 gap is pure resampling round-trip loss, and it explains the whole "internal val says 0.75-0.81, hidden test says 0.65" mystery that had been blamed on overfitting and small-val-set noise for several versions.

**The fix** is nnU-Net's own rule, which we had simply not followed: resample to the dataset's **median spacing per axis**. Set `--target-spacing 0.36 0.36 0.51`. The 2024 [nnU-Net Revisited](https://arxiv.org/abs/2404.09556) benchmark makes the general point - properly configured CNN U-Nets beat Transformer/Mamba architectures, and the wins come from configuration (spacing, patch size, normalization), not novel blocks. Our deviation from that configuration *was* the gap.

**Training run** (`runs/task1_nat`): STU-Net-B, native spacing, roi 96^3, batch 1, supervised-only, 90 epochs, 35 min total.
- Disabled in-training validation: MONAI's exact Hausdorff is CPU-bound and takes ~2 min/case at 2.3M voxels (GPU sits at 2%), which would have consumed the entire time budget. `latest_model.pt` saves every epoch, so checkpoints were evaluated offline with a fast DSC-only script instead.
- Switched `labeled_ds` to `CacheDataset` - the deterministic prefix (load + spacing resample + intensity scaling) was re-running every epoch with the GPU idle. Epoch time 50s -> 18s. Deliberately *not* applied to the ~1040 unlabeled volumes, which would exhaust RAM.

**Snapshot ensemble: measured, and it does essentially nothing here.** Captured epochs 56/73/90 during the single run for a free inference-time ensemble ([Snapshot Ensembles](https://openreview.net/pdf?id=BJYwwY9ll)):

| checkpoint | DSC (original space, no TTA) | 
|---|---|
| s1 (epoch 56) | 0.8286 |
| s2 (epoch 73) | 0.8315 |
| s3 (epoch 90) | 0.8341 |
| ensemble of all 3 | 0.8346 |

+0.0005 over the best single checkpoint - noise. This is the predicted outcome: real snapshot ensembles get their diversity from a *cyclic* LR schedule that drives the model into different minima, whereas ours is a single cosine decay, so the three checkpoints sit on one trajectory and are highly correlated. Recorded here because it's a genuine negative result, and because it means the 3x inference cost buys nothing measurable.

**Other things checked and rejected (negative results worth keeping):**
- *Temporal smoothing for Task 3*: frames are NOT densely consecutive - median gap 5-17 frames, max 196. In a beating-heart surgical video, "neighbouring" labeled frames are seconds apart, so averaging across them would blur real motion. Dropped before implementing.
- *The astronomic HD (28577) on Task 1*: not our bug. `cemrg` (Task3 HD 23777) and `willenhou` (Task1 HD 14291) show the same signature on the public leaderboard - it's a scorer sentinel on some degenerate case, not something our postprocessing caused.
- *Task 3 HD ~300 is normal*: the entire top of the leaderboard sits at 146-592 HD / 28-481 ASD for Task 3. Our 359/80 was mid-pack, not a defect. Time spent chasing it would have been wasted.

**A note on the evaluation's honesty.** The 8 cases are held out of training, but they've now informed several decisions (probability-space upsampling, largest-component postprocessing, checkpoint selection). Each reuse leaks a little, so these numbers are mildly optimistic relative to the hidden test. With 27 labeled cases there's no clean alternative. The 0.650->0.834 jump is far too large to be a selection artifact, but treat the third decimal as noise.

**Task 3 higher-resolution fine-tune: tried and REJECTED.** Added a `--init-ckpt` warm-start flag to `baseline_ref/task3/train.py` and fine-tuned the existing model from 256x448 up to 384x672 (the frames are natively 720x1280, so the same "you're throwing away resolution" logic that fixed Task 1 seemed to apply). It did not transfer:

| | val_dice | composite score |
|---|---|---|
| existing checkpoint (256x448) | **0.7420** | **0.5677** |
| hires fine-tune, best epoch 6/15 | 0.7108 | 0.5089 |

Worse on every metric, with `train_dice` climbing to 0.85 while validation stalled around 0.69 - overfitting. 120 labeled frames is not enough to re-adapt the encoder to a new input resolution in 15 epochs. Kept the existing Task 3 checkpoint. Useful reminder that a fix which produces a huge win on one task is not automatically right for another.

**Final v7 configuration and measured inference cost.** Single best checkpoint (epoch 90) rather than the 3-snapshot ensemble, because the ensemble's +0.0005 is noise and tripling inference cost only buys timeout exposure - and a timeout scores zero. Measured end-to-end smoke test: **279s** for 2 Task1 + 2 Task2 + 6 Task3 inputs (~90-140s per Task 1 case with full 8-way TTA at native spacing). Extrapolated to a ~25-case hidden test that is roughly 4500s against the 21600s budget - about 4.8x margin. Kept sliding-window overlap at nnU-Net's default 0.5 rather than spending the remaining margin on an unverified increase.

| Tag | Task1 | Task2 | Task3 | Status |
|---|---|---|---|---|
| **`v7`** | **STU-Net-B at native spacing 0.36/0.36/0.51, single ckpt (epoch 90), 8-way TTA, overlap 0.5, prob-space resampling, largest-CC** | unchanged | unchanged from v5fix (hires fine-tune rejected) | **Built, smoke-tested (all 3 tasks pass, outputs sane: Task1 1.77-1.92% fg, 1 component), pushed `valpip/mvaa2026-submission:v7` (`sha256:41254495...`). Package: `submission_package_v7/`.** |

## v8: Task 3 - the augmentation was the bottleneck (2026-08-01)

**Result: Task 3 foreground DSC 0.7744 -> 0.8018 (+0.027)** on a fair head-to-head - same held-out video (`REC_20250322`, which *neither* checkpoint trained on), scored at original 720x1280 resolution through the exact production `infer.py` pipeline (2-scale + 4-way flip TTA, probability upsampling, checkpoint threshold, connected-component filter). Background false positives unchanged at 2/11.

**Diagnosis.** Task 3's geometric augmentation was only axis flips and 90-degree rotations. Both of those map the pixel grid onto itself, so they generate no genuinely new geometry - the model never saw an intermediate pose or scale. With 120 labeled frames the result was the classic under-augmentation signature: `train_dice` 0.85 against `val_dice` 0.69, with validation peaking at epoch 11 and declining afterwards.

**Fix.** Added `_affine_jitter` to `baseline_ref/task3/dataset.py` (new `--affine-prob` flag): continuous random rotation (+/-15 deg), scale (+/-20%) and translation (+/-8%), applied through a single shared `affine_grid` so image and mask stay aligned - bilinear for the image, **nearest for the mask** so labels stay strictly binary (unit-tested: mask values remain exactly {0,1} while foreground area varies with scale). Also raised the labeled training set from 120 to 150 frames by moving from a 2-video to a 1-video validation split (`--val-video-count 1`).

The overfitting gap closed and the model kept improving far longer:

| | train_dice | val_dice | best epoch |
|---|---|---|---|
| before (flips + rot90 only) | 0.85 | 0.69 | 11/20 |
| after (+ affine jitter, +25% data) | 0.83 | 0.77 | 23/26 |

**Note on comparing runs:** the new run validates on 1 video and the old on 2, so their logged `val_dice` values are NOT comparable. That is exactly why the head-to-head above was run on a single common video with a single common pipeline, rather than trusting the two runs' own reported numbers.

**Task 1 second fold: attempted, abandoned deliberately.** Launched a genuine different-split fold (seed 123) to build the kind of cross-validation ensemble the literature actually supports (unlike the correlated snapshot ensemble, which measured +0.0005). Running it concurrently with Task 3 caused GPU contention on the 6GB card - fold 2 slowed to 55s/epoch and projected past the time budget - so it was killed in favour of Task 3. The gap arithmetic drove the call: Task 1 sits 0.02 from the leader (0.834 vs 0.8541) while Task 3 was 0.17 away (0.69 vs 0.8593). Splitting the budget would have half-finished both. Task 1 fold 2 remains the obvious next move given a longer window.

| Tag | Task1 | Task2 | Task3 | Status |
|---|---|---|---|---|
| **`v8`** | unchanged from v7 (native spacing, DSC 0.834) | unchanged | **affine jitter + 150 frames, epoch 23 (fgDSC 0.8018 vs 0.7744)** | **Built, smoke-tested (all 3 pass, 193s, Task1 1 component / 1.8-1.9% fg), pushed `valpip/mvaa2026-submission:v8` (`sha256:509fb582...`). Package: `submission_package_v8/`.** |

## Deep data + literature review (2026-08-01, ~03:00-03:30)

A systematic pass over the data, our own results, the organizers' rules, and the literature - looking for what we had *not* done. Several findings changed the plan materially.

### Data findings

**Task 1: the mitral valve is an extremely thin sheet.** Measured across all 27 labeled cases:

| property | value |
|---|---|
| boundary voxels / total voxels | **0.642** |
| median max-inscribed radius | **1.82 mm** (so ~3.6mm thick) |
| connected components per case | **1** (min = median = max) |
| foreground fraction | 1.19% - 2.76% |

Three consequences. (a) Region losses (Dice/CE) integrate over *volume*, so for a structure that is ~64% surface they systematically under-weight precisely what HD and ASD measure - and HD+ASD are two thirds of the normalized task score. (b) It retroactively explains how catastrophic the old 1.5mm resampling was: the valve would have been ~2.4 voxels thick, essentially unresolvable. (c) Every case has exactly one component, which fully validates the largest-connected-component postprocessing - that was a guess before, it is now measured.

**Task 1: our intensity window is mismatched with the backbone's pretraining.** The valve occupies HU [-98, 649] (foreground p0.5/p99.5 over 536k voxels), but the pipeline maps a fixed [-1000, 1000] window onto [0,1]. That spends only ~37% of the input dynamic range on the structure of interest - and, more seriously, STU-Net was pretrained inside nnU-Net, which normalizes CT as *clip to foreground percentiles, then z-score*. We were feeding the pretrained backbone a different input distribution than it was trained on, handicapping the very transfer learning the whole approach depends on.

**Task 1: labeled and unlabeled data are the same distribution** (labeled shape 161x140x104 / spacing 0.353; unlabeled 158x144x104 / spacing 0.357; comparable HU percentiles). There is no domain-shift barrier to using the 1,040 unlabeled volumes.

**Task 3: the unlabeled pool is 7.5x more scene-diverse than the labeled set.** The 1,379 unlabeled frames span **45 distinct videos with zero overlap** with our 6 labeled videos, at identical 1280x720 resolution. Combined with the measured per-video variation (median foreground 4.3% - 13.4%) and the fact that the hidden test is different videos again, this identifies cross-video generalization as Task 3's actual bottleneck - and the unlabeled set as the direct remedy.

**Task 3: our best run barely used it.** `task3_aug` ended at epoch 26 with a 30-epoch unsupervised ramp, so `lambda_u` only ever reached **0.44 of its 0.60 maximum** - and validation was still improving at epoch 23/26. The run stopped exactly as the unlabeled data was coming online.

**Hidden test size**: ~70 Task 1 cases (public val has 30), inferred from the penalty arithmetic - our 2 failed cases produced HD 28577 = (2x1e6 + 68x~6)/70, and `willenhou`'s single failure produced exactly half that.

### Literature findings

- [nnU-Net Revisited (MICCAI 2024)](https://arxiv.org/abs/2404.09556): properly configured CNN U-Nets beat Transformer/Mamba variants; gains come from configuration, not architecture. Validated our decision not to swap architectures.
- **Boundary / compound losses**: [Karimi & Salcudean](https://arxiv.org/abs/1904.10030) and follow-ups report **18-45% HD reduction without degrading Dice**. Directly relevant given HD+ASD are 2/3 of the score and the valve is 64% surface.
- **Threshold tuning**: [Bice et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8364283/) - 0.5 is not optimal for imbalanced/small structures, and being +/-0.05 off the optimum costs a median 11.8% Dice. We had always used argmax, which for 2 classes *is* threshold 0.5, and had never swept it.
- **SWA / model soups** ([Wortsman et al.](https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf)): averaging *weights* finds flatter minima and improves generalization at 1x inference cost. This is the correct version of our failed snapshot ensemble, which averaged *predictions* of correlated checkpoints for +0.0005.
- **TTA**: gains are real but bounded (~0.1-2.3% DSC), plateauing past ~20 augmentations - so our 8-way mirror TTA is already near the useful limit.
- **Surgical-video domain gaps** are attributed mainly to colour distribution, lighting and scope characteristics - our Task 3 photometric augmentation was only contrast +/-10% / brightness +/-5%, far too mild for 45 unseen scenes.

### Changes implemented from this review

1. **nnU-Net CT normalization** (`--ct-norm nnunet`): clip to [-98, 649], z-score by foreground mean 253.55 / std 150.00. Wired through training *and* `infer.py`, with the mode recorded in the checkpoint so inference can never silently mismatch training.
2. **nnU-Net-style 3D augmentation** (`--strong-aug`): continuous affine (+/-20 deg, 0.8-1.2x), Gaussian noise, Gaussian blur, intensity scale/shift, gamma. `RandRotate90` deliberately dropped - 90 degree rotations of a cardiac CT are anatomically impossible and spend capacity on inputs that never occur.
3. **DiceCE + boundary loss** (`--boundary-loss`): signed-distance boundary term, ramped in after the region loss stabilises. Unit-tested for correct ordering of good vs bad predictions.
4. **Strong colour augmentation for Task 3** (`--color-aug`): per-channel gain (white balance), saturation, gamma, wider brightness/contrast - aimed squarely at cross-video generalization.
5. **Task 3 frame confidence gate**: verified to cut false positives 5/62 -> 1/62 at ~zero DSC cost.
6. **Task 1 empty-prediction guard**: prevents the 1,000,000-per-case penalty that cost v6 two cases.

## Reference: public leaderboard (as of 2026-07-31)

Only Task 1 and Task 3 affect ranking (Task 2 is normalized to 100 for everyone).

- Task 1 DSC: top cluster **0.83–0.85** (leader: 0.8541)
- Task 3 DSC: top **~0.85–0.86** (leader: 0.8593)
