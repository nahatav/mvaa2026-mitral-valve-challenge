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

## Reference: public leaderboard (as of 2026-07-31)

Only Task 1 and Task 3 affect ranking (Task 2 is normalized to 100 for everyone).

- Task 1 DSC: top cluster **0.83–0.85** (leader: 0.8541)
- Task 3 DSC: top **~0.85–0.86** (leader: 0.8593)
