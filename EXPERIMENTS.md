# Experiment Log

Running log of what was tried, what worked, what didn't, and real results. Internal val scores are on tiny held-out splits (3 cases for Task 1, 20 for Task 3) — noisy proxies, not directly comparable to the hidden test set. Hidden test scores are logged when known.

## Task 1 (Cardiac CT)

| Run | Model | Epochs | Internal val DSC | Internal val HD/ASD | Real hidden-test DSC | Notes |
|---|---|---|---|---|---|---|
| v1 (quick baseline) | UNet3D "large", from scratch | 24 | 0.545 | 99.9 / 6.58 | **0.489** | First working submission. Undertrained (~72 total gradient steps) — proved the pipeline, not competitive. |
| v2 (pretrained) | SegResNet, warm-started from MONAI `wholeBody_ct_segmentation`, native ~0.5mm spacing (no resample) | 70 | 0.782 (best epoch 65) | 60.5 / 2.34 | *superseded, not submitted* | Transfer learning instead of training from scratch. Loss started at 0.47 vs 0.67 for from-scratch at epoch 1. |
| **v3 (pretrained + spacing-matched)** | SegResNet, same pretrained backbone, **resampled to 1.5mm isotropic to match pretraining spacing** | 80 | 0.722 (best epoch 70) | **3.11 / 0.36** | *pending resubmit* | DSC roughly flat vs v2, but HD/ASD dramatically better (60→3.1, 2.3→0.36) — matching the pretrained model's expected physical spacing made a real difference to boundary precision, per literature ("fine-tuning data should mirror the pretraining preprocessing"). ASD 0.36 and HD 3.1 are competitive with the current #1 (HD 5.1, ASD 0.31). |

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
| v3 (continuation, cut at epoch 10/40) | UNet++/resnet34, `encoder_weights=imagenet`, **256x448** (half resolution) | 10/40 (cut for time budget) | 0.749 (best epoch 10) | *pending resubmit* | Halving resolution cut per-epoch cost ~4x — that was the real fix for "too slow", not epoch count. Cut short to respect the training time cap; checkpoint still an improvement over v2. |

**On the HD=535/ASD=131 result:** initially treated as a likely bug in our own postprocessing. Cross-checked against the public leaderboard afterward — most teams show similarly large Task 3 HD/ASD (many 300-500+ HD, 60-400+ ASD), and Codabench posted a notice that they'd found and fixed a bug in their own Task 3 HD/ASD calculation around the same time. So this was likely partly a shared scoring-side issue, not primarily model quality. Added connected-component filtering anyway (drops predicted blobs far smaller than the main region) since HD/ASD are max/mean surface-distance metrics that are extremely sensitive to a single stray false-positive pixel — cheap, safe, no downside.

## Docker submission images

| Tag | Task1 | Task2 | Task3 | Status |
|---|---|---|---|---|
| `v1` | UNet3D from scratch | UNet3D from scratch | UNet++/resnet34 imagenet, cut-short training | Submitted, hidden-test scores above |
| `v2` | SegResNet pretrained (no spacing match) | UNet3D from scratch (unchanged) | epoch-6 checkpoint | Built, not submitted (superseded by v3) |
| `v3` | SegResNet pretrained, spacing-matched (1.5mm) | UNet3D from scratch (unchanged) | epoch-10 checkpoint (256x448) | **Built, tested end-to-end, pushed to `valpip/mvaa2026-submission:v3`. Ready to submit.** |

`docker/weights/*.pt` are gitignored (large binaries) — the trained checkpoints live in `runs/task1_pretrained_v2/`, `runs/task3_sota2/`, `runs/task2_large/` locally.

## Reference: public leaderboard (as of 2026-07-31)

Only Task 1 and Task 3 affect ranking (Task 2 is normalized to 100 for everyone).

- Task 1 DSC: top cluster **0.83–0.85** (leader: 0.8541)
- Task 3 DSC: top **~0.85–0.86** (leader: 0.8593)
