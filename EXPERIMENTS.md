# Experiment Log

Running log of what was tried, what worked, what didn't, and real results. Internal val scores are on tiny held-out splits (3 cases for Task 1, 20 for Task 3) — noisy proxies, not directly comparable to the hidden test set. Hidden test scores are logged when known.

## Task 1 (Cardiac CT)

| Run | Model | Epochs | Internal val DSC | Real hidden-test DSC | Notes |
|---|---|---|---|---|---|
| v1 (quick baseline) | UNet3D "large", from scratch | 24 | 0.545 | **0.489** | First working submission. Undertrained (~72 total gradient steps) — proved the pipeline, not competitive. |
| v2 (pretrained) | SegResNet, warm-started from MONAI `wholeBody_ct_segmentation` (TotalSegmentator-104) | 70 | **0.782** (best epoch 65) | *pending resubmit* | Transfer learning instead of training from scratch. Loss started at 0.47 vs 0.67 for from-scratch at epoch 1 — pretrained features genuinely help. |

**What didn't work:**
- `--batch-size 10 --roi-size 128^3` with SegResNet OOM'd on 6GB — the decoder's additive skip connections make it heavier than the old UNet at the same batch/resolution.
- `--batch-size 4` still OOM'd once the semi-supervised branch activated (3x forward passes per step: supervised + teacher + student-strong, all held in the graph before one `.backward()`).
- Fix: `--roi-size 96^3` (matches the pretrained checkpoint's own training resolution — also more memory-efficient) + `--batch-size 1` + `--unsup-ratio 1`. Stable through the full 70-epoch run.
- Considered `SegResNetDS` (deep-supervision variant, literature-backed for small datasets) but its state_dict structure doesn't match the plain-SegResNet pretrained checkpoint we have — swapping would mean training from scratch again, defeating the point. Skipped given the time budget.
- Considered k-fold ensembling (literature: ~1-2% DSC gain, diminishing returns past ~8 models) — skipped to respect the 1-hour total training cap.

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
| v3 (continuation, in progress) | UNet++/resnet34, `encoder_weights=imagenet`, **256x448** (half resolution) | running | DSC 0.72 @ epoch 8, climbing | *pending* | Halving resolution cut per-epoch cost ~4x. Real fix for the "too slow" problem was resolution, not epoch count. |

**On the HD=535/ASD=131 result:** initially treated as a likely bug in our own postprocessing. Cross-checked against the public leaderboard afterward — most teams show similarly large Task 3 HD/ASD (many 300-500+ HD, 60-400+ ASD), and Codabench posted a notice that they'd found and fixed a bug in their own Task 3 HD/ASD calculation around the same time. So this was likely partly a shared scoring-side issue, not primarily model quality. Added connected-component filtering anyway (drops predicted blobs far smaller than the main region) since HD/ASD are max/mean surface-distance metrics that are extremely sensitive to a single stray false-positive pixel — cheap, safe, no downside.

## Docker submission images

| Tag | Task1 | Task2 | Task3 | Status |
|---|---|---|---|---|
| `v1` | UNet3D from scratch | UNet3D from scratch | UNet++/resnet34 imagenet, cut-short training | Submitted, hidden-test scores above |
| `v2` | SegResNet pretrained | UNet3D from scratch (unchanged) | *pending v3 checkpoint* | Built, not yet pushed/submitted |

## Reference: public leaderboard (as of 2026-07-31)

Only Task 1 and Task 3 affect ranking (Task 2 is normalized to 100 for everyone).

- Task 1 DSC: top cluster **0.83–0.85** (leader: 0.8541)
- Task 3 DSC: top **~0.85–0.86** (leader: 0.8593)
