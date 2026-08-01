# MVAA 2026 — Mitral Valve Anatomy Analysis Challenge

> **Disclaimer:** This project was developed as a Proof of Concept (POC) during a limited time frame on a single RTX 4050 laptop GPU (6GB VRAM). It is not a full-scale exhaustive run.

Submission pipeline for the [MVAA 2026 Challenge](https://mwm2026.github.io/mvaa) (MICCAI 2026 Medical World Model Workshop, hosted on [Codabench](https://www.codabench.org/competitions/17301)).

Three segmentation tasks across three imaging modalities:

| Task | Modality | Labeled / Unlabeled | Metric |
|---|---|---|---|
| 1 | Cardiac CT | 27 / 1,040 | DSC, HD, ASD (semi-supervised) |
| 2 | 3D TEE (ultrasound) | 105 / — | DSC, HD, ASD (fully supervised) |
| 3 | Surgical video frames | 180 / 1,379+ | DSC, HD, ASD (semi-supervised) |

Final ranking uses only Task 1 and Task 3 — Task 2's score is fixed at 100 points for every eligible team since it uses a public dataset, so it only needs a valid submission, not a competitive one.

## Approach (Final Submission v11)

**Highest Scores (Hidden Test):**
- **Task 1:** DSC 0.811 | HD 6.86 mm | ASD 0.810 mm
- **Task 3:** DSC 0.775 | HD 361 mm | ASD 233 mm

- **Task 1**: STU-Net-B (58.26M params), supervised-pretrained on TotalSegmentator labels. Fine-tuned with mean-teacher semi-supervised learning over 1,040 unlabeled scans. Preprocessing fixes (native median spacing, nnU-Net normalization) and a compound Dice-CE-boundary loss are the key performance drivers.
- **Task 2**: 3D UNet (MONAI), fully supervised.
- **Task 3**: 2D UNet++ (`segmentation-models-pytorch`) with an ImageNet-pretrained ResNet34 encoder, mean-teacher semi-supervised learning. Inference includes multi-scale flip TTA and strong color/affine augmentations to counter cross-video brightness shifts.

All three tasks share the mean-teacher / EMA-teacher semi-supervised training pattern from the organizers' own [baseline code](https://github.com/db0725/MVAA) (`baseline_ref/`), extended here with the pretrained backbone, boundary losses, and a unified inference/Docker packaging layer.

## Repo layout

```
baseline_ref/       organizer-provided baseline (training scripts, dataset loaders)
docker/              submission image: unified inference entrypoint + Dockerfile
  src/infer.py       runs all 3 tasks against /input, writes /output per the spec
  Dockerfile         pinned to the organizers' own tested base image (torch 2.5.1+cu124, V100/sm_70 safe)
submission_package/  submission.json + submission.zip for Codabench upload
```

`data/`, `runs/`, `pretrained/`, and `.venv/` are gitignored (large — see below for how to reproduce).

## Reproducing

1. Download the dataset from the links on the [Data Access Method](https://www.codabench.org/competitions/17301) tab into `data/MVAA_Data/reference_data/`.
2. `python -m venv .venv && pip install torch monai segmentation-models-pytorch timm nibabel pillow`
3. Train each task: see `baseline_ref/task{1,2,3}/train.py --help`. Task 1 additionally takes `--model segresnet --pretrained-ckpt <path>`.
4. Copy best checkpoints into `docker/weights/` as `task{1,2,3}_best.pt`.
5. `docker build -t mvaa-submission -f docker/Dockerfile docker/`
6. Push to a registry, point `submission_package/submission.json` at the digest, zip it, upload to Codabench.

## Notes on the Docker requirement

The competition's final test phase never releases hidden test images/labels to participants — instead, each team submits a self-contained Docker image that the organizers pull and run against data only they hold, on their own V100 32GB server, offline (`--network none`). This guarantees a reproducible runtime and prevents test-set leakage.

A subtle but important gotcha: `monai`/`segmentation-models-pytorch` will silently pull in a newer `torch`/`torchvision` as a transitive dependency if installed normally, which can override the base image's known-compatible CUDA build with one that may not support the older V100 architecture (compute capability sm_70). The Dockerfile installs them with `--no-deps` and asserts the torch version post-install to guard against this.
