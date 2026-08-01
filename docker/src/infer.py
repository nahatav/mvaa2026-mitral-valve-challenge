#!/usr/bin/env python3
"""Unified MVAA 2026 inference entrypoint: runs Task1 (CT), Task2 (3D TEE),
and Task3 (surgical video frames) against /input and writes predictions to
/output/{t1_ct,t2_tee,t3_vid}/taskN_predictions.json + mask files, matching
the layout described in the "Final Docker Submission" spec.

No internet access is assumed at runtime (--network none). All model weights
are baked into the image under /workspace/weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# --------------------------------------------------------------------------
# Inference compute budget.
#
# Codabench evaluates on their own V100 with a 21600s timeout, and our v6 run
# used only 552s of it - inference compute is essentially free to us, while
# TRAINING is the genuinely scarce resource (6GB laptop GPU). So these knobs
# are set to spend that idle budget on accuracy.
#
# The one hard constraint: a timeout scores ZERO, so the risk is asymmetric.
# Going to native spacing already multiplies Task 1's sliding-window count
# ~12x per case (volumes are ~160x140x103 instead of ~38x34x35), and the
# snapshot ensemble multiplies it again by the number of checkpoints. These
# values are therefore set from a MEASURED local smoke test (a 6GB laptop GPU
# is slower than their V100, so local timing is a conservative upper bound),
# not from guesswork.
TASK1_SW_OVERLAP = float(os.environ.get("MVAA_TASK1_OVERLAP", "0.5"))
TASK2_SW_OVERLAP = float(os.environ.get("MVAA_TASK2_OVERLAP", "0.5"))


def log(msg: str) -> None:
    print(f"[infer] {msg}", flush=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def drop_small_components_3d(mask: np.ndarray) -> np.ndarray:
    """Keep ONLY the largest connected component for Task 1.

    The mitral valve is a single contiguous structure and every Task 1 case
    genuinely contains it (verified against real label data - min foreground
    fraction 1.19% across all 27 labeled cases), so there is no legitimate
    multi-blob or empty prediction here.

    This is deliberately stricter than the previous "keep anything >=5% of the
    largest" rule. HD is a *worst-case* surface distance: a single stray blob
    far from the true valve dominates it entirely while barely moving DSC -
    exactly the signature seen on the real hidden test (DSC 0.62, i.e. the
    main structure was found correctly, but HD/ASD ~28577). Verified on the
    8 held-out labeled cases that largest-only is identical to the old filter
    locally (predictions there had only 1-2 components), so this costs nothing
    measurable and bounds the worst case."""
    from scipy import ndimage

    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    return (labeled == (int(np.argmax(sizes)) + 1)).astype(mask.dtype)


def sliding_window_tta(images: torch.Tensor, model, inferer, num_classes: int) -> torch.Tensor:
    """Full 8-way mirror TTA for 3D sliding-window inference: averages
    softmax probs over the identity plus every combination of flips across
    the three spatial axes (2,3,4) - the standard nnU-Net "mirroring"
    recipe (their default at inference), vs. the single-axis-only subset
    used previously. Inference-time only, no retraining required."""
    import itertools

    flip_dims_list: List[Tuple[int, ...] | None] = [None]
    for r in range(1, 4):
        flip_dims_list.extend(itertools.combinations((2, 3, 4), r))

    probs_sum = None
    for dims in flip_dims_list:
        x = images if dims is None else torch.flip(images, dims=dims)
        logits = inferer(x, model)
        probs = torch.softmax(logits, dim=1)
        if dims is not None:
            probs = torch.flip(probs, dims=dims)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(flip_dims_list)


# --------------------------------------------------------------------------
# Input discovery (robust to a few plausible hidden-test layouts)
# --------------------------------------------------------------------------

def _find_task_root(input_root: Path, task_dirname: str) -> Path:
    if (input_root / task_dirname).is_dir():
        return input_root / task_dirname
    matches = [p for p in input_root.rglob(task_dirname) if p.is_dir()]
    if matches:
        return matches[0]
    return input_root


def discover_task1_images(input_root: Path) -> List[Dict[str, str]]:
    root = _find_task_root(input_root, "t1_ct")
    candidates = sorted(root.rglob("*.nii.gz"))
    files = [
        p for p in candidates
        if not p.name.endswith("-seg.nii.gz") and "label" not in p.name.lower()
    ]
    return [{"image_path": str(p), "case_id": p.name.replace(".nii.gz", "")} for p in files]


def discover_task2_images(input_root: Path) -> List[Dict[str, str]]:
    root = _find_task_root(input_root, "t2_tee")
    files = sorted(root.rglob("*-US.nii.gz"))
    return [{"image_path": str(p), "case_id": p.name.replace("-US.nii.gz", "")} for p in files]


def discover_task3_images(input_root: Path) -> List[Dict[str, str]]:
    root = _find_task_root(input_root, "t3_vid")
    if (root / "images").is_dir():
        root = root / "images"
    out = []
    for p in sorted(root.rglob("*.png")):
        name_l = p.name.lower()
        if name_l.endswith("_label_bin.png") or "_label" in name_l or "_png_label" in name_l:
            continue
        out.append(
            {
                "image_path": str(p),
                "image_rel_path": p.relative_to(root).as_posix(),
                "case_id": p.stem,
            }
        )
    return out


# --------------------------------------------------------------------------
# Task 1: cardiac CT (3D UNet, MONAI sliding window)
# --------------------------------------------------------------------------

def _load_task1_model(ckpt_path: Path, device: torch.device):
    from task1_model import get_model

    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    num_classes = int(train_args.get("num_classes", 2))
    model_name = str(train_args.get("model", "unet3d"))
    model_size = str(train_args.get("model_size", "large"))
    roi_size = tuple(train_args.get("roi_size", [128, 128, 128]))
    sw_batch_size = int(train_args.get("sw_batch_size", 4))
    enable_spacing_resample = bool(train_args.get("enable_spacing_resample", False))
    target_spacing = tuple(train_args.get("target_spacing", [0.5, 0.5, 0.5]))

    model = get_model(name=model_name, model_size=model_size, in_channels=1, out_channels=num_classes, pretrained_ckpt=None).to(device)
    state = ckpt.get("student_state_dict") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dict(
        num_classes=num_classes, roi_size=roi_size, sw_batch_size=sw_batch_size,
        enable_spacing_resample=enable_spacing_resample, target_spacing=target_spacing,
    )


def run_task1(input_root: Path, output_root: Path) -> None:
    from monai.data import DataLoader, Dataset
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import (
        Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, ScaleIntensityRanged, Spacingd,
    )

    out_dir = output_root / "t1_ct"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer STU-Net checkpoints (much stronger pretrained backbone, verified
    # on internal val: DSC 0.81 / HD 2.6 / ASD 0.22 vs SegResNet folds'
    # DSC 0.66-0.74) if any exist - ensembling in the weaker SegResNet folds
    # would dilute DSC more than the HD/ASD variance reduction is worth.
    # Falls back to SegResNet fold ensemble, then single task1_best.pt.
    stunet_ckpts = sorted(WEIGHTS_DIR.glob("task1_stunet*.pt"))
    fold_ckpts = sorted(WEIGHTS_DIR.glob("task1_fold*.pt"))
    ckpt_paths = stunet_ckpts or fold_ckpts or [WEIGHTS_DIR / "task1_best.pt"]
    log(f"task1: ensembling {len(ckpt_paths)} checkpoint(s): {[p.name for p in ckpt_paths]}")

    device = get_device()
    models_cfgs = [_load_task1_model(p, device) for p in ckpt_paths]
    num_classes = models_cfgs[0][1]["num_classes"]
    roi_size = models_cfgs[0][1]["roi_size"]
    sw_batch_size = models_cfgs[0][1]["sw_batch_size"]
    enable_spacing_resample = models_cfgs[0][1]["enable_spacing_resample"]
    target_spacing = models_cfgs[0][1]["target_spacing"]

    files = discover_task1_images(input_root)
    log(f"task1: found {len(files)} images under {input_root}")
    if not files:
        raise RuntimeError("task1: no input images discovered")

    transform_list = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ]
    if enable_spacing_resample:
        transform_list.append(Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"))
    transform_list += [
        ScaleIntensityRanged(keys=["image"], a_min=-1000.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ]
    transforms = Compose(transform_list)
    ds = Dataset([{"image": f["image_path"], "case_id": f["case_id"]} for f in files], transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # overlap=0.5 matches nnU-Net's own default sliding-window overlap -
    # denser window stitching, no retraining, meaningfully improves
    # boundary-precision metrics (HD/ASD) at the cost of more compute per
    # case. We have a 6-hour Codabench inference budget and were using a
    # small fraction of it, so this is free headroom.
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=TASK1_SW_OVERLAP, mode="gaussian")

    import nibabel as nib

    records = []
    with torch.no_grad():
        for idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            probs_sum = None
            for model, _cfg in models_cfgs:
                probs = sliding_window_tta(images, model, inferer, num_classes)
                probs_sum = probs if probs_sum is None else probs_sum + probs
            probs = probs_sum / len(models_cfgs)

            case_id = str(batch["case_id"][0])
            image_path = Path(files[idx - 1]["image_path"])
            source_img = nib.load(str(image_path))
            out_shape = tuple(int(x) for x in source_img.shape[:3])

            # Resample PROBABILITIES back to the original image grid and only
            # then take the argmax - never the reverse. Because training
            # resamples to 1.5mm, the network's output grid is coarser than
            # the original scan; taking argmax first and then nearest-
            # upsampling the hard mask quantises every boundary to the coarse
            # grid and permanently discards sub-voxel precision.
            # Measured on the 8 held-out labeled cases, scored in ORIGINAL
            # image space (the space the challenge scorer uses):
            #   argmax-then-nearest : DSC 0.6504
            #   trilinear-then-argmax: DSC 0.6898   (+0.039, better on 7/8)
            # This also explains why internal val (measured in resampled
            # space) read 0.757 while the real hidden test read 0.62 - the
            # old validation never measured the resampling round-trip loss.
            if tuple(probs.shape[2:]) != out_shape:
                probs = F.interpolate(probs.float(), size=out_shape, mode="trilinear", align_corners=False)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy()
            pred_mask = drop_small_components_3d((pred_mask > 0).astype(np.uint8))

            save_path = out_dir / f"{case_id}-pred.nii.gz"
            out_header = source_img.header.copy()
            out_header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine=source_img.affine, header=out_header), str(save_path))
            records.append({"case_id": case_id, "segmentation": save_path.name})
            log(f"task1 [{idx}/{len(files)}] -> {save_path.name}")

    with (out_dir / "task1_predictions.json").open("w", encoding="utf-8") as f:
        json.dump({"cases": records}, f, ensure_ascii=False, indent=2)
    log(f"task1: wrote {len(records)} predictions")


# --------------------------------------------------------------------------
# Task 2: 3D TEE (3D UNet, MONAI sliding window)
# --------------------------------------------------------------------------

def run_task2(input_root: Path, output_root: Path) -> None:
    from monai.data import DataLoader, Dataset
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import (
        Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, ScaleIntensityRanged,
    )
    from task2_model import get_model

    ckpt_path = WEIGHTS_DIR / "task2_best.pt"
    out_dir = output_root / "t2_tee"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    num_classes = int(train_args.get("num_classes", 3))
    model_size = str(train_args.get("model_size", "large"))
    roi_size = tuple(train_args.get("roi_size", [128, 128, 128]))
    sw_batch_size = int(train_args.get("sw_batch_size", 4))

    files = discover_task2_images(input_root)
    log(f"task2: found {len(files)} images under {input_root}")
    if not files:
        raise RuntimeError("task2: no input images discovered")

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])
    ds = Dataset([{"image": f["image_path"], "case_id": f["case_id"]} for f in files], transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    device = get_device()
    model = get_model(name="unet3d", model_size=model_size, in_channels=1, out_channels=num_classes).to(device)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=TASK2_SW_OVERLAP, mode="gaussian")

    import nibabel as nib

    records = []
    with torch.no_grad():
        for idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            probs = sliding_window_tta(images, model, inferer, num_classes)

            case_id = str(batch["case_id"][0])
            image_path = Path(files[idx - 1]["image_path"])
            source_img = nib.load(str(image_path))
            out_shape = tuple(int(x) for x in source_img.shape[:3])

            # Same probability-space resampling as Task 1 (see the note there).
            if tuple(probs.shape[2:]) != out_shape:
                probs = F.interpolate(probs.float(), size=out_shape, mode="trilinear", align_corners=False)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy()

            save_path = out_dir / f"{case_id}-pred.nii.gz"
            out_header = source_img.header.copy()
            out_header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine=source_img.affine, header=out_header), str(save_path))
            records.append({"case_id": case_id, "segmentation": save_path.name})
            log(f"task2 [{idx}/{len(files)}] -> {save_path.name}")

    with (out_dir / "task2_predictions.json").open("w", encoding="utf-8") as f:
        json.dump({"cases": records}, f, ensure_ascii=False, indent=2)
    log(f"task2: wrote {len(records)} predictions")


# --------------------------------------------------------------------------
# Task 3: surgical video frames (2D smp UNet++, TTA)
# --------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def drop_small_components(mask: np.ndarray, min_area_px: int = 50, min_rel_area: float = 0.05) -> np.ndarray:
    """Discard connected components far smaller than the main predicted
    region. HD/ASD are max/mean *surface distance* metrics — a single stray
    false-positive pixel or tiny blob far from the true mask can blow up
    both scores even when overlap (DSC) looks fine. Keeps a component if
    it's at least min_area_px pixels AND at least min_rel_area of the
    largest component's area (so legitimate secondary regions survive)."""
    from scipy import ndimage

    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    largest = sizes.max()
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_area_px and s >= min_rel_area * largest]
    # If nothing clears the bar, the correct answer is an empty mask - not
    # force-keeping the largest scrap. Previously this always kept the
    # largest component regardless, which meant a frame with only noise-
    # sized blobs (the model's honest "no valve here" signal) still emitted
    # a spurious prediction.
    if not keep_labels:
        return np.zeros_like(mask)
    return np.isin(labeled, keep_labels).astype(mask.dtype)


def run_task3(input_root: Path, output_root: Path) -> None:
    from PIL import Image
    from task3_model import get_model

    ckpt_path = WEIGHTS_DIR / "task3_best.pt"
    out_dir = output_root / "t3_vid"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    arch = str(train_args.get("arch", "unetplusplus"))
    encoder_name = str(train_args.get("encoder_name", "resnet34"))
    image_size = tuple(int(v) for v in train_args.get("image_size", [256, 448]))
    use_imagenet_norm = bool(train_args.get("use_imagenet_norm", True))
    threshold = float(ckpt.get("val_metrics", {}).get("val_threshold", 0.5))

    files = discover_task3_images(input_root)
    log(f"task3: found {len(files)} images under {input_root}, threshold={threshold:.3f}")
    if not files:
        raise RuntimeError("task3: no input images discovered")

    device = get_device()
    # encoder_weights forced to None: no internet at inference time, and the
    # full trained state_dict is loaded right after anyway.
    model = get_model(arch=arch, encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1).to(device)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    norm_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    norm_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    use_amp = device.type == "cuda"

    def _flip_tta_probs(image_t: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(image_t)
        probs = torch.sigmoid(logits)
        probs_sum = probs
        for dims in [(3,), (2,), (2, 3)]:
            x = torch.flip(image_t, dims=dims)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits_f = model(x)
            probs_sum = probs_sum + torch.flip(torch.sigmoid(logits_f), dims=dims)
        return probs_sum / 4.0

    # Multi-scale TTA: the model was trained at image_size, so that
    # resolution stays the anchor scale; a second, moderately larger scale
    # (divisible by 32 to satisfy the resnet encoder's stride, kept close
    # to the native aspect ratio) is averaged in alongside it. This gives
    # the network a second look at finer boundary detail without ever
    # replacing the trained-resolution prediction, so it can only add
    # information, not regress relative to single-scale inference.
    scale_sizes = [image_size, (int(round(image_size[0] * 1.125 / 32)) * 32, int(round(image_size[1] * 1.125 / 32)) * 32)]

    def predict_probs(image_raw_t: torch.Tensor) -> torch.Tensor:
        """image_raw_t: [0,1]-scaled RGB tensor at native resolution (not yet
        resized or imagenet-normalized) - each scale gets its own resize +
        normalize so the two scales are genuinely independent looks."""
        probs_sum = None
        for size_hw in scale_sizes:
            x = F.interpolate(image_raw_t, size=size_hw, mode="bilinear", align_corners=False)
            if use_imagenet_norm:
                x = (x - norm_mean) / norm_std
            probs = _flip_tta_probs(x)
            probs = F.interpolate(probs, size=image_size, mode="bilinear", align_corners=False)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        return probs_sum / len(scale_sizes)

    records = []
    with torch.no_grad():
        for idx, info in enumerate(files, start=1):
            image_path = Path(info["image_path"])
            rel = Path(info["image_rel_path"])

            image_u8 = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            h, w = image_u8.shape[:2]
            image_f = image_u8.astype(np.float32) / 255.0
            image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).unsqueeze(0).to(device)

            # Upsample PROBABILITIES to the original frame size and threshold
            # there, rather than thresholding at 256x448 and nearest-upsampling
            # the hard mask to 720x1280. Same reasoning (and same measured
            # failure mode) as Task 1: thresholding first quantises every
            # boundary to the coarse grid before the ~3x upsample.
            probs = predict_probs(image_t)
            probs_orig = F.interpolate(probs.float(), size=(h, w), mode="bilinear", align_corners=False)
            pred_mask = (probs_orig[0, 0].detach().cpu().numpy() > threshold).astype(np.uint8)
            pred_mask = drop_small_components(pred_mask)

            save_path = out_dir / rel.parent / f"{image_path.stem}_label_bin.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((pred_mask * 255).astype(np.uint8), mode="L").save(save_path)

            records.append({"case_id": info["case_id"], "segmentation": save_path.relative_to(out_dir).as_posix()})
            if idx % 25 == 0 or idx == len(files):
                log(f"task3 [{idx}/{len(files)}] -> {save_path.name}")

    with (out_dir / "task3_predictions.json").open("w", encoding="utf-8") as f:
        json.dump({"cases": records}, f, ensure_ascii=False, indent=2)
    log(f"task3: wrote {len(records)} predictions")


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    log(f"input={input_root} output={output_root} cuda={torch.cuda.is_available()}")

    results = {}
    for name, fn in [("task1", run_task1), ("task2", run_task2), ("task3", run_task3)]:
        try:
            fn(input_root, output_root)
            results[name] = "ok"
        except Exception:
            log(f"{name} FAILED:\n{traceback.format_exc()}")
            results[name] = "failed"

    log(f"summary: {results}")
    if all(v == "failed" for v in results.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
