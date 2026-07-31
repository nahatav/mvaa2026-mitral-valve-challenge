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
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def log(msg: str) -> None:
    print(f"[infer] {msg}", flush=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sliding_window_tta(images: torch.Tensor, model, inferer, num_classes: int) -> torch.Tensor:
    """Mirror-flip TTA for 3D sliding-window inference: average softmax probs
    over identity + single-axis flips (spatial dims 2,3,4), matching the
    standard nnU-Net-style boundary-precision trick. Free at inference time,
    no extra training required."""
    flip_dims_list = [None, (2,), (3,), (4,)]
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

def run_task1(input_root: Path, output_root: Path) -> None:
    from monai.data import DataLoader, Dataset
    from monai.inferers import SlidingWindowInferer
    from monai.transforms import (
        Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, ScaleIntensityRanged, Spacingd,
    )
    from task1_model import get_model

    ckpt_path = WEIGHTS_DIR / "task1_best.pt"
    out_dir = output_root / "t1_ct"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    num_classes = int(train_args.get("num_classes", 2))
    model_name = str(train_args.get("model", "unet3d"))
    model_size = str(train_args.get("model_size", "large"))
    roi_size = tuple(train_args.get("roi_size", [128, 128, 128]))
    sw_batch_size = int(train_args.get("sw_batch_size", 4))
    enable_spacing_resample = bool(train_args.get("enable_spacing_resample", False))
    target_spacing = tuple(train_args.get("target_spacing", [0.5, 0.5, 0.5]))

    files = discover_task1_images(input_root)
    log(f"task1: found {len(files)} images under {input_root}, model={model_name}")
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

    device = get_device()
    # pretrained_ckpt=None: we're loading the full fine-tuned state_dict directly
    # below, so no need to re-fetch the original pretrained backbone here.
    model = get_model(name=model_name, model_size=model_size, in_channels=1, out_channels=num_classes, pretrained_ckpt=None).to(device)
    state = ckpt.get("student_state_dict") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=0.25, mode="gaussian")

    import nibabel as nib

    records = []
    with torch.no_grad():
        for idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            probs = sliding_window_tta(images, model, inferer, num_classes)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy()

            case_id = str(batch["case_id"][0])
            image_path = Path(files[idx - 1]["image_path"])
            source_img = nib.load(str(image_path))
            out_shape = tuple(int(x) for x in source_img.shape[:3])
            if pred_mask.shape != out_shape:
                t = torch.from_numpy(pred_mask.astype(np.float32))[None, None, ...]
                pred_mask = F.interpolate(t, size=out_shape, mode="nearest")[0, 0].numpy().astype(np.uint8)

            save_path = out_dir / f"{case_id}-pred.nii.gz"
            nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine=source_img.affine, header=source_img.header.copy()), str(save_path))
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

    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=0.25, mode="gaussian")

    import nibabel as nib

    records = []
    with torch.no_grad():
        for idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            probs = sliding_window_tta(images, model, inferer, num_classes)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0).detach().cpu().numpy()

            case_id = str(batch["case_id"][0])
            image_path = Path(files[idx - 1]["image_path"])
            source_img = nib.load(str(image_path))
            out_shape = tuple(int(x) for x in source_img.shape[:3])
            if pred_mask.shape != out_shape:
                t = torch.from_numpy(pred_mask.astype(np.float32))[None, None, ...]
                pred_mask = F.interpolate(t, size=out_shape, mode="nearest")[0, 0].numpy().astype(np.uint8)

            save_path = out_dir / f"{case_id}-pred.nii.gz"
            nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine=source_img.affine, header=source_img.header.copy()), str(save_path))
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
    if not keep_labels:
        keep_labels = [int(np.argmax(sizes)) + 1]
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

    def predict_probs(image_t: torch.Tensor) -> torch.Tensor:
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

    records = []
    with torch.no_grad():
        for idx, info in enumerate(files, start=1):
            image_path = Path(info["image_path"])
            rel = Path(info["image_rel_path"])

            image_u8 = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            h, w = image_u8.shape[:2]
            image_f = image_u8.astype(np.float32) / 255.0
            image_t = torch.from_numpy(image_f.transpose(2, 0, 1)).unsqueeze(0).to(device)
            image_t = F.interpolate(image_t, size=image_size, mode="bilinear", align_corners=False)
            if use_imagenet_norm:
                image_t = (image_t - norm_mean) / norm_std

            probs = predict_probs(image_t)
            pred_small = (probs > threshold).float()
            pred_orig = F.interpolate(pred_small, size=(h, w), mode="nearest")
            pred_mask = (pred_orig[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
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
