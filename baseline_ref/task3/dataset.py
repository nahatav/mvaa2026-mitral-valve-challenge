#!/usr/bin/env python3
"""Dataset utilities for task3 baseline semi-supervised segmentation."""

from __future__ import annotations

import gzip
import io
import math
import random
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMG_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d{6})\.png$")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    label_kind: str  # "tar" or "bin_png"
    video_id: str
    frame_idx: int


def _dtype_from_nifti_code(code: int):
    mapping = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        256: np.int8,
        512: np.uint16,
        768: np.uint32,
    }
    return mapping.get(code)


def _read_nifti_2d_from_bytes(nii_gz_bytes: bytes) -> np.ndarray:
    with gzip.GzipFile(fileobj=io.BytesIO(nii_gz_bytes), mode="rb") as gz:
        hdr = gz.read(348)
        dim = np.frombuffer(hdr[40:56], dtype="<i2")
        datatype_code = int(np.frombuffer(hdr[70:72], dtype="<i2")[0])
        vox_offset = int(float(np.frombuffer(hdr[108:112], dtype="<f4")[0]))
        dtype = _dtype_from_nifti_code(datatype_code)
        if dtype is None:
            raise ValueError(f"Unsupported NIfTI datatype code: {datatype_code}")

        nx = int(dim[1]) if int(dim[0]) >= 1 else 1
        ny = int(dim[2]) if int(dim[0]) >= 2 else 1
        nz = int(dim[3]) if int(dim[0]) >= 3 else 1
        count = max(nx * ny * nz, 1)

        if vox_offset > 348:
            gz.read(vox_offset - 348)

        raw = gz.read(np.dtype(dtype).itemsize * count)
        arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"), count=count)
        return arr.reshape(ny, nx)


def read_binary_mask_from_label_tar(label_tar_path: Path, target_label: int) -> np.ndarray:
    with tarfile.open(label_tar_path, mode="r") as tf:
        nii_member = None
        for member in tf.getmembers():
            if member.isfile() and member.name.endswith("_Label.nii.gz"):
                nii_member = member
                break
        if nii_member is None:
            raise FileNotFoundError(f"No _Label.nii.gz found in {label_tar_path}")
        f = tf.extractfile(nii_member)
        if f is None:
            raise FileNotFoundError(f"Failed to extract {nii_member.name} from {label_tar_path}")
        mask_raw = _read_nifti_2d_from_bytes(f.read())
        return (mask_raw == int(target_label)).astype(np.uint8)


def discover_samples(data_root: str | Path) -> List[Sample]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")

    dedup: Dict[Tuple[str, int], Sample] = {}
    for png_path in root.rglob("*.png"):
        m = IMG_RE.match(png_path.name)
        if not m:
            continue

        prefix = m.group("prefix")
        idx_str = m.group("idx")
        frame_idx = int(idx_str)
        label_bin_png = png_path.with_name(f"{prefix}_{idx_str}_label_bin.png")
        label_tar = png_path.with_name(f"{prefix}_{idx_str}_png_Label.tar")

        if label_bin_png.exists():
            label_path = label_bin_png
            label_kind = "bin_png"
        elif label_tar.exists():
            label_path = label_tar
            label_kind = "tar"
        else:
            continue

        key = (prefix, frame_idx)
        if key not in dedup:
            dedup[key] = Sample(
                image_path=png_path,
                label_path=label_path,
                label_kind=label_kind,
                video_id=prefix,
                frame_idx=frame_idx,
            )

    samples = sorted(dedup.values(), key=lambda s: (s.video_id, s.frame_idx))
    if not samples:
        raise RuntimeError(f"No valid (png, label) pairs found under {root}")
    return samples


def split_train_val_by_video(
    samples: Sequence[Sample],
    val_video_count: int = 2,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[str], List[str]]:
    by_video: Dict[str, List[Sample]] = {}
    for s in samples:
        by_video.setdefault(s.video_id, []).append(s)
    for v in list(by_video):
        by_video[v] = sorted(by_video[v], key=lambda x: x.frame_idx)

    video_ids = sorted(by_video)
    if len(video_ids) < 2:
        raise ValueError("Need at least 2 videos for train/val split.")
    if val_video_count <= 0 or val_video_count >= len(video_ids):
        raise ValueError(f"val_video_count must be in [1, {len(video_ids)-1}], got {val_video_count}")

    rng = random.Random(seed)
    shuffled = video_ids[:]
    rng.shuffle(shuffled)

    val_videos = sorted(shuffled[:val_video_count])
    train_videos = sorted(shuffled[val_video_count:])
    val_set = set(val_videos)

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    for v in video_ids:
        if v in val_set:
            val_samples.extend(by_video[v])
        else:
            train_samples.extend(by_video[v])

    if not train_samples or not val_samples:
        raise RuntimeError("Train or validation split is empty.")
    return train_samples, val_samples, train_videos, val_videos


def discover_unlabeled_images(unlabeled_root: Path) -> List[Path]:
    if not unlabeled_root.exists():
        return []
    out: List[Path] = []
    for p in sorted(unlabeled_root.rglob("*.png")):
        name = p.name
        if name.endswith("_label_bin.png"):
            continue
        if "_png_label_vis" in name:
            continue
        out.append(p)
    return out


def sample_has_foreground(sample: Sample, target_label: int = 10) -> bool:
    if sample.label_kind == "bin_png":
        arr = np.asarray(Image.open(sample.label_path).convert("L"), dtype=np.uint8)
        return bool((arr > 127).any())
    if sample.label_kind == "tar":
        mask = read_binary_mask_from_label_tar(sample.label_path, target_label=int(target_label))
        return bool(mask.any())
    raise ValueError(f"Unsupported label kind: {sample.label_kind}")


def build_fg_balanced_weights(
    fg_ratios: Sequence[float],
    power: float = 0.5,
    min_weight: float = 0.5,
    max_weight: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not fg_ratios:
        return torch.ones(1, dtype=torch.double)
    arr = np.asarray(fg_ratios, dtype=np.float64)
    out = np.full_like(arr, float(min_weight), dtype=np.float64)
    pos_mask = arr > float(eps)
    if pos_mask.any():
        inv = np.power(arr[pos_mask], -float(power))
        inv = inv / max(float(inv.mean()), 1e-8)
        inv = np.clip(inv, float(min_weight), float(max_weight))
        out[pos_mask] = inv
    # Keep pure-background samples at low weight to avoid collapsing to all-background predictions.
    return torch.as_tensor(out, dtype=torch.double)


def _resize_chw(image_t: torch.Tensor, size_hw: Tuple[int, int], mode: str) -> torch.Tensor:
    align_corners = False if mode == "bilinear" else None
    return F.interpolate(image_t.unsqueeze(0), size=size_hw, mode=mode, align_corners=align_corners).squeeze(0)


def _to_tensor_chw(img_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_np.transpose(2, 0, 1)).float()


def _apply_geom(image: np.ndarray, hflip: bool, vflip: bool, rot_k: int) -> np.ndarray:
    out = image
    if hflip:
        out = np.ascontiguousarray(out[:, ::-1, ...])
    if vflip:
        out = np.ascontiguousarray(out[::-1, :, ...])
    if rot_k > 0:
        out = np.ascontiguousarray(np.rot90(out, k=rot_k, axes=(0, 1)))
    return out


def _affine_jitter(
    image_t: torch.Tensor,
    mask_t: torch.Tensor,
    rng: random.Random,
    max_rot_deg: float = 15.0,
    max_scale: float = 0.20,
    max_shift: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random rotation / scale / translation applied identically to image and mask.

    The existing geometric augmentation only did axis flips and 90-degree
    rotations - both of which map the pixel grid onto itself, so they add no
    genuinely new geometry. With only ~120 labeled frames the model overfits
    hard (train_dice 0.85 vs val_dice 0.69), which is the classic signature of
    too-weak geometric augmentation. Continuous affine jitter gives the network
    intermediate poses and scales it never sees otherwise.

    Bilinear for the image, nearest for the mask so labels stay binary.
    """
    angle = math.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    scale = 1.0 + rng.uniform(-max_scale, max_scale)
    tx = rng.uniform(-max_shift, max_shift)
    ty = rng.uniform(-max_shift, max_shift)

    cos_a = math.cos(angle) / scale
    sin_a = math.sin(angle) / scale
    theta = torch.tensor(
        [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=torch.float32
    ).unsqueeze(0)

    img = image_t.unsqueeze(0)
    msk = mask_t.unsqueeze(0)
    grid = F.affine_grid(theta, list(img.shape), align_corners=False)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    msk = F.grid_sample(msk, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return img.squeeze(0), msk.squeeze(0)


def _color_jitter_strong(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Aggressive photometric augmentation for cross-VIDEO generalisation.

    Task 3's real bottleneck is generalising to surgical scenes we have never
    seen: we have only 6 labeled videos, while the unlabeled pool spans 45
    distinct videos and the hidden test is different videos again. Measured
    per-video median foreground varies 4.3%-13.4%, and the surgical-video
    literature attributes cross-domain failure mainly to variation in colour
    distribution, lighting and camera/scope characteristics.

    The existing `_photo_aug_weak` only perturbs contrast +/-10% and
    brightness +/-5%, which is far too mild to simulate a different scope or
    light source. This adds per-channel gain (white balance), saturation,
    gamma and wider brightness/contrast.

    img: HWC RGB float in [0,1].
    """
    out = img
    # per-channel gain ~ white balance / scope colour response
    if rng.random() < 0.8:
        gains = np.array([1.0 + rng.uniform(-0.18, 0.18) for _ in range(3)], dtype=np.float32)
        out = np.clip(out * gains[None, None, :], 0.0, 1.0)
    # saturation
    if rng.random() < 0.5:
        gray = out.mean(axis=2, keepdims=True)
        sat = 1.0 + rng.uniform(-0.35, 0.35)
        out = np.clip(gray + (out - gray) * sat, 0.0, 1.0)
    # brightness / contrast, wider than the weak version
    if rng.random() < 0.8:
        contrast = 1.0 + rng.uniform(-0.25, 0.25)
        brightness = rng.uniform(-0.12, 0.12)
        out = np.clip((out - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    # gamma
    if rng.random() < 0.5:
        gamma = float(np.exp(rng.uniform(np.log(0.7), np.log(1.45))))
        out = np.clip(np.power(np.clip(out, 1e-6, 1.0), gamma), 0.0, 1.0)
    return out.astype(np.float32)


def _photo_aug_weak(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        contrast = 1.0 + rng.uniform(-0.10, 0.10)
        brightness = rng.uniform(-0.05, 0.05)
        img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    return img


def _photo_aug_strong(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.8:
        contrast = 1.0 + rng.uniform(-0.25, 0.25)
        brightness = rng.uniform(-0.12, 0.12)
        img = np.clip((img - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

    if rng.random() < 0.6:
        gamma = np.exp(rng.uniform(np.log(0.7), np.log(1.5)))
        img = np.clip(np.power(np.clip(img, 0.0, 1.0), gamma), 0.0, 1.0)

    if rng.random() < 0.5:
        noise = np.random.normal(0.0, 0.03, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    return img


def _normalize_if_needed(image_t: torch.Tensor, use_imagenet_norm: bool) -> torch.Tensor:
    if not use_imagenet_norm:
        return image_t
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (image_t - mean) / std


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: Tuple[int, int],
        target_label: int,
        train: bool,
        cache_masks: bool,
        use_imagenet_norm: bool,
        seed: int,
        affine_prob: float = 0.0,
        color_aug: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.target_label = int(target_label)
        self.train = bool(train)
        self.affine_prob = float(affine_prob)
        self.color_aug = bool(color_aug)
        self.cache_masks = bool(cache_masks)
        self.use_imagenet_norm = bool(use_imagenet_norm)
        self.rng = random.Random(seed)
        self._mask_cache: Dict[Path, np.ndarray] = {}

        if self.cache_masks:
            for s in self.samples:
                self._mask_cache[s.label_path] = self._read_mask(s)
        self.sample_fg_ratio = [float(self._load_mask(s).mean()) for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _read_mask(self, sample: Sample) -> np.ndarray:
        if sample.label_kind == "bin_png":
            arr = np.asarray(Image.open(sample.label_path).convert("L"), dtype=np.uint8)
            return (arr > 127).astype(np.uint8)
        if sample.label_kind == "tar":
            return read_binary_mask_from_label_tar(sample.label_path, target_label=self.target_label)
        raise ValueError(f"Unsupported label kind: {sample.label_kind}")

    def _load_mask(self, sample: Sample) -> np.ndarray:
        if sample.label_path in self._mask_cache:
            return self._mask_cache[sample.label_path]
        return self._read_mask(sample)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = np.asarray(Image.open(s.image_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = self._load_mask(s).astype(np.float32)

        if self.train:
            hflip = self.rng.random() < 0.5
            vflip = self.rng.random() < 0.3
            rot_k = self.rng.randint(0, 3) if self.rng.random() < 0.4 else 0
            image = _apply_geom(image, hflip=hflip, vflip=vflip, rot_k=rot_k)
            mask = _apply_geom(mask[..., None], hflip=hflip, vflip=vflip, rot_k=rot_k)[..., 0]
            if self.color_aug:
                image = _color_jitter_strong(image, self.rng)
            else:
                image = _photo_aug_weak(image, self.rng)
            if self.rng.random() < 0.2:
                noise = np.random.normal(0.0, 0.02, size=image.shape).astype(np.float32)
                image = np.clip(image + noise, 0.0, 1.0)

        image_t = _to_tensor_chw(image)
        mask_t = torch.from_numpy(mask[None, ...]).float()

        if tuple(image_t.shape[1:]) != self.image_size:
            image_t = _resize_chw(image_t, self.image_size, mode="bilinear")
            mask_t = F.interpolate(mask_t.unsqueeze(0), size=self.image_size, mode="nearest").squeeze(0)

        # Affine jitter after the resize so it operates at the network's input
        # resolution (cheaper, and the sampling grid matches what the model sees).
        if self.train and self.affine_prob > 0.0 and self.rng.random() < self.affine_prob:
            image_t, mask_t = _affine_jitter(image_t, mask_t, self.rng)

        image_t = _normalize_if_needed(image_t, self.use_imagenet_norm)
        return {
            "image": image_t,
            "label": mask_t,
            "video_id": s.video_id,
            "frame_idx": s.frame_idx,
            "image_path": str(s.image_path),
            "label_path": str(s.label_path),
            "label_kind": s.label_kind,
        }


class UnlabeledPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        image_size: Tuple[int, int],
        use_imagenet_norm: bool,
        seed: int,
    ) -> None:
        self.image_paths = list(image_paths)
        self.image_size = image_size
        self.use_imagenet_norm = bool(use_imagenet_norm)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        image = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

        hflip = self.rng.random() < 0.5
        vflip = self.rng.random() < 0.3
        rot_k = self.rng.randint(0, 3) if self.rng.random() < 0.4 else 0

        base = _apply_geom(image, hflip=hflip, vflip=vflip, rot_k=rot_k)
        weak = _photo_aug_weak(base.copy(), self.rng)
        strong = _photo_aug_strong(base.copy(), self.rng)

        weak_t = _to_tensor_chw(weak)
        strong_t = _to_tensor_chw(strong)

        if tuple(weak_t.shape[1:]) != self.image_size:
            weak_t = _resize_chw(weak_t, self.image_size, mode="bilinear")
            strong_t = _resize_chw(strong_t, self.image_size, mode="bilinear")

        weak_t = _normalize_if_needed(weak_t, self.use_imagenet_norm)
        strong_t = _normalize_if_needed(strong_t, self.use_imagenet_norm)

        return {
            "weak": weak_t,
            "strong": strong_t,
            "image_path": str(p),
        }
