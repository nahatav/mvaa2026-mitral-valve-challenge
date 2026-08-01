#!/usr/bin/env python3
"""Model and loss factory for task1 semi-supervised 3D segmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
from monai.losses import DiceCELoss
from monai.networks.nets import UNet, SegResNet

logger = logging.getLogger(__name__)

# Exact architecture config matching the MONAI Model Zoo `wholeBody_ct_segmentation`
# pretrained checkpoint (SegResNet, 104 TotalSegmentator structures + background).
# https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation
PRETRAINED_SEGRESNET_CONFIG = dict(
    spatial_dims=3,
    init_filters=32,
    in_channels=1,
    out_channels=105,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    dropout_prob=0.2,
)


def get_model(
    name: str = "unet3d",
    model_size: str = "large",
    in_channels: int = 1,
    out_channels: int = 3,
    channels: Sequence[int] | None = None,
    strides: Sequence[int] = (2, 2, 2, 2),
    num_res_units: int | None = None,
    pretrained_ckpt: str | Path | None = None,
):
    """Create segmentation model by name."""
    name = name.lower()
    model_size = model_size.lower()

    if name in {"segresnet", "segresnet_pretrained"}:
        model = SegResNet(
            spatial_dims=3,
            init_filters=PRETRAINED_SEGRESNET_CONFIG["init_filters"],
            in_channels=in_channels,
            out_channels=PRETRAINED_SEGRESNET_CONFIG["out_channels"] if pretrained_ckpt else out_channels,
            blocks_down=PRETRAINED_SEGRESNET_CONFIG["blocks_down"],
            blocks_up=PRETRAINED_SEGRESNET_CONFIG["blocks_up"],
            dropout_prob=PRETRAINED_SEGRESNET_CONFIG["dropout_prob"],
        )
        if pretrained_ckpt:
            load_pretrained_segresnet_backbone(model, pretrained_ckpt)
            model = replace_segresnet_head(model, out_channels)
        return model

    if name in {"stunet", "stunet_pretrained"}:
        from stunet_arch import STUNet

        model = STUNet(
            input_channels=in_channels,
            num_classes=105 if pretrained_ckpt else out_channels,
            deep_supervision=False,
        )
        if pretrained_ckpt:
            load_pretrained_stunet_backbone(model, pretrained_ckpt)
            model = replace_stunet_head(model, out_channels)
        return model

    if channels is None or num_res_units is None:
        if model_size == "small":
            channels = (16, 32, 64, 128, 256)
            num_res_units = 2
        elif model_size == "base":
            channels = (24, 48, 96, 192, 384)
            num_res_units = 2
        elif model_size == "large":
            channels = (24, 48, 96, 192, 384)
            num_res_units = 3
        else:
            raise ValueError(f"Unsupported model_size: {model_size}")

    if name in {"unet", "unet3d"}:
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            act="PRELU",
            norm="INSTANCE",
            dropout=0.0,
        )

    raise ValueError(f"Unsupported model name: {name}")


def load_pretrained_segresnet_backbone(model: SegResNet, ckpt_path: str | Path) -> None:
    """Load MONAI wholeBody_ct_segmentation pretrained weights into a fresh
    105-class SegResNet. All keys must match exactly except the final head,
    which is swapped out afterward via replace_segresnet_head for our
    smaller class count."""
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=True)
    logger.info("Loaded pretrained SegResNet backbone from %s", ckpt_path)


def replace_segresnet_head(model: SegResNet, out_channels: int) -> SegResNet:
    """Reinitialize the final 1x1x1 conv head for a new class count, keeping
    every other pretrained layer intact."""
    import torch.nn as nn
    from monai.networks.blocks import UpSample

    old_head = model.conv_final[2].conv
    new_head = nn.Conv3d(
        old_head.in_channels,
        out_channels,
        kernel_size=old_head.kernel_size,
        stride=old_head.stride,
        padding=old_head.padding,
        bias=old_head.bias is not None,
    )
    model.conv_final[2].conv = new_head
    return model


def load_pretrained_stunet_backbone(model, ckpt_path: str | Path) -> None:
    """Load uni-medical/STU-Net-B pretrained weights (TotalSegmentator,
    105 classes) into a fresh STUNet instance. Checkpoint is nnU-Net v1
    format ({'state_dict': ...}); all 130 keys must match exactly since
    this is our own from-source reimplementation of their architecture -
    see stunet_arch.py."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=True)
    logger.info("Loaded pretrained STU-Net backbone from %s", ckpt_path)


def replace_stunet_head(model, out_channels: int):
    """Swap every deep-supervision seg_outputs head for a new class count
    (only seg_outputs[-1] is actually used since deep_supervision=False,
    but keep all in sync in case that's toggled later)."""
    import torch.nn as nn

    new_heads = nn.ModuleList()
    for old_head in model.seg_outputs:
        new_heads.append(nn.Conv3d(old_head.in_channels, out_channels, kernel_size=1))
    model.seg_outputs = new_heads
    model.num_classes = out_channels
    return model


class DiceCEBoundaryLoss(torch.nn.Module):
    """DiceCE plus a boundary (surface-distance-weighted) term.

    Why this exists: the mitral valve is an extremely thin sheet. Measured on
    all 27 labeled cases, 64.2% of its voxels are surface voxels and the
    median max-inscribed-radius is only 1.82mm (~3.6mm thick). Region losses
    like Dice/CE integrate over volume, so for a structure that is mostly
    surface they systematically under-weight exactly what HD and ASD measure -
    and HD+ASD are two thirds of this challenge's normalized task score.

    The boundary term is the classic Kervadec et al. formulation: integrate
    the predicted foreground probability against a signed distance map of the
    ground truth. Voxels far outside the true surface are penalised in
    proportion to how far outside they are, which is precisely what drives a
    worst-case surface metric like HD. Literature reports 18-45% HD reduction
    from boundary/compound losses without degrading Dice.

    The boundary weight is ramped in (alpha) rather than applied from step 0:
    the distance term is unstable when predictions are still random, so the
    standard recipe is region-loss-first, then blend the boundary term in.
    """

    def __init__(self, lambda_dice=0.8, lambda_ce=0.2, boundary_max=0.5):
        super().__init__()
        self.dice_ce = DiceCELoss(to_onehot_y=True, softmax=True,
                                  lambda_dice=lambda_dice, lambda_ce=lambda_ce)
        self.boundary_max = float(boundary_max)
        self.alpha = 0.0  # set by the training loop as it ramps

    @staticmethod
    def _signed_distance(one_hot_fg: torch.Tensor) -> torch.Tensor:
        """Signed distance map of the foreground: negative inside, positive
        outside, in voxel units. Computed on CPU with scipy (no autograd needed
        - it is a constant target derived from the labels)."""
        from scipy import ndimage
        import numpy as np

        fg = one_hot_fg.detach().cpu().numpy().astype(bool)
        out = np.zeros(fg.shape, dtype=np.float32)
        for b in range(fg.shape[0]):
            pos = fg[b]
            if pos.any() and (~pos).any():
                d_out = ndimage.distance_transform_edt(~pos)
                d_in = ndimage.distance_transform_edt(pos)
                out[b] = d_out - d_in
            elif not pos.any():
                out[b] = ndimage.distance_transform_edt(~pos)
        return torch.from_numpy(out).to(one_hot_fg.device)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.dice_ce(logits, target.unsqueeze(1) if target.dim() == logits.dim() - 1 else target)
        if self.alpha <= 0.0:
            return base
        probs = torch.softmax(logits, dim=1)
        fg_prob = probs[:, 1]                      # foreground channel
        tgt = target if target.dim() == fg_prob.dim() else target.squeeze(1)
        sdf = self._signed_distance((tgt > 0).float())
        boundary = (fg_prob * sdf).mean()
        return base + self.alpha * self.boundary_max * boundary


def get_supervised_loss_fn(boundary: bool = False):
    """Dice + CE for supervised segmentation branch, optionally with a
    boundary/surface-distance term (see DiceCEBoundaryLoss)."""
    if boundary:
        return DiceCEBoundaryLoss(lambda_dice=0.8, lambda_ce=0.2, boundary_max=0.5)
    return DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.8, lambda_ce=0.2)


def get_unsupervised_loss_fn():
    """Pseudo-label cross entropy for unlabeled branch."""
    return torch.nn.CrossEntropyLoss(reduction="none")
