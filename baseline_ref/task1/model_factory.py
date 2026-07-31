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


def get_supervised_loss_fn():
    """Dice + CE for supervised segmentation branch."""
    return DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.8, lambda_ce=0.2)


def get_unsupervised_loss_fn():
    """Pseudo-label cross entropy for unlabeled branch."""
    return torch.nn.CrossEntropyLoss(reduction="none")
