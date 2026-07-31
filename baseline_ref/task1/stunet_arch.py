#!/usr/bin/env python3
"""Standalone STU-Net architecture, adapted from uni-medical/STU-Net
(nnUNet-1.7.1/nnunet/network_architecture/STUNet.py) to be usable outside
the full nnU-Net v1 framework: SegmentationNetwork base class replaced with
plain nn.Module (we use MONAI's own SlidingWindowInferer for inference
instead of nnU-Net's built-in predict_3D), and deep supervision defaults to
off so forward() returns a single tensor matching the rest of this
pipeline's training/inference code.

Architecture and weights: https://github.com/uni-medical/STU-Net
Pretrained on TotalSegmentator (105 classes incl. background), used here
purely as a transfer-learning backbone - the final head is swapped for our
own class count after loading.
"""
from __future__ import annotations

import torch
from torch import nn


class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class UpsampleLayerNearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode="nearest"):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        return self.conv(x)


class STUNet(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        depth=(1, 1, 1, 1, 1, 1),
        dims=(32, 64, 128, 256, 512, 512),
        pool_op_kernel_sizes=None,
        conv_kernel_sizes=None,
        deep_supervision=False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.do_ds = deep_supervision

        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [[2, 2, 2]] * (len(dims) - 1)
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [[3, 3, 3]] * len(dims)

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = [[i // 2 for i in krnl] for krnl in conv_kernel_sizes]

        num_pool = len(pool_op_kernel_sizes)
        assert num_pool == len(dims) - 1

        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0] - 1)],
        )
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(
                BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d], stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d]) for _ in range(depth[d] - 1)],
            )
            self.conv_blocks_context.append(stage)

        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            self.upsample_layers.append(UpsampleLayerNearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u]))

        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(
                BasicResBlock(dims[-2 - u] * 2, dims[-2 - u], self.conv_kernel_sizes[-2 - u], self.conv_pad_sizes[-2 - u], use_1x1conv=True),
                *[BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u], self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)],
            )
            self.conv_blocks_localization.append(stage)

        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.seg_outputs[u](x))

        if self.do_ds:
            return tuple(seg_outputs[::-1])
        return seg_outputs[-1]
