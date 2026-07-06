"""
pointnet2_layers.py

PointNet++のSet Abstraction Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import (
    sample_and_group,
    sample_and_group_all,
)


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction Layer。

    FPSで代表点を選び、Ball Queryで近傍点を集め、
    Shared MLP + Max Poolingで局所特徴を作る。
    """

    def __init__(
        self,
        num_centroids: int | None,
        radius: float | None,
        num_samples: int | None,
        in_channels: int,
        mlp_channels: list[int],
        group_all: bool = False,
    ):
        super().__init__()

        self.num_centroids = num_centroids
        self.radius = radius
        self.num_samples = num_samples
        self.group_all = group_all

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        last_channels = in_channels

        for out_channels in mlp_channels:
            self.convs.append(
                nn.Conv2d(
                    last_channels,
                    out_channels,
                    kernel_size=1,
                )
            )
            self.bns.append(
                nn.BatchNorm2d(out_channels)
            )
            last_channels = out_channels

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor | None,
    ):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            shape = (B, N, 3)

        points : torch.Tensor | None
            shape = (B, N, D)

        Returns
        -------
        new_xyz : torch.Tensor
            shape = (B, S, 3)

        new_points : torch.Tensor
            shape = (B, S, D')
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(
                xyz,
                points,
            )
        else:
            new_xyz, new_points = sample_and_group(
                self.num_centroids,
                self.radius,
                self.num_samples,
                xyz,
                points,
            )

        # (B, S, K, C) -> (B, C, K, S)
        new_points = new_points.permute(0, 3, 2, 1)

        for conv, bn in zip(self.convs, self.bns):
            new_points = F.relu(
                bn(conv(new_points))
            )

        # Max Pooling over K
        # (B, C, K, S) -> (B, C, S)
        new_points = torch.max(
            new_points,
            dim=2,
        )[0]

        # (B, C, S) -> (B, S, C)
        new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points