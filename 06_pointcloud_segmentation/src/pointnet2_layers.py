"""
pointnet2_layers.py

PointNet++ Segmentation用レイヤー。

実装内容:
- Set Abstraction
- Feature Propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import (
    interpolate_features,
    sample_and_group,
    sample_and_group_all,
)


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction Layer。

    処理:
        1. Farthest Point Sampling
        2. Ball Query
        3. Shared MLP
        4. Max Pooling

    入力:
        xyz    : (B, N, 3)
        points : (B, N, D) または None

    出力:
        new_xyz    : (B, S, 3)
        new_points : (B, S, C_out)
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
                    in_channels=last_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        xyz : torch.Tensor
            点座標。
            shape = (B, N, 3)

        points : torch.Tensor | None
            点特徴。
            shape = (B, N, D)

        Returns
        -------
        new_xyz : torch.Tensor
            代表点座標。
            shape = (B, S, 3)

        new_points : torch.Tensor
            代表点特徴。
            shape = (B, S, C_out)
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(
                xyz=xyz,
                points=points,
            )

            # new_xyz    : (B, 1, 3)
            # new_points : (B, 1, N, 3 + D)
        else:
            if (
                self.num_centroids is None
                or self.radius is None
                or self.num_samples is None
            ):
                raise ValueError(
                    "num_centroids, radius and num_samples "
                    "are required when group_all=False."
                )

            new_xyz, new_points = sample_and_group(
                num_centroids=self.num_centroids,
                radius=self.radius,
                num_samples=self.num_samples,
                xyz=xyz,
                points=points,
            )

            # new_xyz    : (B, S, 3)
            # new_points : (B, S, K, 3 + D)

        new_points = new_points.permute(
            0,
            3,
            2,
            1,
        ).contiguous()

        # (B, S, K, C_in)
        # -> (B, C_in, K, S)

        for conv, bn in zip(
            self.convs,
            self.bns,
        ):
            new_points = F.relu(
                bn(conv(new_points)),
                inplace=True,
            )

            # (B, C_in, K, S)
            # -> (B, C_out, K, S)

        new_points = torch.max(
            new_points,
            dim=2,
        )[0]

        # Max Pooling over K:
        # (B, C_out, K, S)
        # -> (B, C_out, S)

        new_points = new_points.transpose(
            1,
            2,
        ).contiguous()

        # (B, C_out, S)
        # -> (B, S, C_out)

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation Layer。

    低解像度側の特徴を高解像度側へ補間し、
    Encoder側の特徴とSkip Connectionで結合する。

    処理:
        1. 3-NN補間
        2. Skip Connection
        3. Shared MLP

    入力:
        target_xyz    : (B, N, 3)
        source_xyz    : (B, S, 3)
        target_points : (B, N, D1) または None
        source_points : (B, S, D2)

    出力:
        new_points : (B, N, C_out)
    """

    def __init__(
        self,
        in_channels: int,
        mlp_channels: list[int],
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        last_channels = in_channels

        for out_channels in mlp_channels:
            self.convs.append(
                nn.Conv1d(
                    in_channels=last_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False,
                )
            )

            self.bns.append(
                nn.BatchNorm1d(out_channels)
            )

            last_channels = out_channels

    def forward(
        self,
        target_xyz: torch.Tensor,
        source_xyz: torch.Tensor,
        target_points: torch.Tensor | None,
        source_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        target_xyz : torch.Tensor
            特徴を復元する高解像度側の座標。
            shape = (B, N, 3)

        source_xyz : torch.Tensor
            補間元となる低解像度側の座標。
            shape = (B, S, 3)

        target_points : torch.Tensor | None
            Encoder側のSkip Connection特徴。
            shape = (B, N, D1)

        source_points : torch.Tensor
            低解像度側の特徴。
            shape = (B, S, D2)

        Returns
        -------
        torch.Tensor
            高解像度側へ伝播した特徴。
            shape = (B, N, C_out)
        """
        interpolated_points = interpolate_features(
            target_xyz=target_xyz,
            source_xyz=source_xyz,
            source_points=source_points,
        )

        # source_points      : (B, S, D2)
        # interpolated_points: (B, N, D2)

        if target_points is not None:
            new_points = torch.cat(
                [
                    target_points,
                    interpolated_points,
                ],
                dim=-1,
            )

            # target_points      : (B, N, D1)
            # interpolated_points: (B, N, D2)
            # new_points         : (B, N, D1 + D2)
        else:
            new_points = interpolated_points

            # new_points : (B, N, D2)

        new_points = new_points.transpose(
            1,
            2,
        ).contiguous()

        # (B, N, C_in)
        # -> (B, C_in, N)

        for conv, bn in zip(
            self.convs,
            self.bns,
        ):
            new_points = F.relu(
                bn(conv(new_points)),
                inplace=True,
            )

            # (B, C_in, N)
            # -> (B, C_out, N)

        new_points = new_points.transpose(
            1,
            2,
        ).contiguous()

        # (B, C_out, N)
        # -> (B, N, C_out)

        return new_points