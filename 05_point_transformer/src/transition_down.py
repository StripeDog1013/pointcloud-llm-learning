"""
transition_down.py

Point TransformerのTransition Down Layer
"""

import torch
import torch.nn as nn

from model_utils import (
    farthest_point_sample,
    index_points,
    knn_query,
)


class TransitionDown(nn.Module):
    """
    点数を削減しながら特徴次元を増加させる。

    Parameters
    ----------
    in_channels:
        入力特徴次元

    out_channels:
        出力特徴次元

    num_points:
        FPSで残す代表点数

    k:
        各代表点について集約する近傍点数
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_points: int,
        k: int,
    ):
        super().__init__()

        self.num_points = num_points
        self.k = k

        self.feature_mlp = nn.Sequential(
            nn.Linear(
                in_channels + 3,
                out_channels,
                bias=False,
            ),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(
                out_channels,
                out_channels,
                bias=False,
            ),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        xyz:
            元の点座標。
            shape = (B, N, 3)

        features:
            元の点特徴。
            shape = (B, N, C)

        Returns
        -------
        new_xyz:
            FPSで選択した代表点。
            shape = (B, S, 3)

        new_features:
            集約後の代表点特徴。
            shape = (B, S, out_channels)
        """
        fps_indices = farthest_point_sample(
            xyz=xyz,
            num_samples=self.num_points,
        )

        new_xyz = index_points(
            xyz,
            fps_indices,
        )

        neighbor_indices = knn_query(
            query_xyz=new_xyz,
            support_xyz=xyz,
            k=self.k,
        )

        grouped_xyz = index_points(
            xyz,
            neighbor_indices,
        )

        grouped_features = index_points(
            features,
            neighbor_indices,
        )

        relative_xyz = (
            grouped_xyz
            - new_xyz.unsqueeze(2)
        )

        local_features = torch.cat(
            [
                relative_xyz,
                grouped_features,
            ],
            dim=-1,
        )

        local_features = self.feature_mlp(
            local_features
        )

        # Global Max pooling
        new_features = torch.max(
            local_features,
            dim=2,
        )[0]

        return new_xyz, new_features