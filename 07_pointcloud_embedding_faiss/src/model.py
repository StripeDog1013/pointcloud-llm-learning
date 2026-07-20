"""
model.py

PointNet++分類モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    NUM_CLASSES,
    SA1_NUM_POINTS,
    SA1_NUM_SAMPLES,
    SA1_RADIUS,
    SA2_NUM_POINTS,
    SA2_NUM_SAMPLES,
    SA2_RADIUS,
)
from pointnet2_layers import PointNetSetAbstraction


class PointNet2Classifier(nn.Module):
    """
    PointNet++分類モデル。

    入力:
        x: (B, N, 3)

    出力:
        logits: (B, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            num_centroids=SA1_NUM_POINTS,
            radius=SA1_RADIUS,
            num_samples=SA1_NUM_SAMPLES,
            in_channels=3,
            mlp_channels=[64, 64, 128],
            group_all=False,
        )

        self.sa2 = PointNetSetAbstraction(
            num_centroids=SA2_NUM_POINTS,
            radius=SA2_RADIUS,
            num_samples=SA2_NUM_SAMPLES,
            in_channels=128 + 3,
            mlp_channels=[128, 128, 256],
            group_all=False,
        )

        self.sa3 = PointNetSetAbstraction(
            num_centroids=None,
            radius=None,
            num_samples=None,
            in_channels=256 + 3,
            mlp_channels=[256, 512, 1024],
            group_all=True,
        )

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward_features(
    self,
    x: torch.Tensor,
    ) -> torch.Tensor:
        """
        点群からグローバル特徴量を抽出する。

        Parameters
        ----------
        x : torch.Tensor
            shape = (B, N, 3)

        Returns
        -------
        torch.Tensor
            shape = (B, 1024)
        """
        xyz = x
        points = None

        l1_xyz, l1_points = self.sa1(
            xyz,
            points,
        )
        # xyz: (B, N, 3)
        # l1_xyz: (B, SA1_NUM_POINTS, 3)
        # l1_points: (B, SA1_NUM_POINTS, 128)

        l2_xyz, l2_points = self.sa2(
            l1_xyz,
            l1_points,
        )
        # l2_xyz: (B, SA2_NUM_POINTS, 3)
        # l2_points: (B, SA2_NUM_POINTS, 256)

        _, l3_points = self.sa3(
            l2_xyz,
            l2_points,
        )
        # l3_points: (B, 1, 1024)

        features = l3_points.squeeze(1)
        # (B, 1, 1024) -> (B, 1024)

        return features

    def forward(
    self,
    x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            shape = (B, N, 3)

        Returns
        -------
        torch.Tensor
            shape = (B, num_classes)
        """
        x = self.forward_features(x)
        # (B, N, 3) -> (B, 1024)

        x = F.relu(self.bn1(self.fc1(x)))
        # (B, 1024) -> (B, 512)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(x)
        # (B, 512) -> (B, 256)

        x = self.fc3(x)
        # (B, 256) -> (B, num_classes)

        return x


def count_parameters(model: nn.Module) -> int:
    """
    学習対象パラメータ数を返す。
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():
    model = PointNet2Classifier(
        num_classes=NUM_CLASSES,
    )

    dummy_points = torch.randn(4, 1024, 3)

    logits = model(dummy_points)

    print(model)
    print(f"Input shape  : {dummy_points.shape}")
    print(f"Output shape : {logits.shape}")
    print(f"Parameters   : {count_parameters(model):,}")


if __name__ == "__main__":
    main()