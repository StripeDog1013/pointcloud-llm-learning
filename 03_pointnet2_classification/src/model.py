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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        xyz = x
        points = None

        l1_xyz, l1_points = self.sa1(
            xyz,
            points,
        )

        l2_xyz, l2_points = self.sa2(
            l1_xyz,
            l1_points,
        )

        _, l3_points = self.sa3(
            l2_xyz,
            l2_points,
        )

        x = l3_points.squeeze(1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

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