"""
model.py

DGCNN分類モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    DROPOUT,
    EDGE_CONV_CHANNELS,
    EMBEDDING_DIM,
    K,
    NUM_CLASSES,
)
from dgcnn_layers import EdgeConvBlock


class DGCNNClassifier(nn.Module):
    """
    DGCNN Classification Network

    Input
    -----
    (B, N, 3)

    Output
    ------
    (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        k: int = K,
    ):
        super().__init__()

        c1, c2, c3, c4 = EDGE_CONV_CHANNELS

        self.edge_conv1 = EdgeConvBlock(
            in_channels=3,
            out_channels=c1,
            k=k,
        )

        self.edge_conv2 = EdgeConvBlock(
            in_channels=c1,
            out_channels=c2,
            k=k,
        )

        self.edge_conv3 = EdgeConvBlock(
            in_channels=c2,
            out_channels=c3,
            k=k,
        )

        self.edge_conv4 = EdgeConvBlock(
            in_channels=c3,
            out_channels=c4,
            k=k,
        )

        concat_channels = c1 + c2 + c3 + c4

        self.embedding = nn.Sequential(
            nn.Conv1d(
                concat_channels,
                EMBEDDING_DIM,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(EMBEDDING_DIM),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                EMBEDDING_DIM * 2,
                512,
                bias=False,
            ),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
            nn.Dropout(DROPOUT),

            nn.Linear(
                512,
                256,
                bias=False,
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
            nn.Dropout(DROPOUT),

            nn.Linear(
                256,
                num_classes,
            ),
        )

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

        # (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1)

        x1 = self.edge_conv1(x)

        x2 = self.edge_conv2(x1)

        x3 = self.edge_conv3(x2)

        x4 = self.edge_conv4(x3)

        x = torch.cat((x1, x2, x3, x4),
            dim=1,
        )

        x = self.embedding(x)

        x_max = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)

        x = torch.cat((x_max, x_avg), dim=1)

        logits = self.classifier(x)

        return logits


def count_parameters(
    model: nn.Module,
) -> int:
    """
    学習対象パラメータ数を返す。
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():
    model = DGCNNClassifier()

    dummy_points = torch.randn(4, 1024, 3)

    logits = model(dummy_points)

    print(model)

    print(f"Input shape  : {dummy_points.shape}")
    print(f"Output shape : {logits.shape}")

    print(
        f"Parameters   : {count_parameters(model):,}"
    )


if __name__ == "__main__":
    main()