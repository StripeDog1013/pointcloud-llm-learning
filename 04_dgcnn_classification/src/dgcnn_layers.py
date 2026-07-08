"""
dgcnn_layers.py

DGCNNのEdgeConv Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import get_graph_feature


class EdgeConvBlock(nn.Module):
    """
    EdgeConv Block。

    処理の流れ:
        1. kNNで近傍点を取得
        2. Edge Feature [x_j - x_i, x_i] を作成
        3. Shared MLPを適用
        4. 近傍方向にMax Pooling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
    ):
        super().__init__()

        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            shape = (B, C, N)

        Returns
        -------
        torch.Tensor
            shape = (B, out_channels, N)
        """
        x = get_graph_feature(x=x, k=self.k)

        x = self.conv(x)

        x = x.max(dim=-1)[0]

        return x