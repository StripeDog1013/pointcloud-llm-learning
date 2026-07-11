"""
positional_encoding.py

Point Transformer用 Relative Position Encoding
"""

import torch
import torch.nn as nn


class RelativePositionEncoding(nn.Module):
    """
    相対座標を特徴ベクトルへ変換する。

    入力:
        relative_xyz: (B, N, K, 3)

    出力:
        position_feature: (B, N, K, embed_dim)
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        self.position_mlp = nn.Sequential(
            nn.Linear(3, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        relative_xyz: torch.Tensor,
    ) -> torch.Tensor:
        return self.position_mlp(relative_xyz)