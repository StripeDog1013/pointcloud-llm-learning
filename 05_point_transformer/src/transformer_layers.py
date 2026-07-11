"""
transformer_layers.py

Point TransformerのAttention LayerとTransformer Block
"""

import torch
import torch.nn as nn

from model_utils import group_points
from positional_encoding import RelativePositionEncoding


class PointTransformerLayer(nn.Module):
    """
    Point Transformer Layer。

    各点についてk近傍を取得し、以下からAttentionを計算する。

        attention = MLP(Q_i - K_j + position_encoding)

    近傍特徴には相対位置特徴を加える。

        value = V_j + position_encoding

    Parameters
    ----------
    embed_dim : int
        入出力特徴量の次元

    k : int
        各点の近傍点数

    attention_dropout : float
        Attention Weightに対するDropout率
    """

    def __init__(
        self,
        embed_dim: int,
        k: int,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.k = k

        self.query = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        self.key = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        self.value = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )

        self.position_encoding = RelativePositionEncoding(
            embed_dim=embed_dim,
        )

        self.attention_mlp = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                embed_dim,
                embed_dim,
                bias=False,
            ),
        )

        self.attention_dropout = nn.Dropout(
            attention_dropout
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : torch.Tensor
            点の座標
            shape = (B, N, 3)

        features : torch.Tensor
            点ごとの特徴量
            shape = (B, N, C)

        Returns
        -------
        torch.Tensor
            更新後の特徴量
            shape = (B, N, C)
        """
        grouped_xyz, grouped_features = group_points(
            xyz=xyz,
            features=features,
            k=self.k,
        )

        # 中心点のQuery
        # (B, N, C)
        query = self.query(features)

        # 近傍点のKey / Value
        # (B, N, K, C)
        key = self.key(grouped_features)
        value = self.value(grouped_features)

        # 中心点から近傍点への相対座標
        # (B, N, 1, 3) - (B, N, K, 3)
        relative_xyz = (
            xyz.unsqueeze(2)
            - grouped_xyz
        )

        # 相対座標を特徴空間へ変換
        # (B, N, K, C)
        position_feature = self.position_encoding(
            relative_xyz
        )

        # (B, N, C) -> (B, N, 1, C)
        query = query.unsqueeze(2)

        # Vector Attention
        # (B, N, K, C)
        attention = self.attention_mlp(
            query - key + position_feature
        )

        # 近傍点方向にSoftmax
        attention = self.softmax(attention)
        attention = self.attention_dropout(attention)

        # 近傍特徴と相対位置特徴を集約
        # (B, N, K, C) -> (B, N, C)
        output = torch.sum(
            attention * (value + position_feature),
            dim=2,
        )

        return output


class PointTransformerBlock(nn.Module):
    """
    Point Transformer Block。

    処理順:

        Linear
        ↓
        Point Transformer Layer
        ↓
        Linear
        ↓
        Residual Connection
        ↓
        ReLU
    """

    def __init__(
        self,
        embed_dim: int,
        k: int,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim,
                bias=False,
            ),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.transformer = PointTransformerLayer(
            embed_dim=embed_dim,
            k=k,
            attention_dropout=attention_dropout,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim,
                bias=False,
            ),
            nn.BatchNorm1d(embed_dim),
        )

        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def apply_feature_block(
        features: torch.Tensor,
        block: nn.Sequential,
    ) -> torch.Tensor:
        """
        (B, N, C)形式の特徴量へ、
        BatchNorm1dを含むブロックを適用する。
        """
        features = features.transpose(1, 2)
        features = block(features.transpose(1, 2)).transpose(1, 2)

        return features

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        xyz : torch.Tensor
            shape = (B, N, 3)

        features : torch.Tensor
            shape = (B, N, C)

        Returns
        -------
        torch.Tensor
            shape = (B, N, C)
        """
        identity = features

        # Linear + BatchNorm + ReLU
        x = self.input_projection[0](features)

        x = x.transpose(1, 2)
        x = self.input_projection[1](x)
        x = x.transpose(1, 2)

        x = self.input_projection[2](x)

        # Point Transformer
        x = self.transformer(
            xyz=xyz,
            features=x,
        )

        # Linear + BatchNorm
        x = self.output_projection[0](x)

        x = x.transpose(1, 2)
        x = self.output_projection[1](x)
        x = x.transpose(1, 2)

        # Residual Connection
        x = self.activation(x + identity)

        return x