"""
embedding_model.py

既存のPointNet++分類モデルから特徴量を抽出し、
点群Embeddingとして出力する。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudEmbeddingModel(nn.Module):
    """
    PointNet++を利用した点群Embeddingモデル。

    Parameters
    ----------
    backbone:
        既存のPointNet++分類モデル。
        forward_features(points)を実装している必要がある。

    normalize:
        Trueの場合、EmbeddingをL2正規化する。
    """

    def __init__(
        self,
        backbone: nn.Module,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if not hasattr(
            backbone,
            "forward_features",
        ):
            raise AttributeError(
                "backbone must implement "
                "forward_features(points)."
            )

        self.backbone = backbone
        self.normalize = normalize

    def forward(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        点群からEmbeddingを生成する。

        Parameters
        ----------
        points:
            点群テンソル。

            shape:
                (batch_size, num_points, 3)

        Returns
        -------
        embeddings:
            点群Embedding。

            shape:
                (batch_size, feature_dim)
        """
        features = self.backbone.forward_features(
            points
        )
        # (B, N, 3) -> (B, feature_dim)

        if features.ndim != 2:
            raise ValueError(
                "forward_features() must return "
                "a 2D tensor with shape "
                "(batch_size, feature_dim). "
                f"got: {tuple(features.shape)}"
            )

        if self.normalize:
            features = F.normalize(
                features,
                p=2,
                dim=1,
            )
            # (B, feature_dim) -> (B, feature_dim)

        return features


def create_embedding_model(
    backbone: nn.Module,
    normalize: bool = True,
) -> PointCloudEmbeddingModel:
    """
    Embeddingモデルを生成する。
    """
    return PointCloudEmbeddingModel(
        backbone=backbone,
        normalize=normalize,
    )


def freeze_backbone(
    model: PointCloudEmbeddingModel,
) -> None:
    """
    バックボーンのパラメータを凍結する。

    今回は学習済みPointNet++からEmbeddingを抽出するため、
    通常は凍結した状態で使用する。
    """
    for parameter in (
        model.backbone.parameters()
    ):
        parameter.requires_grad = False


def unfreeze_backbone(
    model: PointCloudEmbeddingModel,
) -> None:
    """
    バックボーンのパラメータを学習可能にする。
    """
    for parameter in (
        model.backbone.parameters()
    ):
        parameter.requires_grad = True