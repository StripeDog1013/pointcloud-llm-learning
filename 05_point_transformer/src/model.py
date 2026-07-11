"""
model.py

Transition Downを含む階層型Point Transformer分類モデル
"""

import torch
import torch.nn as nn

from config import (
    ATTENTION_DROPOUT,
    DROPOUT,
    INPUT_DIM,
    K,
    NUM_CLASSES,
    STAGE_DEPTHS,
    STAGE_DIMS,
    STAGE_NUM_POINTS,
    TRANSITION_K,
)
from transformer_layers import PointTransformerBlock
from transition_down import TransitionDown


class PointFeatureEmbedding(nn.Module):
    """
    XYZ座標を最初の点特徴へ変換する。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(
                input_dim,
                output_dim,
                bias=False,
            ),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                output_dim,
                output_dim,
                bias=False,
            ),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        xyz: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding(xyz)


class PointTransformerStage(nn.Module):
    """
    同じ解像度・特徴次元でPoint Transformer Blockを繰り返す。
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        k: int,
        attention_dropout: float,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                PointTransformerBlock(
                    embed_dim=embed_dim,
                    k=k,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            features = block(
                xyz=xyz,
                features=features,
            )

        return features


class PointTransformerEncoder(nn.Module):
    """
    5段階の階層型Point Transformer Encoder。

    点数:
        N → N/4 → N/16 → N/64 → N/256

    特徴次元:
        64 → 128 → 256 → 512 → 1024
    """

    def __init__(self):
        super().__init__()

        if not (
            len(STAGE_NUM_POINTS)
            == len(STAGE_DIMS)
            == len(STAGE_DEPTHS)
        ):
            raise ValueError(
                "STAGE_NUM_POINTS, STAGE_DIMS and "
                "STAGE_DEPTHS must have the same length."
            )

        self.input_embedding = PointFeatureEmbedding(
            input_dim=INPUT_DIM,
            output_dim=STAGE_DIMS[0],
        )

        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for stage_index, (
            stage_dim,
            stage_depth,
        ) in enumerate(
            zip(STAGE_DIMS, STAGE_DEPTHS)
        ):
            self.stages.append(
                PointTransformerStage(
                    embed_dim=stage_dim,
                    depth=stage_depth,
                    k=K,
                    attention_dropout=ATTENTION_DROPOUT,
                )
            )

            if stage_index < len(STAGE_DIMS) - 1:
                self.transitions.append(
                    TransitionDown(
                        in_channels=stage_dim,
                        out_channels=STAGE_DIMS[
                            stage_index + 1
                        ],
                        num_points=STAGE_NUM_POINTS[
                            stage_index + 1
                        ],
                        k=TRANSITION_K,
                    )
                )

    def forward(
        self,
        xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        xyz:
            shape = (B, N, 3)

        Returns
        -------
        xyz:
            最終ステージの点座標。
            shape = (B, 4, 3)

        features:
            最終ステージの点特徴。
            shape = (B, 4, 1024)
        """
        features = self.input_embedding(xyz)

        for stage_index, stage in enumerate(
            self.stages
        ):
            features = stage(
                xyz=xyz,
                features=features,
            )

            if stage_index < len(self.transitions):
                xyz, features = self.transitions[
                    stage_index
                ](
                    xyz=xyz,
                    features=features,
                )

        return xyz, features


class PointTransformerClassifier(nn.Module):
    """
    階層型Point Transformer分類モデル。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.encoder = PointTransformerEncoder()

        final_dim = STAGE_DIMS[-1]

        self.classifier = nn.Sequential(
            nn.Linear(
                final_dim * 2,
                512,
                bias=False,
            ),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(
                512,
                256,
                bias=False,
            ),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(
                256,
                num_classes,
            ),
        )

    def forward(
        self,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        points:
            (B, N, 3)

        return:
            (B, NUM_CLASSES)
        """
        _, features = self.encoder(points)

        max_features = torch.max(
            features,
            dim=1,
        )[0]

        average_features = torch.mean(
            features,
            dim=1,
        )

        global_features = torch.cat(
            [
                max_features,
                average_features,
            ],
            dim=-1,
        )

        logits = self.classifier(
            global_features
        )

        return logits


def count_parameters(
    model: nn.Module,
) -> int:
    """
    学習対象パラメータ数を返す。
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    model = PointTransformerClassifier(
        num_classes=NUM_CLASSES,
    )

    dummy_points = torch.randn(
        2,
        STAGE_NUM_POINTS[0],
        INPUT_DIM,
    )

    model.eval()

    with torch.no_grad():
        logits = model(dummy_points)

    print(model)
    print(f"Input shape  : {dummy_points.shape}")
    print(f"Output shape : {logits.shape}")
    print(f"Parameters   : {count_parameters(model):,}")

    expected_shape = (
        dummy_points.shape[0],
        NUM_CLASSES,
    )

    if logits.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {logits.shape}"
        )

    print("Model check succeeded.")


if __name__ == "__main__":
    main()