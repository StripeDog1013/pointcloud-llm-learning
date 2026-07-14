"""
model.py

PointNet++による点群Part Segmentationモデル。

入力:
    points:
        (B, N, 3)

    categories:
        (B,)

    features:
        (B, N, F)
        法線などの追加特徴。使用しない場合はNoneまたは(B, N, 0)。

出力:
    logits:
        (B, N, NUM_PART_CLASSES)
"""

import torch
import torch.nn as nn

from config import (
    DROPOUT,
    FP_MLP_CHANNELS,
    GLOBAL_SA_MLP,
    NUM_CLASSES,
    NUM_PART_CLASSES,
    NUM_POINTS,
    SA_CONFIG,
)
from pointnet2_layers import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
)


class PointNet2PartSegmentation(nn.Module):
    """
    PointNet++ Part Segmentationモデル。

    Encoder:
        Set Abstraction × 2
        Global Set Abstraction

    Decoder:
        Feature Propagation × 3

    最後に各点について部品クラスを予測する。
    """

    def __init__(
        self,
        num_categories: int = NUM_CLASSES,
        num_part_classes: int = NUM_PART_CLASSES,
        input_feature_dim: int = 0,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        if len(SA_CONFIG) != 2:
            raise ValueError(
                "This model expects exactly two entries in SA_CONFIG."
            )

        if len(FP_MLP_CHANNELS) != 3:
            raise ValueError(
                "FP_MLP_CHANNELS must contain three stages."
            )

        self.num_categories = num_categories
        self.num_part_classes = num_part_classes
        self.input_feature_dim = input_feature_dim

        # ----------------------------------------------------------------------
        # Set Abstraction 1
        # ----------------------------------------------------------------------

        sa1_config = SA_CONFIG[0]

        self.sa1 = PointNetSetAbstraction(
            num_centroids=sa1_config["npoint"],
            radius=sa1_config["radius"],
            num_samples=sa1_config["nsample"],
            # 相対XYZ 3次元 + 入力特徴F次元
            in_channels=3 + input_feature_dim,
            mlp_channels=sa1_config["mlp"],
            group_all=False,
        )

        sa1_output_dim = sa1_config["mlp"][-1]

        # ----------------------------------------------------------------------
        # Set Abstraction 2
        # ----------------------------------------------------------------------

        sa2_config = SA_CONFIG[1]

        self.sa2 = PointNetSetAbstraction(
            num_centroids=sa2_config["npoint"],
            radius=sa2_config["radius"],
            num_samples=sa2_config["nsample"],
            # 相対XYZ 3次元 + SA1特徴
            in_channels=3 + sa1_output_dim,
            mlp_channels=sa2_config["mlp"],
            group_all=False,
        )

        sa2_output_dim = sa2_config["mlp"][-1]

        # ----------------------------------------------------------------------
        # Global Set Abstraction
        # ----------------------------------------------------------------------

        self.sa3 = PointNetSetAbstraction(
            num_centroids=None,
            radius=None,
            num_samples=None,
            # XYZ 3次元 + SA2特徴
            in_channels=3 + sa2_output_dim,
            mlp_channels=GLOBAL_SA_MLP,
            group_all=True,
        )

        global_output_dim = GLOBAL_SA_MLP[-1]

        # ----------------------------------------------------------------------
        # Feature Propagation 3
        # Global feature: 1点 → SA2の点数
        # ----------------------------------------------------------------------

        self.fp3 = PointNetFeaturePropagation(
            # SA2 skip特徴 + Global特徴
            in_channels=sa2_output_dim + global_output_dim,
            mlp_channels=FP_MLP_CHANNELS[0],
        )

        fp3_output_dim = FP_MLP_CHANNELS[0][-1]

        # ----------------------------------------------------------------------
        # Feature Propagation 2
        # SA2の点数 → SA1の点数
        # ----------------------------------------------------------------------

        self.fp2 = PointNetFeaturePropagation(
            # SA1 skip特徴 + FP3特徴
            in_channels=sa1_output_dim + fp3_output_dim,
            mlp_channels=FP_MLP_CHANNELS[1],
        )

        fp2_output_dim = FP_MLP_CHANNELS[1][-1]

        # ----------------------------------------------------------------------
        # Feature Propagation 1
        # SA1の点数 → 元の点数
        # ----------------------------------------------------------------------

        self.fp1 = PointNetFeaturePropagation(
            # 入力特徴 + カテゴリone-hot + FP2特徴
            in_channels=(
                input_feature_dim
                + num_categories
                + fp2_output_dim
            ),
            mlp_channels=FP_MLP_CHANNELS[2],
        )

        fp1_output_dim = FP_MLP_CHANNELS[2][-1]

        # ----------------------------------------------------------------------
        # Point-wise Segmentation Head
        # ----------------------------------------------------------------------

        self.segmentation_head = nn.Sequential(
            nn.Conv1d(
                fp1_output_dim,
                128,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                128,
                num_part_classes,
                kernel_size=1,
            ),
        )

    def _prepare_input_features(
        self,
        points: torch.Tensor,
        features: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        追加特徴を検証する。

        Parameters
        ----------
        points:
            (B, N, 3)

        features:
            (B, N, F)、None、または(B, N, 0)
        """
        if features is None:
            if self.input_feature_dim != 0:
                raise ValueError(
                    "features is required when input_feature_dim > 0."
                )

            return None

        if features.ndim != 3:
            raise ValueError(
                "features must have shape (B, N, F), "
                f"got {tuple(features.shape)}"
            )

        if features.shape[:2] != points.shape[:2]:
            raise ValueError(
                "points and features must have the same B and N."
            )

        if features.shape[-1] == 0:
            # (B, N, 0)は追加特徴なしとして扱う
            if self.input_feature_dim != 0:
                raise ValueError(
                    "Empty features cannot be used when "
                    "input_feature_dim > 0."
                )

            return None

        if features.shape[-1] != self.input_feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_feature_dim}, "
                f"got {features.shape[-1]}."
            )

        return features

    def forward(
        self,
        points: torch.Tensor,
        categories: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        順伝播。

        Parameters
        ----------
        points:
            (B, N, 3)

        categories:
            (B,)

        features:
            (B, N, F)、None、または(B, N, 0)

        Returns
        -------
        logits:
            (B, N, NUM_PART_CLASSES)
        """
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                "points must have shape (B, N, 3), "
                f"got {tuple(points.shape)}"
            )

        batch_size, num_points, _ = points.shape

        # points : (B, N, 3)

        if categories.shape != (batch_size,):
            raise ValueError(
                "categories must have shape (B,), "
                f"got {tuple(categories.shape)}"
            )

        input_features = self._prepare_input_features(
            points=points,
            features=features,
        )
        # input_features : (B, N, F) またはNone

        # ----------------------------------------------------------------------
        # Encoder
        # ----------------------------------------------------------------------

        l0_xyz = points
        # l0_xyz : (B, N, 3)

        l0_points = input_features
        # l0_points : (B, N, F) またはNone

        l1_xyz, l1_points = self.sa1(
            xyz=l0_xyz,
            points=l0_points,
        )
        # l0_xyz    : (B, N, 3)
        # l1_xyz    : (B, 512, 3)
        # l1_points : (B, 512, C1)

        l2_xyz, l2_points = self.sa2(
            xyz=l1_xyz,
            points=l1_points,
        )
        # l1_xyz    : (B, 512, 3)
        # l2_xyz    : (B, 128, 3)
        # l2_points : (B, 128, C2)

        l3_xyz, l3_points = self.sa3(
            xyz=l2_xyz,
            points=l2_points,
        )
        # l2_xyz    : (B, 128, 3)
        # l3_xyz    : (B, 1, 3)
        # l3_points : (B, 1, C3)

        # ----------------------------------------------------------------------
        # Decoder: Feature Propagation
        # ----------------------------------------------------------------------

        l2_points = self.fp3(
            target_xyz=l2_xyz,
            source_xyz=l3_xyz,
            target_points=l2_points,
            source_points=l3_points,
        )
        # source: (B, 1, C3)
        # target: (B, 128, C2)
        # l2_points : (B, 128, FP3_OUT)

        l1_points = self.fp2(
            target_xyz=l1_xyz,
            source_xyz=l2_xyz,
            target_points=l1_points,
            source_points=l2_points,
        )
        # source: (B, 128, FP3_OUT)
        # target: (B, 512, C1)
        # l1_points : (B, 512, FP2_OUT)

        # ----------------------------------------------------------------------
        # 物体カテゴリを各点へ展開
        # ----------------------------------------------------------------------

        category_one_hot = torch.nn.functional.one_hot(
            categories,
            num_classes=self.num_categories,
        ).to(
            dtype=points.dtype,
            device=points.device,
        )
        # categories       : (B,)
        # category_one_hot : (B, NUM_CLASSES)

        category_features = category_one_hot.unsqueeze(1)
        # (B, NUM_CLASSES) -> (B, 1, NUM_CLASSES)

        category_features = category_features.repeat(
            1,
            num_points,
            1,
        )
        # (B, 1, NUM_CLASSES)
        # -> (B, N, NUM_CLASSES)

        if l0_points is not None:
            l0_skip_features = torch.cat(
                [
                    l0_points,
                    category_features,
                ],
                dim=-1,
            )
            # l0_points         : (B, N, F)
            # category_features : (B, N, NUM_CLASSES)
            # l0_skip_features  : (B, N, F + NUM_CLASSES)
        else:
            l0_skip_features = category_features
            # l0_skip_features : (B, N, NUM_CLASSES)

        l0_points = self.fp1(
            target_xyz=l0_xyz,
            source_xyz=l1_xyz,
            target_points=l0_skip_features,
            source_points=l1_points,
        )
        # source: (B, 512, FP2_OUT)
        # target: (B, N, F + NUM_CLASSES)
        # l0_points : (B, N, FP1_OUT)

        # ----------------------------------------------------------------------
        # Point-wise Classification
        # ----------------------------------------------------------------------

        logits = l0_points.transpose(
            1,
            2,
        ).contiguous()
        # (B, N, FP1_OUT)
        # -> (B, FP1_OUT, N)

        logits = self.segmentation_head(
            logits
        )
        # (B, FP1_OUT, N)
        # -> (B, NUM_PART_CLASSES, N)

        logits = logits.transpose(
            1,
            2,
        ).contiguous()
        # (B, NUM_PART_CLASSES, N)
        # -> (B, N, NUM_PART_CLASSES)

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
    """
    モデル単体の動作確認。
    """
    batch_size = 2

    model = PointNet2PartSegmentation(
        num_categories=NUM_CLASSES,
        num_part_classes=NUM_PART_CLASSES,
        input_feature_dim=0,
    )

    dummy_points = torch.randn(
        batch_size,
        NUM_POINTS,
        3,
    )
    # dummy_points : (B, N, 3)

    dummy_categories = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(batch_size,),
    )
    # dummy_categories : (B,)

    dummy_features = torch.empty(
        batch_size,
        NUM_POINTS,
        0,
    )
    # dummy_features : (B, N, 0)

    model.eval()

    with torch.no_grad():
        logits = model(
            points=dummy_points,
            categories=dummy_categories,
            features=dummy_features,
        )

    # logits : (B, N, NUM_PART_CLASSES)

    print(model)
    print()
    print(f"Points shape     : {dummy_points.shape}")
    print(f"Categories shape : {dummy_categories.shape}")
    print(f"Features shape   : {dummy_features.shape}")
    print(f"Output shape     : {logits.shape}")
    print(f"Parameters       : {count_parameters(model):,}")

    expected_shape = (
        batch_size,
        NUM_POINTS,
        NUM_PART_CLASSES,
    )

    if logits.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {logits.shape}"
        )

    print("Model check succeeded.")


if __name__ == "__main__":
    main()