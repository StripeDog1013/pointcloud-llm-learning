"""
dataset.py

ShapeNet Partタ用の点群セグメンテーションDataset
"""

from pathlib import Path

import torch
from torch.utils.data import (
    Dataset,
)
from torch_geometric.datasets import ShapeNet

from dataset_utils import (
    normalize_points_tensor,
    sample_points_and_labels_tensor,
)

class ShapeNetPartDataset(Dataset):
    """
    PyTorch GeometricのShapeNetをラップする。

    戻り値
    -------
    points : torch.Tensor
        (NUM_POINTS, 3)

    category : torch.Tensor
        scalar

    part_labels : torch.Tensor
        (NUM_POINTS,)

    features : torch.Tensor
        (NUM_POINTS, F)
        特徴がない場合は空のTensor (NUM_POINTS, 0)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        categories: list[str] | str | None,
        num_points: int,
        include_normals: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = num_points
        self.include_normals = include_normals

        self.dataset = ShapeNet(
            root=str(self.root_dir),
            categories=categories,
            include_normals=include_normals,
            split=split,
        )

        self.class_names = list(
            self.dataset.categories
        )

        self.class_to_idx = {
            class_name: index
            for index, class_name
            in enumerate(self.class_names)
        }

        self.num_part_classes = int(
            self.dataset.num_classes
        )

        self.part_label_mask = (
            self.dataset.y_mask.clone()
        )
        # part_label_mask : (選択カテゴリ数, 50)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        data = self.dataset[index]

        points = data.pos.float()
        # points : (N, 3)

        part_labels = data.y.long()
        # part_labels : (N,)

        features = None

        if data.x is not None:
            features = data.x.float()
            # features : (N, 3) 通常は法線

        (
            points,
            part_labels,
            features,
        ) = sample_points_and_labels_tensor(
            points=points,
            labels=part_labels,
            num_points=self.num_points,
            features=features,
        )

        # points      : (NUM_POINTS, 3)
        # part_labels : (NUM_POINTS,)
        # features    : (NUM_POINTS, F) またはNone

        points = normalize_points_tensor(
            points
        )
        # points : (NUM_POINTS, 3)

        category = torch.as_tensor(
            data.category,
            dtype=torch.long,
        ).reshape(())
        # category : scalar

        if features is None:
            features = torch.empty(
                self.num_points,
                0,
                dtype=torch.float32,
            )
            # features : (NUM_POINTS, 0)

        return (
            points,
            category,
            part_labels,
            features,
        )