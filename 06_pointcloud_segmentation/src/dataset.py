"""
dataset.py

ShapeNet Partおよび自前データ用の
点群セグメンテーションDataset
"""

from pathlib import Path
import sys

# common/ を import できるようにする
sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.utils import (
    print_header,
    print_subheader,
)

import torch
from torch.utils.data import (
    DataLoader,
)

from folder_dataset import LocalPartSegmentationDataset
from shapenet_dataset import ShapeNetPartDataset

from config import (
    BATCH_SIZE,
    CATEGORIES,
    DATA_ROOT,
    DATASET_TYPE,
    NUM_POINTS,
    NUM_WORKERS,
)

SHAPENET_CATEGORY_NAMES = [
    "Airplane",
    "Bag",
    "Cap",
    "Car",
    "Chair",
    "Earphone",
    "Guitar",
    "Knife",
    "Lamp",
    "Laptop",
    "Motorbike",
    "Mug",
    "Pistol",
    "Rocket",
    "Skateboard",
    "Table",
]

def get_datasets(
    dataset_type: str = DATASET_TYPE,
):
    """
    学習用・検証用・テスト用Datasetを作成する。
    """
    if dataset_type == "shapenet_part":
        train_dataset = ShapeNetPartDataset(
            root_dir=DATA_ROOT,
            split="train",
            categories=CATEGORIES,
            num_points=NUM_POINTS,
            include_normals=False,
        )

        val_dataset = ShapeNetPartDataset(
            root_dir=DATA_ROOT,
            split="val",
            categories=CATEGORIES,
            num_points=NUM_POINTS,
            include_normals=False,
        )

        test_dataset = ShapeNetPartDataset(
            root_dir=DATA_ROOT,
            split="test",
            categories=CATEGORIES,
            num_points=NUM_POINTS,
            include_normals=False,
        )

    elif dataset_type == "folder":
        custom_root = (
            Path(DATA_ROOT)
            / "custom"
        )

        train_dataset = LocalPartSegmentationDataset(
            root_dir=custom_root / "train",
            num_points=NUM_POINTS,
        )

        # trainと同一のカテゴリ順を使用する。
        val_dataset = LocalPartSegmentationDataset(
            root_dir=custom_root / "val",
            num_points=NUM_POINTS,
            class_names=train_dataset.class_names,
        )

        test_dataset = LocalPartSegmentationDataset(
            root_dir=custom_root / "test",
            num_points=NUM_POINTS,
            class_names=train_dataset.class_names,
        )

    else:
        raise ValueError(
            f"Unsupported dataset_type: {dataset_type}"
        )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


def get_dataloaders(
    dataset_type: str = DATASET_TYPE,
):
    """
    学習・検証・テスト用DataLoaderを作成する。
    """
    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = get_datasets(
        dataset_type=dataset_type
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )


def main():
    print_header(
        "Point Cloud Segmentation Dataset Check"
    )

    (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = get_dataloaders()

    print_subheader("Dataset")

    print(f"Dataset type  : {DATASET_TYPE}")
    print(f"Categories    : {train_dataset.class_names}")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")

    batch = next(
        iter(train_loader)
    )

    (
        points,
        categories,
        part_labels,
        features,
    ) = batch

    print_subheader("Batch Shape")

    print(f"Points      : {points.shape}")
    # (B, NUM_POINTS, 3)

    print(f"Categories  : {categories.shape}")
    # (B,)

    print(f"Part labels : {part_labels.shape}")
    # (B, NUM_POINTS)

    print(f"Features    : {features.shape}")
    # (B, NUM_POINTS, F)


if __name__ == "__main__":
    main()