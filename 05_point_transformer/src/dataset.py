"""
dataset.py

Dataset生成用エントリーポイント
"""
from pathlib import Path
import sys

from torch.utils.data import DataLoader

from config import (
    DATASET_TYPE,
    BATCH_SIZE,
    NUM_POINTS,
    NUM_WORKERS,
    TEST_DIR,
    TRAIN_DIR,
)
from folder_dataset import PointCloudFolderDataset
from modelnet_dataset import ModelNetDataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.utils import print_header, print_subheader


def get_datasets(
    dataset_type: str = "modelnet10",
):
    """
    学習用・テスト用Datasetを作成する。

    Parameters
    ----------
    dataset_type : str
        "modelnet10", "modelnet40", "folder" のいずれか
    """
    if dataset_type == "modelnet10":
        train_dataset = ModelNetDataset(
            root_dir="../../02_pointnet_classification/data/ModelNet10/",
            train=True,
            num_points=NUM_POINTS,
            name="10",
        )

        test_dataset = ModelNetDataset(
            root_dir="../../02_pointnet_classification/data/ModelNet10/",
            train=False,
            num_points=NUM_POINTS,
            name="10",
        )

    elif dataset_type == "modelnet40":
        train_dataset = ModelNetDataset(
            root_dir="../../02_pointnet_classification/data/ModelNet40/",
            train=True,
            num_points=NUM_POINTS,
            name="40",
        )

        test_dataset = ModelNetDataset(
            root_dir="../../02_pointnet_classification/data/ModelNet40/",
            train=False,
            num_points=NUM_POINTS,
            name="40",
        )

    elif dataset_type == "folder":
        train_dataset = PointCloudFolderDataset(
            root_dir=TRAIN_DIR,
            num_points=NUM_POINTS,
        )

        test_dataset = PointCloudFolderDataset(
            root_dir=TEST_DIR,
            num_points=NUM_POINTS,
        )

    else:
        raise ValueError(
            f"Unsupported dataset_type: {dataset_type}"
        )

    return train_dataset, test_dataset


def get_dataloaders(
    dataset_type: str = "modelnet10",
):
    """
    学習用・テスト用DataLoaderを作成する。
    """
    train_dataset, test_dataset = get_datasets(
        dataset_type=dataset_type
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, test_loader, train_dataset, test_dataset


def main():
    print_header("Dataset Check")

    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
        dataset_type=DATASET_TYPE
    )

    print_subheader("Dataset")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")

    print_subheader("DataLoader")
    print(f"Train batches : {len(train_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    points, labels = next(iter(train_loader))

    print_subheader("Batch")
    print(f"Points shape : {points.shape}")
    print(f"Labels shape : {labels.shape}")
    print(f"Points dtype  : {points.dtype}")
    print(f"Labels dtype  : {labels.dtype}")


if __name__ == "__main__":
    main()