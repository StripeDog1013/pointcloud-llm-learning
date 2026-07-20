"""
dataset.py

PointNet++で使用したPyTorch Geometric版ModelNetデータセットを
Embedding生成用に再利用する。
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.point_utils import normalize_points_tensor

import torch
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints



from config import (
    DATASET_TYPE,
    NUM_POINTS,
    TEST_DIR,
    TRAIN_DIR,
)

MODELNET10_CLASS_NAMES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]

MODELNET40_CLASS_NAMES = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]

class ModelNetDataset(Dataset):
    """
    PyTorch GeometricのModelNet10/40をラップするデータセット。

    各サンプルは次の形式で返す。

    points:
        shape = (NUM_POINTS, 3)

    label:
        shape = ()
    """

    def __init__(
        self,
        data_dir: str | Path,
        dataset_type: str,
        split: str,
        num_points: int,
    ) -> None:
        super().__init__()

        dataset_type = dataset_type.lower()
        split = split.lower()

        if dataset_type not in {
            "modelnet10",
            "modelnet40",
        }:
            raise ValueError(
                "dataset_type must be "
                "'modelnet10' or 'modelnet40'. "
                f"got: {dataset_type}"
            )

        if split not in {
            "train",
            "test",
        }:
            raise ValueError(
                "split must be 'train' or 'test'. "
                f"got: {split}"
            )

        if num_points <= 0:
            raise ValueError(
                "num_points must be greater than 0."
            )

        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.num_points = num_points

        modelnet_name = (
            "10"
            if dataset_type == "modelnet10"
            else "40"
        )

        self.dataset = ModelNet(
            root=str(self.data_dir),
            name=modelnet_name,
            train=(split == "train"),
            transform=SamplePoints(
                num=num_points,
            ),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        data = self.dataset[index]

        points = data.pos.float()
        # points : (NUM_POINTS, 3)

        points = normalize_points_tensor(
            points
        )
        # (NUM_POINTS, 3) -> (NUM_POINTS, 3)

        label = data.y.squeeze().long()
        # data.y : (1,) -> label : ()

        return points, label

    @property
    def class_names(self) -> list[str]:
        if self.dataset_type == "modelnet10":
            return MODELNET10_CLASS_NAMES

        if self.dataset_type == "modelnet40":
            return MODELNET40_CLASS_NAMES

        raise ValueError(
            f"未対応のデータセットです: {self.dataset_type}"
    )


def create_dataset(
    split: str,
) -> ModelNetDataset:
    """
    指定したsplitのデータセットを生成する。
    """
    split = split.lower()

    if DATASET_TYPE not in {
        "modelnet10",
        "modelnet40",
    }:
        raise ValueError(
            "07_pointcloud_embedding_faissでは現在、"
            "modelnet10 / modelnet40に対応しています。"
            f" DATASET_TYPE={DATASET_TYPE}"
        )

    data_dir = (
        TRAIN_DIR
        if split == "train"
        else TEST_DIR
    )

    return ModelNetDataset(
        data_dir=data_dir,
        dataset_type=DATASET_TYPE,
        split=split,
        num_points=NUM_POINTS,
    )


def get_datasets() -> tuple[
    ModelNetDataset,
    ModelNetDataset,
]:
    """
    訓練・テストデータセットを取得する。

    Returns
    -------
    train_dataset:
        ModelNetの訓練データ。

    test_dataset:
        ModelNetのテストデータ。
    """
    train_dataset = create_dataset(
        split="train"
    )

    test_dataset = create_dataset(
        split="test"
    )

    return (
        train_dataset,
        test_dataset,
    )


def get_all_dataset() -> ConcatDataset:
    """
    Embedding生成用に訓練・テストデータを結合する。

    Dataset全体のインデックスは次の並びになる。

    0 ～ len(train_dataset) - 1:
        訓練データ

    len(train_dataset)以降:
        テストデータ
    """
    train_dataset, test_dataset = (
        get_datasets()
    )

    return ConcatDataset(
        [
            train_dataset,
            test_dataset,
        ]
    )


def get_class_names() -> list[str]:
    """
    ModelNetのクラス名一覧を取得する。
    """
    train_dataset = create_dataset(
        split="train"
    )

    return train_dataset.class_names