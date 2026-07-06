"""
visualize_samples.py

Dataset内の点群サンプルを可視化する
"""

import argparse

import numpy as np
import open3d as o3d

from config import DATASET_TYPE
from dataset import get_datasets
from utils import (
    print_header,
    print_subheader,
    print_point_cloud_info,
    visualize,
)


def points_to_point_cloud(points) -> o3d.geometry.PointCloud:
    """
    TensorまたはNumPy配列をOpen3D PointCloudへ変換する。
    """
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()

    points = np.asarray(points)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud


def get_class_name(dataset, label: int) -> str:
    """
    Datasetからクラス名を取得する。
    """
    if hasattr(dataset, "class_names"):
        return dataset.class_names[label]

    return str(label)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize point cloud samples"
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default=DATASET_TYPE,
        choices=["modelnet10", "modelnet40", "folder"],
        help="Dataset type",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index",
    )

    return parser.parse_args()


def main():
    print_header("Visualize Dataset Sample")

    args = parse_args()

    train_dataset, test_dataset = get_datasets(
        dataset_type=args.dataset_type
    )

    dataset = train_dataset if args.split == "train" else test_dataset

    print_subheader("Dataset")
    print(f"Dataset type : {args.dataset_type}")
    print(f"Split        : {args.split}")
    print(f"Samples      : {len(dataset)}")

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(
            f"Index out of range: {args.index} / {len(dataset)}"
        )

    points, label = dataset[args.index]

    if hasattr(label, "item"):
        label = label.item()

    class_name = get_class_name(dataset, label)

    print_subheader("Sample")
    print(f"Index      : {args.index}")
    print(f"Label      : {label}")
    print(f"Class name : {class_name}")
    print(f"Shape      : {points.shape}")

    point_cloud = points_to_point_cloud(points)

    print_point_cloud_info(point_cloud)

    visualize(
        point_cloud,
        window_name=f"{args.split} #{args.index}: {class_name}",
    )


if __name__ == "__main__":
    main()