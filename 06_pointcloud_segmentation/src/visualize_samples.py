"""
visualize_samples.py

点群セグメンテーションDatasetのサンプルを可視化する。

表示内容:
- 点群カテゴリ
- 点単位の正解部品ラベル
- ラベルごとの点数
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

# common/ をimportできるようにする
sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

from common.run_utils import timer
from common.utils import (
    print_header,
    print_subheader,
)

from config import DATASET_TYPE
from dataset import get_datasets


PART_COLORS = np.asarray(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.70, 0.20],
        [0.10, 0.30, 0.90],
        [0.90, 0.70, 0.10],
        [0.70, 0.20, 0.80],
        [0.10, 0.80, 0.80],
        [0.90, 0.40, 0.10],
        [0.50, 0.50, 0.50],
        [0.60, 0.20, 0.20],
        [0.20, 0.60, 0.20],
        [0.20, 0.20, 0.60],
        [0.60, 0.60, 0.20],
        [0.60, 0.20, 0.60],
        [0.20, 0.60, 0.60],
        [0.95, 0.55, 0.65],
        [0.55, 0.75, 0.95],
        [0.75, 0.95, 0.55],
        [0.95, 0.75, 0.55],
        [0.75, 0.55, 0.95],
        [0.55, 0.95, 0.75],
    ],
    dtype=np.float64,
)
# PART_COLORS : (20, 3)


def labels_to_colors(
    labels: np.ndarray,
) -> np.ndarray:
    """
    部品ラベルを表示色へ変換する。

    Parameters
    ----------
    labels:
        shape = (N,)

    Returns
    -------
    np.ndarray
        shape = (N, 3)
    """
    labels = np.asarray(
        labels,
        dtype=np.int64,
    ).reshape(-1)
    # labels : (N,)

    color_indices = (
        labels % len(PART_COLORS)
    )
    # color_indices : (N,)

    colors = PART_COLORS[
        color_indices
    ]
    # (N,) -> (N, 3)

    return colors


def create_colored_point_cloud(
    points: np.ndarray,
    labels: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    ラベルごとに色分けしたOpen3D点群を作成する。

    Parameters
    ----------
    points:
        shape = (N, 3)

    labels:
        shape = (N,)
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(
            "points must have shape (N, 3), "
            f"got {points.shape}"
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must have shape (N,), "
            f"got {labels.shape}"
        )

    if points.shape[0] != labels.shape[0]:
        raise ValueError(
            "points and labels must contain "
            "the same number of entries."
        )

    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = (
        o3d.utility.Vector3dVector(points)
    )

    colors = labels_to_colors(
        labels
    )
    # labels : (N,)
    # colors : (N, 3)

    point_cloud.colors = (
        o3d.utility.Vector3dVector(colors)
    )

    return point_cloud


def print_label_summary(
    labels: torch.Tensor,
) -> None:
    """
    ラベルごとの点数と割合を表示する。

    Parameters
    ----------
    labels:
        shape = (N,)
    """
    unique_labels, counts = torch.unique(
        labels,
        return_counts=True,
    )
    # unique_labels : (U,)
    # counts        : (U,)

    total_points = labels.numel()

    for label, count in zip(
        unique_labels.tolist(),
        counts.tolist(),
    ):
        ratio = count / total_points

        print(
            f"Part {label:02d}: "
            f"{count:5d} points "
            f"({ratio:.2%})"
        )


def visualize_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    window_name: str,
    point_size: float,
) -> None:
    """
    Open3Dで点群を表示する。
    """
    visualizer = o3d.visualization.Visualizer()

    visualizer.create_window(
        window_name=window_name,
        width=960,
        height=720,
    )

    visualizer.add_geometry(
        point_cloud
    )

    render_option = (
        visualizer.get_render_option()
    )

    render_option.point_size = point_size

    visualizer.run()
    visualizer.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize point segmentation dataset samples"
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=[
            "train",
            "val",
            "test",
        ],
        help="Dataset split",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Dataset sample index",
    )

    parser.add_argument(
        "--point_size",
        type=float,
        default=3.0,
        help="Open3D point size",
    )

    return parser.parse_args()


@timer
def main() -> None:
    print_header(
        "Visualize Segmentation Dataset Sample"
    )

    args = parse_args()

    print_subheader("Load Dataset")

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = get_datasets(
        dataset_type=DATASET_TYPE,
    )

    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }

    dataset = datasets[
        args.split
    ]

    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"index must be between 0 and "
            f"{len(dataset) - 1}."
        )

    (
        points,
        category,
        part_labels,
        features,
    ) = dataset[args.index]

    # points      : (N, 3)
    # category    : scalar
    # part_labels : (N,)
    # features    : (N, F) または (N, 0)

    category_index = int(
        category.item()
    )

    class_names = list(
        dataset.class_names
    )

    if not 0 <= category_index < len(
        class_names
    ):
        raise ValueError(
            "Category index is outside class_names: "
            f"{category_index}"
        )

    print(f"Dataset type  : {DATASET_TYPE}")
    print(f"Split         : {args.split}")
    print(f"Dataset size  : {len(dataset)}")
    print(f"Sample index  : {args.index}")
    print(f"Category ID   : {category_index}")
    print(
        f"Category name : "
        f"{class_names[category_index]}"
    )
    print(f"Points shape  : {points.shape}")
    print(f"Labels shape  : {part_labels.shape}")
    print(f"Features shape: {features.shape}")

    print_subheader("Part Label Summary")

    print_label_summary(
        part_labels
    )

    if hasattr(
        dataset,
        "part_label_mask",
    ):
        part_label_mask = (
            dataset.part_label_mask
        )
        # part_label_mask : (NUM_CLASSES, NUM_PART_CLASSES)

        valid_part_indices = torch.nonzero(
            part_label_mask[
                category_index
            ],
            as_tuple=False,
        ).squeeze(1)
        # valid_part_indices : (P,)

        print(
            "Valid part labels: "
            f"{valid_part_indices.tolist()}"
        )

    points_numpy = points.cpu().numpy()
    # points_numpy : (N, 3)

    labels_numpy = (
        part_labels.cpu().numpy()
    )
    # labels_numpy : (N,)

    point_cloud = (
        create_colored_point_cloud(
            points=points_numpy,
            labels=labels_numpy,
        )
    )

    print_subheader("Visualization")

    print(
        "部品ラベルごとに色分けして表示します。"
    )

    visualize_point_cloud(
        point_cloud=point_cloud,
        window_name=(
            f"{args.split} #{args.index} - "
            f"{class_names[category_index]}"
        ),
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()