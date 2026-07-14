"""
visualize_prediction.py

PointNet++ Part Segmentationの予測結果をOpen3Dで可視化する。

表示内容:
- Ground Truth
- Prediction
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

from common.device import (
    get_device,
    print_device_info,
)
from common.run_utils import timer
from common.train_utils import load_checkpoint
from common.utils import (
    print_header,
    print_subheader,
)

from config import (
    CHECKPOINT_DIR,
    CHECKPOINT_NAME,
    CUDA_ID,
    DATASET_TYPE,
    DROPOUT,
    NUM_CLASSES,
    NUM_PART_CLASSES,
)
from dataset import get_datasets
from model import PointNet2PartSegmentation


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


def predict_valid_parts(
    logits: torch.Tensor,
    category_index: int,
    part_label_mask: torch.Tensor,
) -> torch.Tensor:
    """
    指定カテゴリで有効な部品ラベルだけから予測する。

    Parameters
    ----------
    logits:
        shape = (1, N, NUM_PART_CLASSES)

    part_label_mask:
        shape = (NUM_CLASSES, NUM_PART_CLASSES)

    Returns
    -------
    torch.Tensor
        shape = (N,)
    """
    valid_part_indices = torch.nonzero(
        part_label_mask[category_index],
        as_tuple=False,
    ).squeeze(1)
    # valid_part_indices : (P,)

    category_logits = logits[
        0,
        :,
        valid_part_indices,
    ]
    # logits          : (1, N, NUM_PART_CLASSES)
    # category_logits : (N, P)

    local_predictions = category_logits.argmax(
        dim=-1
    )
    # (N, P) -> (N,)

    predictions = valid_part_indices[
        local_predictions
    ]
    # local_predictions : (N,)
    # predictions       : (N,)

    return predictions


def calculate_instance_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_part_indices: torch.Tensor,
) -> float:
    """
    1点群分のmIoUを計算する。

    prediction:
        shape = (N,)

    target:
        shape = (N,)
    """
    part_ious = []

    for part_index in valid_part_indices.tolist():
        predicted_mask = (
            prediction == part_index
        )
        # predicted_mask : (N,)

        target_mask = (
            target == part_index
        )
        # target_mask : (N,)

        intersection = torch.logical_and(
            predicted_mask,
            target_mask,
        ).sum().item()

        union = torch.logical_or(
            predicted_mask,
            target_mask,
        ).sum().item()

        part_iou = (
            1.0
            if union == 0
            else intersection / union
        )

        part_ious.append(part_iou)

    return sum(part_ious) / len(part_ious)


def print_label_summary(
    title: str,
    labels: torch.Tensor,
) -> None:
    """
    ラベルごとの点数を表示する。

    labels:
        shape = (N,)
    """
    print_subheader(title)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize PointNet++ Part Segmentation prediction"
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Test dataset sample index",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            Path(CHECKPOINT_DIR)
            / CHECKPOINT_NAME
        ),
        help="Checkpoint path",
    )

    parser.add_argument(
        "--point_size",
        type=float,
        default=3.0,
        help="Open3D point size",
    )

    return parser.parse_args()


def visualize_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    window_name: str,
    point_size: float,
) -> None:
    """
    点群を指定した点サイズで可視化する。
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


@timer
def main() -> None:
    print_header(
        "Visualize PointNet++ Part Segmentation"
    )

    args = parse_args()

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print_subheader("Load Dataset")

    (
        _,
        _,
        test_dataset,
    ) = get_datasets(
        dataset_type=DATASET_TYPE,
    )

    if not 0 <= args.index < len(test_dataset):
        raise IndexError(
            f"index must be between 0 and "
            f"{len(test_dataset) - 1}."
        )

    if not hasattr(
        test_dataset,
        "part_label_mask",
    ):
        raise AttributeError(
            "Dataset must provide part_label_mask."
        )

    (
        points,
        category,
        target_labels,
        features,
    ) = test_dataset[args.index]

    # points        : (N, 3)
    # category      : scalar
    # target_labels : (N,)
    # features      : (N, F) または (N, 0)

    category_index = int(
        category.item()
    )

    class_names = list(
        test_dataset.class_names
    )

    print(f"Dataset index : {args.index}")
    print(
        f"Category      : "
        f"{class_names[category_index]}"
    )
    print(f"Points shape  : {points.shape}")
    print(f"Labels shape  : {target_labels.shape}")

    device = get_device(
        cuda_id=CUDA_ID,
    )

    print_device_info(
        cuda_id=CUDA_ID,
    )

    points_batch = points.unsqueeze(0).to(
        device
    )
    # (N, 3) -> (1, N, 3)

    categories_batch = torch.tensor(
        [category_index],
        dtype=torch.long,
        device=device,
    )
    # categories_batch : (1,)

    features_batch = features.unsqueeze(0).to(
        device
    )
    # (N, F) -> (1, N, F)

    part_label_mask = (
        test_dataset.part_label_mask.to(
            device
        )
    )
    # part_label_mask : (NUM_CLASSES, NUM_PART_CLASSES)

    print_subheader("Load Model")

    model = PointNet2PartSegmentation(
        num_categories=NUM_CLASSES,
        num_part_classes=NUM_PART_CLASSES,
        input_feature_dim=0,
        dropout=DROPOUT,
    ).to(device)

    checkpoint = load_checkpoint(
        model=model,
        optimizer=None,
        file_path=checkpoint_path,
        map_location=device,
    )

    model.eval()

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Epoch      : {checkpoint['epoch']}")

    print_subheader("Predict")

    with torch.no_grad():
        logits = model(
            points=points_batch,
            categories=categories_batch,
            features=features_batch,
        )
        # logits : (1, N, NUM_PART_CLASSES)

        predictions = predict_valid_parts(
            logits=logits,
            category_index=category_index,
            part_label_mask=part_label_mask,
        )
        # predictions : (N,)

    target_labels_device = target_labels.to(
        device
    )
    # target_labels_device : (N,)

    valid_part_indices = torch.nonzero(
        part_label_mask[category_index],
        as_tuple=False,
    ).squeeze(1)
    # valid_part_indices : (P,)

    point_accuracy = (
        predictions
        == target_labels_device
    ).float().mean().item()

    instance_miou = calculate_instance_iou(
        prediction=predictions,
        target=target_labels_device,
        valid_part_indices=valid_part_indices,
    )

    print(f"Point Accuracy : {point_accuracy:.4f}")
    print(f"Instance mIoU  : {instance_miou:.4f}")
    print(
        f"Valid labels   : "
        f"{valid_part_indices.tolist()}"
    )

    print_label_summary(
        title="Ground Truth Summary",
        labels=target_labels_device,
    )

    print_label_summary(
        title="Prediction Summary",
        labels=predictions,
    )

    points_numpy = points.cpu().numpy()
    # points_numpy : (N, 3)

    target_numpy = (
        target_labels.cpu().numpy()
    )
    # target_numpy : (N,)

    prediction_numpy = (
        predictions.cpu().numpy()
    )
    # prediction_numpy : (N,)

    ground_truth_cloud = (
        create_colored_point_cloud(
            points=points_numpy,
            labels=target_numpy,
        )
    )

    prediction_cloud = (
        create_colored_point_cloud(
            points=points_numpy,
            labels=prediction_numpy,
        )
    )

    print_subheader("Visualization")

    print(
        "Ground Truthを閉じると、"
        "Predictionを表示します。"
    )

    visualize_point_cloud(
        point_cloud=ground_truth_cloud,
        window_name=(
            "Ground Truth - "
            f"{class_names[category_index]}"
        ),
        point_size=args.point_size,
    )

    visualize_point_cloud(
        point_cloud=prediction_cloud,
        window_name=(
            "Prediction - "
            f"{class_names[category_index]}"
        ),
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()