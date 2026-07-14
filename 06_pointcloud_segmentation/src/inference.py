"""
inference.py

学習済みPointNet++ Part Segmentationモデルで
点群の各点に部品ラベルを予測する。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# common/ をimportできるようにする
sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

from common.device import (
    get_device,
    print_device_info,
)
from common.point_io import load_point_cloud
from common.point_utils import (
    normalize_points_numpy,
    sample_points_numpy,
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
    NUM_POINTS,
)
from dataset import get_datasets
from model import PointNet2PartSegmentation


def load_off_points(
    file_path: str | Path,
) -> np.ndarray:
    """
    OFFファイルから頂点座標を読み込む。

    Returns
    -------
    np.ndarray
        shape = (N, 3)
    """
    file_path = Path(file_path)

    with file_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        first_line = file.readline().strip()

        if first_line == "OFF":
            counts_line = file.readline().strip()
        elif first_line.startswith("OFF"):
            counts_line = first_line[3:].strip()
        else:
            raise ValueError(
                f"Invalid OFF file: {file_path}"
            )

        while (
            not counts_line
            or counts_line.startswith("#")
        ):
            counts_line = file.readline().strip()

        num_vertices = int(
            counts_line.split()[0]
        )

        vertices = []

        while len(vertices) < num_vertices:
            line = file.readline()

            if not line:
                raise ValueError(
                    f"Unexpected end of OFF file: {file_path}"
                )

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            values = line.split()

            vertices.append(
                [
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                ]
            )

    points = np.asarray(
        vertices,
        dtype=np.float32,
    )
    # points : (N, 3)

    return points


def load_points(
    file_path: str | Path,
) -> np.ndarray:
    """
    OFF・PLY・PCD・LAS・LAZから点座標を読み込む。

    Returns
    -------
    np.ndarray
        shape = (N, 3)
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".off":
        points = load_off_points(
            file_path
        )
        # points : (N, 3)
    else:
        point_cloud = load_point_cloud(
            file_path
        )

        points = np.asarray(
            point_cloud.points,
            dtype=np.float32,
        )
        # points : (N, 3)

    if len(points) == 0:
        raise ValueError(
            f"Point cloud has no points: {file_path}"
        )

    return points


def preprocess_points(
    points: np.ndarray,
) -> torch.Tensor:
    """
    モデル入力用に点群を前処理する。

    Parameters
    ----------
    points:
        shape = (N, 3)

    Returns
    -------
    torch.Tensor
        shape = (1, NUM_POINTS, 3)
    """
    points = sample_points_numpy(
        points=points,
        num_points=NUM_POINTS,
    )
    # (N, 3) -> (NUM_POINTS, 3)

    points = normalize_points_numpy(
        points
    )
    # points : (NUM_POINTS, 3)

    points_tensor = torch.from_numpy(
        points
    ).float()
    # points_tensor : (NUM_POINTS, 3)

    points_tensor = points_tensor.unsqueeze(0)
    # (NUM_POINTS, 3)
    # -> (1, NUM_POINTS, 3)

    return points_tensor


def get_category_index(
    category: str,
    class_names: list[str],
) -> int:
    """
    カテゴリ名またはカテゴリ番号をindexへ変換する。
    """
    if category.isdigit():
        category_index = int(category)

        if not 0 <= category_index < len(class_names):
            raise ValueError(
                f"Category index must be between "
                f"0 and {len(class_names) - 1}."
            )

        return category_index

    normalized_names = {
        name.lower(): index
        for index, name in enumerate(class_names)
    }

    category_key = category.lower()

    if category_key not in normalized_names:
        raise ValueError(
            f"Unknown category: {category}\n"
            f"Available categories: {class_names}"
        )

    return normalized_names[category_key]


def predict_valid_parts(
    logits: torch.Tensor,
    category_index: int,
    part_label_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    指定カテゴリで有効な部品だけを対象に予測する。

    Parameters
    ----------
    logits:
        shape = (1, N, NUM_PART_CLASSES)

    part_label_mask:
        shape = (NUM_CLASSES, NUM_PART_CLASSES)

    Returns
    -------
    predictions:
        shape = (N,)

    probabilities:
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

    category_probabilities = torch.softmax(
        category_logits,
        dim=-1,
    )
    # category_probabilities : (N, P)

    confidence, local_predictions = torch.max(
        category_probabilities,
        dim=-1,
    )
    # confidence        : (N,)
    # local_predictions : (N,)

    predictions = valid_part_indices[
        local_predictions
    ]
    # local_predictions : (N,)
    # predictions       : (N,) グローバル部品ラベル

    return predictions, confidence


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
        predicted_mask = prediction == part_index
        # predicted_mask : (N,)

        target_mask = target == part_index
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="PointNet++ Part Segmentation inference"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Input point cloud path. "
            "If omitted, a test dataset sample is used."
        ),
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help=(
            "Object category name or index. "
            "Required when --input is specified."
        ),
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

    return parser.parse_args()


@timer
def main() -> None:
    print_header(
        "PointNet++ Part Segmentation Inference"
    )

    args = parse_args()

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print_subheader("Load Dataset Information")

    (
        _,
        _,
        test_dataset,
    ) = get_datasets(
        dataset_type=DATASET_TYPE,
    )

    class_names = list(
        test_dataset.class_names
    )

    if not hasattr(
        test_dataset,
        "part_label_mask",
    ):
        raise AttributeError(
            "Dataset must provide part_label_mask."
        )

    part_label_mask = (
        test_dataset.part_label_mask
    )
    # part_label_mask : (NUM_CLASSES, NUM_PART_CLASSES)

    target_labels = None

    if args.input is None:
        if not 0 <= args.index < len(test_dataset):
            raise IndexError(
                f"index must be between 0 and "
                f"{len(test_dataset) - 1}."
            )

        (
            points,
            category,
            target_labels,
            features,
        ) = test_dataset[args.index]

        # points        : (NUM_POINTS, 3)
        # category      : scalar
        # target_labels : (NUM_POINTS,)
        # features      : (NUM_POINTS, F)

        category_index = int(
            category.item()
        )

        points = points.unsqueeze(0)
        # (NUM_POINTS, 3)
        # -> (1, NUM_POINTS, 3)

        features = features.unsqueeze(0)
        # (NUM_POINTS, F)
        # -> (1, NUM_POINTS, F)

        print(f"Dataset index : {args.index}")
    else:
        input_path = Path(
            args.input
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}"
            )

        if args.category is None:
            raise ValueError(
                "--category is required when "
                "--input is specified."
            )

        category_index = get_category_index(
            category=args.category,
            class_names=class_names,
        )

        points_numpy = load_points(
            input_path
        )
        # points_numpy : (N, 3)

        points = preprocess_points(
            points_numpy
        )
        # points : (1, NUM_POINTS, 3)

        features = torch.empty(
            1,
            NUM_POINTS,
            0,
            dtype=torch.float32,
        )
        # features : (1, NUM_POINTS, 0)

        print(f"Input file    : {input_path}")

    categories = torch.tensor(
        [category_index],
        dtype=torch.long,
    )
    # categories : (1,)

    print(f"Category      : {class_names[category_index]}")
    print(f"Points shape  : {points.shape}")
    print(f"Features shape: {features.shape}")

    device = get_device(
        cuda_id=CUDA_ID,
    )

    print_device_info(
        cuda_id=CUDA_ID,
    )

    points = points.to(
        device
    )
    # points : (1, NUM_POINTS, 3)

    categories = categories.to(
        device
    )
    # categories : (1,)

    features = features.to(
        device
    )
    # features : (1, NUM_POINTS, F)

    part_label_mask = part_label_mask.to(
        device
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
            points=points,
            categories=categories,
            features=features,
        )
        # logits : (1, NUM_POINTS, NUM_PART_CLASSES)

        predictions, confidence = (
            predict_valid_parts(
                logits=logits,
                category_index=category_index,
                part_label_mask=part_label_mask,
            )
        )
        # predictions : (NUM_POINTS,)
        # confidence  : (NUM_POINTS,)

    valid_part_indices = torch.nonzero(
        part_label_mask[category_index],
        as_tuple=False,
    ).squeeze(1)
    # valid_part_indices : (P,)

    print(f"Valid part labels : {valid_part_indices.tolist()}")
    print(
        f"Mean confidence   : "
        f"{confidence.mean().item():.4f}"
    )

    unique_labels, counts = torch.unique(
        predictions,
        return_counts=True,
    )
    # unique_labels : (U,)
    # counts        : (U,)

    print_subheader("Prediction Summary")

    for part_index, count in zip(
        unique_labels.tolist(),
        counts.tolist(),
    ):
        ratio = count / predictions.numel()

        print(
            f"Part {part_index:02d}: "
            f"{count:5d} points "
            f"({ratio:.2%})"
        )

    if target_labels is not None:
        target_labels = target_labels.to(
            device
        )
        # target_labels : (NUM_POINTS,)

        point_accuracy = (
            predictions == target_labels
        ).float().mean().item()

        instance_miou = calculate_instance_iou(
            prediction=predictions,
            target=target_labels,
            valid_part_indices=valid_part_indices,
        )

        print_subheader("Dataset Sample Metrics")

        print(
            f"Point Accuracy : {point_accuracy:.4f}"
        )
        print(
            f"Instance mIoU  : {instance_miou:.4f}"
        )


if __name__ == "__main__":
    main()