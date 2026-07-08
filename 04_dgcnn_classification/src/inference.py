"""
inference.py

学習済みDGCNN分類モデルで1つの点群を分類する
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from common.device import get_device, print_device_info
from common.point_io import load_point_cloud
from common.point_utils import (
    normalize_points_numpy,
    sample_points_numpy,
)
from common.run_utils import timer
from common.train_utils import load_checkpoint
from common.utils import print_header, print_subheader

from config import (
    CHECKPOINT_DIR,
    CHECKPOINT_NAME,
    CUDA_ID,
    DATASET_TYPE,
    NUM_CLASSES,
    NUM_POINTS,
)
from dataset import get_datasets
from model import DGCNNClassifier


DEFAULT_INPUT_PATH = (
    "../../02_pointnet_classification/data/ModelNet10/raw/chair/test/chair_0890.off"
    if DATASET_TYPE == "modelnet10"
    else "../../02_pointnet_classification/data/ModelNet40/raw/chair/test/chair_0890.off"
)


def load_off_points(file_path: str) -> np.ndarray:
    """
    OFFファイルから頂点座標を読み込む。
    """
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

        if first_line != "OFF":
            raise ValueError(f"Invalid OFF file: {file_path}")

        counts = f.readline().strip()

        while counts.startswith("#") or counts == "":
            counts = f.readline().strip()

        num_vertices = int(counts.split()[0])

        vertices = []

        for _ in range(num_vertices):
            values = f.readline().strip().split()
            vertices.append(
                [
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                ]
            )

    return np.asarray(vertices, dtype=np.float32)


def load_points(file_path: str) -> np.ndarray:
    """
    入力ファイルから点群座標を読み込む。
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == ".off":
        return load_off_points(file_path)

    point_cloud = load_point_cloud(file_path)
    return np.asarray(point_cloud.points)


def preprocess_points(points: np.ndarray) -> torch.Tensor:
    """
    DGCNN入力用に点群を前処理する。
    """
    points = sample_points_numpy(
        points,
        NUM_POINTS,
    )

    points = normalize_points_numpy(points)

    points = torch.tensor(
        points,
        dtype=torch.float32,
    )

    points = points.unsqueeze(0)

    return points


def get_class_names(dataset_type: str) -> list[str]:
    """
    Datasetからクラス名一覧を取得する。
    """
    train_dataset, test_dataset = get_datasets(
        dataset_type=dataset_type,
    )

    if hasattr(test_dataset, "class_names"):
        return test_dataset.class_names

    if hasattr(train_dataset, "class_names"):
        return train_dataset.class_names

    raise AttributeError(
        "Dataset does not have class_names property."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="DGCNN inference"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Input point cloud file path",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(CHECKPOINT_DIR) / CHECKPOINT_NAME),
        help="Checkpoint file path",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default=DATASET_TYPE,
        choices=["modelnet10", "modelnet40", "folder"],
        help="Dataset type",
    )

    return parser.parse_args()


@timer
def main():
    print_header("DGCNN Inference")

    args = parse_args()

    input_path = Path(args.input)
    checkpoint_path = Path(args.checkpoint)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    device = get_device(cuda_id=CUDA_ID)
    print_device_info(cuda_id=CUDA_ID)

    print_subheader("Load Point Cloud")
    print(f"Input : {input_path}")

    points = load_points(str(input_path))
    points = preprocess_points(points)
    points = points.to(device)

    print(f"Input tensor shape : {points.shape}")

    print_subheader("Load Model")

    model = DGCNNClassifier(
        num_classes=NUM_CLASSES,
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

    print_subheader("Load Class Names")

    class_names = get_class_names(
        dataset_type=args.dataset_type,
    )

    print(f"Number of classes : {len(class_names)}")

    print_subheader("Predict")

    with torch.no_grad():
        logits = model(points)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        confidence = probabilities[
            0,
            predicted_index,
        ].item()

    print(f"Predicted class : {class_names[predicted_index]}")
    print(f"Class index     : {predicted_index}")
    print(f"Confidence      : {confidence:.4f}")


if __name__ == "__main__":
    main()