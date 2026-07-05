"""
inference.py

学習済みPointNet分類モデルで1つの点群を分類する
"""

from pathlib import Path
import argparse

import numpy as np
import torch

from config import (
    DATASET_TYPE,
    CHECKPOINT_DIR,
    CUDA_ID,
    NUM_CLASSES,
    NUM_POINTS,
)
from dataset import get_datasets
from dataset_utils import (
    normalize_points_numpy,
    sample_points_numpy,
)
from device import get_device, print_device_info
from model import PointNetClassifier
from utils import (
    load_checkpoint,
    load_point_cloud,
    print_header,
    print_subheader,
    timer,
)


DEFAULT_INPUT_PATH = "../data/raw/chair/test/chair_0890.off"
DEFAULT_CHECKPOINT_PATH = Path(CHECKPOINT_DIR) / "pointnet_best.pth"



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
    PointNet入力用に前処理する。
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="PointNet inference"
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
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint file path",
    )

    return parser.parse_args()


@timer
def main():
    print_header("PointNet Inference")

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

    model = PointNetClassifier(
        num_classes=NUM_CLASSES,
        use_feature_transform=True,
    ).to(device)

    checkpoint = load_checkpoint(
        model=model,
        optimizer=None,
        file_path=checkpoint_path,
    )

    model.eval()

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Epoch      : {checkpoint['epoch']}")

    print_subheader("Predict")

    print_subheader("Load Class Names")

    _, test_dataset = get_datasets(
        dataset_type=DATASET_TYPE,
    )

    class_names = test_dataset.class_names

    print(f"Number of classes : {len(class_names)}")

    with torch.no_grad():
        logits, _, _ = model(points)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    print(f"Predicted class : {class_names[pred_idx]}")
    print(f"Class index     : {pred_idx}")
    print(f"Confidence      : {confidence:.4f}")


if __name__ == "__main__":
    main()