"""
inference.py

学習済みPoint Transformer分類モデルで
1つの点群ファイルを分類する
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

# レポジトリ直下のcommonパッケージを読み込めるようにする
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
    NUM_CLASSES,
    NUM_POINTS,
)
from dataset import get_datasets
from model import PointTransformerClassifier


DEFAULT_INPUT_PATH = (
    "../../02_pointnet_classification/data/ModelNet10/raw/chair/test/chair_0890.off"
    if DATASET_TYPE == "modelnet10"
    else "../../02_pointnet_classification/data/ModelNet40/raw/chair/test/chair_0890.off"
)

def load_off_points(
    file_path: str | Path,
) -> np.ndarray:
    """
    OFFファイルから頂点座標を読み込む。
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

            if len(values) < 3:
                raise ValueError(
                    f"Invalid vertex line: {line}"
                )

            vertices.append(
                [
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                ]
            )

    return np.asarray(
        vertices,
        dtype=np.float32,
    )


def load_points(
    file_path: str | Path,
) -> np.ndarray:
    """
    OFF・PLY・PCD・LAS・LAZから点座標を読み込む。
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".off":
        points = load_off_points(file_path)
    else:
        point_cloud = load_point_cloud(file_path)
        points = np.asarray(
            point_cloud.points,
            dtype=np.float32,
        )

    if len(points) == 0:
        raise ValueError(
            f"Point cloud has no points: {file_path}"
        )

    return points


def preprocess_points(
    points: np.ndarray,
) -> torch.Tensor:
    """
    Point Transformer入力用に前処理する。

    出力:
        (1, NUM_POINTS, 3)
    """
    points = sample_points_numpy(
        points=points,
        num_points=NUM_POINTS,
    )

    points = normalize_points_numpy(
        points
    )

    points_tensor = torch.from_numpy(
        points
    ).float()

    return points_tensor.unsqueeze(0)


def get_class_names(
    dataset_type: str,
) -> list[str]:
    """
    Datasetからクラス名一覧を取得する。
    """
    train_dataset, test_dataset = get_datasets(
        dataset_type=dataset_type,
    )

    for dataset in (
        test_dataset,
        train_dataset,
    ):
        if hasattr(dataset, "class_names"):
            return list(dataset.class_names)

    raise AttributeError(
        "Dataset does not provide class_names."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Point Transformer inference"
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
        default=str(
            Path(CHECKPOINT_DIR)
            / CHECKPOINT_NAME
        ),
        help="Checkpoint file path",
    )

    parser.add_argument(
        "--dataset_type",
        type=str,
        default=DATASET_TYPE,
        choices=[
            "modelnet10",
            "modelnet40",
            "folder",
        ],
        help="Dataset type",
    )

    return parser.parse_args()


@timer
def main():
    print_header("Point Transformer Inference")

    args = parse_args()

    input_path = Path(args.input)
    checkpoint_path = Path(args.checkpoint)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = get_device(
        cuda_id=CUDA_ID,
    )
    print_device_info(
        cuda_id=CUDA_ID,
    )

    print_subheader("Load Point Cloud")

    print(f"Input : {input_path}")

    points = load_points(
        input_path
    )

    points = preprocess_points(
        points
    ).to(device)

    print(f"Input tensor shape : {points.shape}")

    print_subheader("Load Model")

    model = PointTransformerClassifier(
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

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            "NUM_CLASSES does not match the dataset: "
            f"NUM_CLASSES={NUM_CLASSES}, "
            f"class_names={len(class_names)}"
        )

    print(f"Number of classes : {len(class_names)}")

    print_subheader("Predict")

    with torch.no_grad():
        logits = model(points)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1,
        )

    predicted_index = predicted_index.item()
    confidence = confidence.item()

    print(
        f"Predicted class : "
        f"{class_names[predicted_index]}"
    )
    print(f"Class index     : {predicted_index}")
    print(f"Confidence      : {confidence:.4f}")


if __name__ == "__main__":
    main()