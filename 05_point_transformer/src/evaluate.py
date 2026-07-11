"""
evaluate.py

学習済みPoint Transformer分類モデルの評価
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

# レポジトリ直下のcommonパッケージを読み込めるようにする
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
    NUM_CLASSES,
)
from dataset import get_dataloaders
from model import PointTransformerClassifier


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    テストデータ全体でLossとAccuracyを計算する。
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc="Evaluate",
    )

    with torch.no_grad():
        for points, labels in progress_bar:
            points = points.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(points)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            batch_correct = (
                predictions == labels
            ).sum().item()

            total_loss += loss.item() * batch_size
            total_correct += batch_correct
            total_samples += batch_size

            batch_accuracy = batch_correct / batch_size

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{batch_accuracy:.4f}",
            )

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_samples

    return average_loss, average_accuracy


@timer
def main():
    print_header(
        "Evaluate Point Transformer Classification"
    )

    device = get_device(
        cuda_id=CUDA_ID,
    )
    print_device_info(
        cuda_id=CUDA_ID,
    )

    print_subheader("Create DataLoader")

    _, test_loader, _, test_dataset = get_dataloaders(
        dataset_type=DATASET_TYPE,
    )

    print(f"Dataset type : {DATASET_TYPE}")
    print(f"Test samples : {len(test_dataset)}")

    print_subheader("Load Model")

    model = PointTransformerClassifier(
        num_classes=NUM_CLASSES,
    ).to(device)

    checkpoint_path = (
        Path(CHECKPOINT_DIR)
        / CHECKPOINT_NAME
    )

    checkpoint = load_checkpoint(
        model=model,
        optimizer=None,
        file_path=checkpoint_path,
        map_location=device,
    )

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Epoch      : {checkpoint['epoch']}")
    print(f"Loss       : {checkpoint['loss']:.4f}")

    criterion = nn.CrossEntropyLoss()

    print_subheader("Start Evaluation")

    test_loss, test_accuracy = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print_subheader("Evaluation Result")

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")


if __name__ == "__main__":
    main()