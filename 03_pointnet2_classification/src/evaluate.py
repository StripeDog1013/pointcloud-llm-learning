"""
evaluate.py

学習済みPointNet++分類モデルの評価
"""

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    CHECKPOINT_NAME,
    CUDA_ID,
    DATASET_TYPE,
    NUM_CLASSES,
)
from dataset import get_dataloaders
from device import get_device, print_device_info
from model import PointNet2Classifier
from utils import (
    calculate_accuracy,
    load_checkpoint,
    print_header,
    print_subheader,
    timer,
)


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Evaluate",
    )

    with torch.no_grad():
        for points, labels in progress_bar:
            points = points.to(device)
            labels = labels.to(device)

            logits = model(points)

            loss = criterion(logits, labels)
            accuracy = calculate_accuracy(logits, labels)

            total_loss += loss.item()
            total_accuracy += accuracy

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{accuracy:.4f}",
            )

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    return avg_loss, avg_accuracy


@timer
def main():
    print_header("Evaluate PointNet++ Classification")

    device = get_device(cuda_id=CUDA_ID)
    print_device_info(cuda_id=CUDA_ID)

    print_subheader("Create DataLoader")

    _, test_loader, _, test_dataset = get_dataloaders(
        dataset_type=DATASET_TYPE,
    )

    print(f"Dataset type : {DATASET_TYPE}")
    print(f"Test samples : {len(test_dataset)}")

    print_subheader("Load Model")

    model = PointNet2Classifier(
        num_classes=NUM_CLASSES,
    ).to(device)

    checkpoint_path = Path(CHECKPOINT_DIR) / CHECKPOINT_NAME

    checkpoint = load_checkpoint(
        model=model,
        optimizer=None,
        file_path=checkpoint_path,
    )

    print(f"Loaded checkpoint : {checkpoint_path}")
    print(f"Checkpoint epoch  : {checkpoint['epoch']}")
    print(f"Checkpoint loss   : {checkpoint['loss']:.4f}")

    criterion = nn.CrossEntropyLoss()

    print_subheader("Evaluate")

    test_loss, test_accuracy = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy:.4f}")


if __name__ == "__main__":
    main()