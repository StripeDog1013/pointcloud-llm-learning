"""
evaluate.py

学習済みPointNet分類モデルの評価
"""

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    CUDA_ID,
    NUM_CLASSES,
)
from dataset import get_dataloaders
from device import get_device, print_device_info
from model import (
    PointNetClassifier,
    feature_transform_regularizer,
)
from utils import (
    calculate_accuracy,
    load_checkpoint,
    print_header,
    print_subheader,
    timer,
)


REGULARIZATION_WEIGHT = 0.001


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
        dataloader
    )
    
    with torch.no_grad():
        for points, labels in progress_bar:
            points = points.to(device)
            labels = labels.to(device)

            logits, _, feature_transform = model(points)

            classification_loss = criterion(logits, labels)
            regularization_loss = feature_transform_regularizer(
                feature_transform
            )

            loss = (
                classification_loss
                + REGULARIZATION_WEIGHT * regularization_loss
            )

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
    print_header("Evaluate PointNet Classification")

    device = get_device(cuda_id=CUDA_ID)
    print_device_info(cuda_id=CUDA_ID)

    print_subheader("Create DataLoader")
    _, test_loader, _, test_dataset = get_dataloaders(
        dataset_type="modelnet10"
    )

    print(f"Test samples : {len(test_dataset)}")

    print_subheader("Load Model")
    model = PointNetClassifier(
        num_classes=NUM_CLASSES,
        use_feature_transform=True,
    ).to(device)

    checkpoint_path = Path(CHECKPOINT_DIR) / "pointnet_best.pth"

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