"""
train.py

PointNet分類モデルの学習
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import (
    DATASET_TYPE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_CLASSES,
    RANDOM_SEED,
    WEIGHT_DECAY,
    CUDA_ID,
)
from dataset import get_dataloaders
from device import get_device, print_device_info
from model import (
    PointNetClassifier,
    feature_transform_regularizer,
)
from utils import (
    calculate_accuracy,
    create_directory,
    save_checkpoint,
    save_json,
    seed_everything,
    print_header,
    print_subheader,
    timer,
)


REGULARIZATION_WEIGHT = 0.001


def train_one_epoch(
    epoch, 
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    total_accuracy = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch: 03d}",
    )
    
    for points, labels in progress_bar:
        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _, feature_transform = model(points)

        classification_loss = criterion(logits, labels)
        regularization_loss = feature_transform_regularizer(
            feature_transform
        )

        loss = (
            classification_loss
            + REGULARIZATION_WEIGHT * regularization_loss
        )

        loss.backward()
        optimizer.step()

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
    print_header("Train PointNet Classification")

    seed_everything(RANDOM_SEED)

    device = get_device(cuda_id=CUDA_ID)
    print_device_info(cuda_id=CUDA_ID)

    checkpoint_dir = create_directory(CHECKPOINT_DIR)

    print_subheader("Create DataLoader")
    train_loader, test_loader, train_dataset, test_dataset = get_dataloaders(
        dataset_type=DATASET_TYPE
    )

    print(f"Train samples : {len(train_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")
    print(f"Batch size    : {BATCH_SIZE}")

    print_subheader("Create Model")
    model = PointNetClassifier(
        num_classes=NUM_CLASSES,
        use_feature_transform=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_params = {
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "regularization_weight": REGULARIZATION_WEIGHT,
        "random_seed": RANDOM_SEED,
        "dataset": DATASET_TYPE,
    }

    log_path = save_json(
        data=train_params,
        prefix="train_params",
    )

    print(f"Saved train params : {log_path}")

    best_accuracy = 0.0
    best_checkpoint_path = (
        Path(checkpoint_dir) / "pointnet_best_10.pth"
        if DATASET_TYPE == "modelnet10"
        else Path(checkpoint_dir) / "pointnet_best_40.pth"
    )

    print_subheader("Start Training")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            epoch=epoch,
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        print(
            f"Loss      : {train_loss:.4f} "
            f"\nAccuracy: {train_accuracy:.4f}"
        )

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                file_path=best_checkpoint_path,
            )

            print(f"Saved best checkpoint : {best_checkpoint_path}")

    print_subheader("Training Finished")
    print(f"Best train accuracy : {best_accuracy:.4f}")


if __name__ == "__main__":
    main()