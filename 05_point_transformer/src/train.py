"""
train.py

階層型Point Transformer分類モデルの学習
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# レポジトリ直下のcommonパッケージを読み込めるようにする
sys.path.append(
    str(Path(__file__).resolve().parents[2])
)

from common.device import (
    get_device,
    print_device_info,
)
from common.log_utils import save_json
from common.path_utils import create_directory
from common.run_utils import timer
from common.train_utils import (
    calculate_accuracy,
    save_checkpoint,
    seed_everything,
)
from common.utils import (
    print_header,
    print_subheader,
)

from config import (
    ATTENTION_DROPOUT,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_NAME,
    CUDA_ID,
    DATASET_TYPE,
    DROPOUT,
    EPOCHS,
    K,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_POINTS,
    RANDOM_SEED,
    STAGE_DEPTHS,
    STAGE_DIMS,
    STAGE_NUM_POINTS,
    TRANSITION_K,
    WEIGHT_DECAY,
)
from dataset import get_dataloaders
from model import PointTransformerClassifier


def train_one_epoch(
    epoch,
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    1エポック分の学習を実行する。
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch:03d}/{EPOCHS}]",
    )

    for points, labels in progress_bar:
        points = points.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(points)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        predictions = logits.argmax(dim=1)
        correct = (
            predictions == labels
        ).sum().item()

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size

        batch_accuracy = calculate_accuracy(
            logits,
            labels,
        )

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
        "Train Point Transformer Classification"
    )

    seed_everything(RANDOM_SEED)

    device = get_device(
        cuda_id=CUDA_ID,
    )
    print_device_info(
        cuda_id=CUDA_ID,
    )

    checkpoint_dir = create_directory(
        CHECKPOINT_DIR
    )
    checkpoint_path = (
        Path(checkpoint_dir)
        / CHECKPOINT_NAME
    )

    print_subheader("Create DataLoader")

    (
        train_loader,
        test_loader,
        train_dataset,
        test_dataset,
    ) = get_dataloaders(
        dataset_type=DATASET_TYPE,
    )

    print(f"Dataset type  : {DATASET_TYPE}")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Num points    : {NUM_POINTS}")

    print_subheader("Create Model")

    model = PointTransformerClassifier(
        num_classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_params = {
        "model_name": MODEL_NAME,
        "dataset_type": DATASET_TYPE,
        "num_classes": NUM_CLASSES,
        "num_points": NUM_POINTS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "random_seed": RANDOM_SEED,
        "stage_num_points": STAGE_NUM_POINTS,
        "stage_dims": STAGE_DIMS,
        "stage_depths": STAGE_DEPTHS,
        "k": K,
        "transition_k": TRANSITION_K,
        "attention_dropout": ATTENTION_DROPOUT,
        "dropout": DROPOUT,
    }

    log_path = save_json(
        data=train_params,
        prefix="train_params",
    )

    print(
        f"Saved train params : {log_path}"
    )

    best_accuracy = 0.0

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
            # f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {train_accuracy:.4f}"
        )

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                file_path=checkpoint_path,
            )

            print(
                "Saved best checkpoint : "
                f"{checkpoint_path}"
            )

    print_subheader("Training Finished")
    print(
        f"Best train accuracy : "
        f"{best_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()