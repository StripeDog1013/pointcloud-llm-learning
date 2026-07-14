"""
train.py

PointNet++ Part Segmentationモデルの学習。

検証指標:
- Validation Loss
- Point Accuracy
- Instance mIoU
- Category mIoU

Validation Instance mIoUが最良のモデルを保存する。
"""

import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# common/ をimportできるようにする
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
    save_checkpoint,
    seed_everything,
)
from common.utils import (
    print_header,
    print_subheader,
)

from config import (
    BATCH_SIZE,
    CATEGORIES,
    CHECKPOINT_DIR,
    CHECKPOINT_NAME,
    CUDA_ID,
    DATASET_TYPE,
    DROPOUT,
    EPOCHS,
    FP_MLP_CHANNELS,
    GLOBAL_SA_MLP,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_PART_CLASSES,
    NUM_POINTS,
    RANDOM_SEED,
    SA_CONFIG,
    WEIGHT_DECAY,
)
from dataset import get_dataloaders
from model import PointNet2PartSegmentation


def calculate_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    点単位のCrossEntropyLossを計算する。

    Parameters
    ----------
    logits:
        (B, N, NUM_PART_CLASSES)

    labels:
        (B, N)
    """
    logits_flat = logits.reshape(
        -1,
        NUM_PART_CLASSES,
    )
    # (B, N, NUM_PART_CLASSES)
    # -> (B * N, NUM_PART_CLASSES)

    labels_flat = labels.reshape(-1)
    # (B, N)
    # -> (B * N,)

    return criterion(
        logits_flat,
        labels_flat,
    )


def predict_valid_parts(
    logits: torch.Tensor,
    categories: torch.Tensor,
    part_label_mask: torch.Tensor,
) -> torch.Tensor:
    """
    各物体カテゴリで有効な部品ラベルだけを対象に予測する。

    Parameters
    ----------
    logits:
        (B, N, NUM_PART_CLASSES)

    categories:
        (B,)

    part_label_mask:
        (NUM_CLASSES, NUM_PART_CLASSES)

    Returns
    -------
    predictions:
        (B, N)
    """
    batch_size, num_points, _ = logits.shape

    predictions = torch.empty(
        batch_size,
        num_points,
        dtype=torch.long,
        device=logits.device,
    )
    # predictions : (B, N)

    for batch_index in range(batch_size):
        category_index = categories[
            batch_index
        ].item()

        valid_part_indices = torch.nonzero(
            part_label_mask[category_index],
            as_tuple=False,
        ).squeeze(1)
        # valid_part_indices : (P,)

        category_logits = logits[
            batch_index,
            :,
            valid_part_indices,
        ]
        # logits           : (B, N, NUM_PART_CLASSES)
        # category_logits : (N, P)

        local_predictions = category_logits.argmax(
            dim=-1
        )
        # (N, P) -> (N,)

        predictions[batch_index] = (
            valid_part_indices[local_predictions]
        )
        # local label (N,)
        # -> global part label (N,)

    return predictions


def calculate_instance_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_part_indices: torch.Tensor,
) -> float:
    """
    1つの点群について部品IoUの平均を計算する。

    Parameters
    ----------
    prediction:
        (N,)

    target:
        (N,)

    valid_part_indices:
        (P,)
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

        if union == 0:
            part_iou = 1.0
        else:
            part_iou = intersection / union

        part_ious.append(part_iou)

    return sum(part_ious) / len(part_ious)


def calculate_segmentation_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int]:
    """
    点単位の正解数と総点数を返す。

    predictions:
        (B, N)

    labels:
        (B, N)
    """
    correct = (
        predictions == labels
    ).sum().item()

    total = labels.numel()

    return correct, total


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    part_label_mask: torch.Tensor,
) -> tuple[float, float]:
    """
    1エポック分の学習を実行する。
    """
    model.train()

    part_label_mask = part_label_mask.to(device)
    # (NUM_CLASSES, NUM_PART_CLASSES)

    total_loss = 0.0
    total_correct = 0
    total_points = 0
    total_samples = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Train [{epoch:03d}/{EPOCHS}]",
    )

    for (
        points,
        categories,
        part_labels,
        features,
    ) in progress_bar:
        # points      : (B, N, 3)
        # categories  : (B,)
        # part_labels : (B, N)
        # features    : (B, N, F) または (B, N, 0)

        points = points.to(
            device,
            non_blocking=True,
        )
        # points : (B, N, 3)

        categories = categories.to(
            device,
            non_blocking=True,
        )
        # categories : (B,)

        part_labels = part_labels.to(
            device,
            non_blocking=True,
        )
        # part_labels : (B, N)

        features = features.to(
            device,
            non_blocking=True,
        )
        # features : (B, N, F)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            points=points,
            categories=categories,
            features=features,
        )
        # logits : (B, N, NUM_PART_CLASSES)

        loss = calculate_loss(
            criterion=criterion,
            logits=logits,
            labels=part_labels,
        )

        loss.backward()
        optimizer.step()

        predictions = predict_valid_parts(
            logits=logits.detach(),
            categories=categories,
            part_label_mask=part_label_mask,
        )
        # predictions : (B, N)

        correct, num_points = (
            calculate_segmentation_accuracy(
                predictions=predictions,
                labels=part_labels,
            )
        )

        batch_size = points.shape[0]

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_points += num_points
        total_samples += batch_size

        batch_accuracy = (
            correct / num_points
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{batch_accuracy:.4f}",
        )

    average_loss = total_loss / total_samples
    average_accuracy = total_correct / total_points

    return average_loss, average_accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    part_label_mask: torch.Tensor,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    検証データで各評価指標を計算する。

    Returns
    -------
    average_loss:
        検証Loss

    point_accuracy:
        全点の正解率

    instance_miou:
        各点群のmIoUを全インスタンスで平均

    category_miou:
        カテゴリごとの平均IoUをカテゴリ間で平均
    """
    model.eval()

    part_label_mask = part_label_mask.to(device)
    # (NUM_CLASSES, NUM_PART_CLASSES)

    total_loss = 0.0
    total_correct = 0
    total_points = 0
    total_samples = 0

    instance_ious = []
    category_instance_ious = defaultdict(list)

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
    )

    with torch.no_grad():
        for (
            points,
            categories,
            part_labels,
            features,
        ) in progress_bar:
            # points      : (B, N, 3)
            # categories  : (B,)
            # part_labels : (B, N)
            # features    : (B, N, F)

            points = points.to(
                device,
                non_blocking=True,
            )
            # points : (B, N, 3)

            categories = categories.to(
                device,
                non_blocking=True,
            )
            # categories : (B,)

            part_labels = part_labels.to(
                device,
                non_blocking=True,
            )
            # part_labels : (B, N)

            features = features.to(
                device,
                non_blocking=True,
            )
            # features : (B, N, F)

            logits = model(
                points=points,
                categories=categories,
                features=features,
            )
            # logits : (B, N, NUM_PART_CLASSES)

            loss = calculate_loss(
                criterion=criterion,
                logits=logits,
                labels=part_labels,
            )

            predictions = predict_valid_parts(
                logits=logits,
                categories=categories,
                part_label_mask=part_label_mask,
            )
            # predictions : (B, N)

            correct, num_points = (
                calculate_segmentation_accuracy(
                    predictions=predictions,
                    labels=part_labels,
                )
            )

            batch_size = points.shape[0]

            total_loss += loss.item() * batch_size
            total_correct += correct
            total_points += num_points
            total_samples += batch_size

            for batch_index in range(batch_size):
                category_index = categories[
                    batch_index
                ].item()

                valid_part_indices = torch.nonzero(
                    part_label_mask[
                        category_index
                    ],
                    as_tuple=False,
                ).squeeze(1)
                # valid_part_indices : (P,)

                instance_iou = (
                    calculate_instance_iou(
                        prediction=predictions[
                            batch_index
                        ],
                        target=part_labels[
                            batch_index
                        ],
                        valid_part_indices=(
                            valid_part_indices
                        ),
                    )
                )

                instance_ious.append(
                    instance_iou
                )

                category_instance_ious[
                    category_index
                ].append(
                    instance_iou
                )

            batch_accuracy = (
                correct / num_points
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{batch_accuracy:.4f}",
            )

    average_loss = total_loss / total_samples
    point_accuracy = total_correct / total_points

    instance_miou = (
        sum(instance_ious)
        / len(instance_ious)
    )

    category_ious = [
        sum(ious) / len(ious)
        for ious in category_instance_ious.values()
    ]

    category_miou = (
        sum(category_ious)
        / len(category_ious)
    )

    return (
        average_loss,
        point_accuracy,
        instance_miou,
        category_miou,
    )


@timer
def main() -> None:
    print_header(
        "Train PointNet++ Part Segmentation"
    )

    seed_everything(
        RANDOM_SEED
    )

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
        val_loader,
        _,
        train_dataset,
        val_dataset,
        test_dataset,
    ) = get_dataloaders(
        dataset_type=DATASET_TYPE,
    )

    print(f"Dataset type  : {DATASET_TYPE}")
    print(f"Categories    : {CATEGORIES}")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Num points    : {NUM_POINTS}")

    if not hasattr(
        train_dataset,
        "part_label_mask",
    ):
        raise AttributeError(
            "Dataset must provide part_label_mask."
        )

    if not hasattr(
        val_dataset,
        "part_label_mask",
    ):
        raise AttributeError(
            "Validation dataset must provide "
            "part_label_mask."
        )

    train_part_label_mask = (
        train_dataset.part_label_mask
    )
    # (NUM_CLASSES, NUM_PART_CLASSES)

    val_part_label_mask = (
        val_dataset.part_label_mask
    )
    # (NUM_CLASSES, NUM_PART_CLASSES)

    print_subheader("Create Model")

    model = PointNet2PartSegmentation(
        num_categories=NUM_CLASSES,
        num_part_classes=NUM_PART_CLASSES,
        input_feature_dim=0,
        dropout=DROPOUT,
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
        "categories": CATEGORIES,
        "num_categories": NUM_CLASSES,
        "num_part_classes": NUM_PART_CLASSES,
        "num_points": NUM_POINTS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "random_seed": RANDOM_SEED,
        "sa_config": SA_CONFIG,
        "global_sa_mlp": GLOBAL_SA_MLP,
        "fp_mlp_channels": FP_MLP_CHANNELS,
        "checkpoint_metric": (
            "validation_instance_miou"
        ),
    }

    log_path = save_json(
        data=train_params,
        prefix="train_params",
    )

    print(
        f"Saved train params : {log_path}"
    )

    best_val_instance_miou = 0.0

    print_subheader("Start Training")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        train_loss, train_accuracy = (
            train_one_epoch(
                epoch=epoch,
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                part_label_mask=(
                    train_part_label_mask
                ),
            )
        )

        (
            val_loss,
            val_accuracy,
            val_instance_miou,
            val_category_miou,
        ) = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            part_label_mask=(
                val_part_label_mask
            ),
        )

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f} "
            f"Val Ins mIoU: "
            f"{val_instance_miou:.4f} "
            f"Val Cat mIoU: "
            f"{val_category_miou:.4f}"
        )

        if (
            val_instance_miou
            > best_val_instance_miou
        ):
            best_val_instance_miou = (
                val_instance_miou
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                file_path=checkpoint_path,
            )

            print(
                "Saved best checkpoint : "
                f"{checkpoint_path}"
            )

    print_subheader("Training Finished")

    print(
        "Best validation Instance mIoU : "
        f"{best_val_instance_miou:.4f}"
    )


if __name__ == "__main__":
    main()