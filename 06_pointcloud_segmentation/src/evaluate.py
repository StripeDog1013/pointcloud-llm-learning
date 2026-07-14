"""
evaluate.py

学習済みPointNet++ Part Segmentationモデルを評価する。

評価指標:
- Test Loss
- Point Accuracy
- Instance mIoU
- Category mIoU
"""

import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

# common/ をimportできるようにする
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
    DROPOUT,
    NUM_CLASSES,
    NUM_PART_CLASSES,
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
    # (B, N) -> (B * N,)

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
    物体カテゴリで有効な部品ラベルだけを対象に予測する。

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
        # logits                 : (B, N, NUM_PART_CLASSES)
        # category_logits       : (N, P)

        local_predictions = category_logits.argmax(
            dim=-1
        )
        # (N, P) -> (N,)

        predictions[batch_index] = valid_part_indices[
            local_predictions
        ]
        # local index (N,) -> global part label (N,)

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

    Returns
    -------
    float
        1インスタンス分のmIoU
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
            # 正解にも予測にも存在しない部品はIoU=1とする
            part_iou = 1.0
        else:
            part_iou = intersection / union

        part_ious.append(part_iou)

    return sum(part_ious) / len(part_ious)


def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    part_label_mask: torch.Tensor,
    class_names: list[str],
) -> dict:
    """
    テストデータ全体を評価する。
    """
    model.eval()

    part_label_mask = part_label_mask.to(
        device
    )
    # part_label_mask : (NUM_CLASSES, NUM_PART_CLASSES)

    total_loss = 0.0
    total_correct = 0
    total_points = 0
    total_samples = 0

    instance_ious = []
    category_instance_ious = defaultdict(list)

    progress_bar = tqdm(
        dataloader,
        desc="Evaluate",
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

            batch_size = points.shape[0]
            num_batch_points = part_labels.numel()

            batch_correct = (
                predictions == part_labels
            ).sum().item()

            total_loss += loss.item() * batch_size
            total_correct += batch_correct
            total_points += num_batch_points
            total_samples += batch_size

            for batch_index in range(batch_size):
                category_index = categories[
                    batch_index
                ].item()

                valid_part_indices = torch.nonzero(
                    part_label_mask[category_index],
                    as_tuple=False,
                ).squeeze(1)
                # valid_part_indices : (P,)

                instance_iou = calculate_instance_iou(
                    prediction=predictions[batch_index],
                    target=part_labels[batch_index],
                    valid_part_indices=valid_part_indices,
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
                batch_correct
                / num_batch_points
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{batch_accuracy:.4f}",
            )

    average_loss = (
        total_loss / total_samples
    )

    point_accuracy = (
        total_correct / total_points
    )

    instance_miou = (
        sum(instance_ious)
        / len(instance_ious)
    )

    category_ious = {}

    for category_index, ious in (
        category_instance_ious.items()
    ):
        category_name = class_names[
            category_index
        ]

        category_ious[category_name] = (
            sum(ious) / len(ious)
        )

    category_miou = (
        sum(category_ious.values())
        / len(category_ious)
    )

    return {
        "loss": average_loss,
        "point_accuracy": point_accuracy,
        "instance_miou": instance_miou,
        "category_miou": category_miou,
        "category_ious": category_ious,
    }


@timer
def main() -> None:
    print_header(
        "Evaluate PointNet++ Part Segmentation"
    )

    device = get_device(
        cuda_id=CUDA_ID,
    )

    print_device_info(
        cuda_id=CUDA_ID,
    )

    print_subheader("Create DataLoader")

    (
        _,
        _,
        test_loader,
        _,
        _,
        test_dataset,
    ) = get_dataloaders(
        dataset_type=DATASET_TYPE,
    )

    print(f"Dataset type : {DATASET_TYPE}")
    print(f"Test samples : {len(test_dataset)}")
    print(f"Categories   : {test_dataset.class_names}")

    if not hasattr(
        test_dataset,
        "part_label_mask",
    ):
        raise AttributeError(
            "Dataset must provide part_label_mask "
            "to calculate ShapeNet Part mIoU."
        )

    part_label_mask = (
        test_dataset.part_label_mask
    )
    # part_label_mask : (NUM_CLASSES, NUM_PART_CLASSES)

    class_names = list(
        test_dataset.class_names
    )

    print_subheader("Load Model")

    model = PointNet2PartSegmentation(
        num_categories=NUM_CLASSES,
        num_part_classes=NUM_PART_CLASSES,
        input_feature_dim=0,
        dropout=DROPOUT,
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

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        part_label_mask=part_label_mask,
        class_names=class_names,
    )

    print_subheader("Evaluation Result")

    print(
        f"Test Loss     : "
        f"{metrics['loss']:.4f}"
    )
    print(
        f"Point Accuracy: "
        f"{metrics['point_accuracy']:.4f}"
    )
    print(
        f"Instance mIoU : "
        f"{metrics['instance_miou']:.4f}"
    )
    print(
        f"Category mIoU : "
        f"{metrics['category_miou']:.4f}"
    )

    print_subheader("Category IoU")

    for category_name, category_iou in sorted(
        metrics["category_ious"].items()
    ):
        print(
            f"{category_name:<12}: "
            f"{category_iou:.4f}"
        )


if __name__ == "__main__":
    main()