"""
train_utils.py

学習処理関連の共通ユーティリティ
"""

from pathlib import Path
import random

import numpy as np
import torch

from common.path_utils import ensure_parent_directory


def seed_everything(seed: int) -> None:
    """
    乱数シードを固定する。
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    分類精度を計算する。
    """
    predicted_label = prediction.argmax(dim=1)

    correct = (predicted_label == target).sum().item()
    total = target.size(0)

    return correct / total


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    loss: float,
    file_path: str | Path,
) -> None:
    """
    checkpointを保存する。
    """
    file_path = ensure_parent_directory(file_path)

    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, file_path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    file_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict:
    """
    checkpointを読み込む。
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {file_path}")

    checkpoint = torch.load(
        file_path,
        map_location=map_location,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint