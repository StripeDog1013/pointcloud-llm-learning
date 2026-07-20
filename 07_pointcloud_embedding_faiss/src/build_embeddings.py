"""
build_embeddings.py

学習済みPointNet++を使用して、
ModelNetデータセットのEmbeddingを生成する。
"""

import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CUDA_ID,
    EMBEDDING_FILE,
    INDEX_FILE,
    LABEL_FILE,
    NORMALIZE_EMBEDDING,
    NUM_CLASSES,
    NUM_WORKERS,
    PHYSICAL_CUDA_ID,
    POINTNET2_CHECKPOINT_PATH,
    RANDOM_SEED,
    USE_CUDA_VISIBLE_DEVICES,
)


# ==============================================================================
# GPU設定
# ==============================================================================

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )


from model import PointNet2Classifier  # noqa: E402

from dataset import get_all_dataset  # noqa: E402
from embedding_model import (  # noqa: E402
    create_embedding_model,
    freeze_backbone,
)


# ==============================================================================
# 乱数固定
# ==============================================================================

def set_random_seed(
    seed: int,
) -> None:
    """
    再現性のため乱数シードを固定する。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# デバイス
# ==============================================================================

def get_device() -> torch.device:
    """
    使用するデバイスを取得する。
    """
    if torch.cuda.is_available():
        return torch.device(
            f"cuda:{CUDA_ID}"
        )

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ==============================================================================
# Checkpoint
# ==============================================================================

def extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    """
    Checkpointからstate_dictを取得する。

    次の保存形式に対応する。

    - torch.save(model.state_dict(), path)
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - {"model": ...}
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        state_dict = checkpoint.get(key)

        if isinstance(state_dict, dict):
            return state_dict

    if all(
        isinstance(value, torch.Tensor)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise KeyError(
        "Checkpointからstate_dictを取得できません。"
    )


def remove_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    DataParallel/DDPで付加されたmodule.を削除する。
    """
    return {
        (
            key.removeprefix("module.")
        ): value
        for key, value in state_dict.items()
    }


def load_backbone(
    device: torch.device,
) -> PointNet2Classifier:
    """
    学習済みPointNet++分類モデルを読み込む。
    """
    checkpoint_path = Path(
        POINTNET2_CHECKPOINT_PATH
    ).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpointが見つかりません。"
            f"\npath: {checkpoint_path}"
        )

    model = PointNet2Classifier(
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = remove_module_prefix(
        state_dict
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


# ==============================================================================
# Embedding抽出
# ==============================================================================

@torch.inference_mode()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    データセット全体からEmbeddingを抽出する。

    Returns
    -------
    embeddings:
        shape = (データ数, 1024)

    labels:
        shape = (データ数,)

    indices:
        shape = (データ数,)
    """
    embedding_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    for points, labels in tqdm(
        dataloader,
        desc="Embedding生成",
    ):
        points = points.to(
            device,
            non_blocking=True,
        )
        # (B, N, 3)

        embeddings = model(points)
        # (B, N, 3) -> (B, 1024)

        embedding_batches.append(
            embeddings
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        label_batches.append(
            labels
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )

    if not embedding_batches:
        raise RuntimeError(
            "Embeddingを1件も生成できませんでした。"
        )

    embeddings = np.concatenate(
        embedding_batches,
        axis=0,
    )
    # [(B, 1024), ...] -> (全データ数, 1024)

    labels = np.concatenate(
        label_batches,
        axis=0,
    )
    # [(B,), ...] -> (全データ数,)

    indices = np.arange(
        len(embeddings),
        dtype=np.int64,
    )
    # (全データ数,)

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )

    return (
        embeddings,
        labels,
        indices,
    )


# ==============================================================================
# 保存
# ==============================================================================

def save_embedding_data(
    embeddings: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> None:
    """
    Embedding・ラベル・Datasetインデックスを保存する。
    """
    embedding_path = Path(
        EMBEDDING_FILE
    )
    label_path = Path(
        LABEL_FILE
    )
    index_path = Path(
        INDEX_FILE
    )

    for path in (
        embedding_path,
        label_path,
        index_path,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    np.save(
        embedding_path,
        embeddings,
    )

    np.save(
        label_path,
        labels,
    )

    np.save(
        index_path,
        indices,
    )


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    """
    Embedding生成処理を実行する。
    """
    set_random_seed(
        RANDOM_SEED
    )

    device = get_device()

    print("=" * 70)
    print("PointNet++ Embedding生成")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Checkpoint : {POINTNET2_CHECKPOINT_PATH}")

    dataset = get_all_dataset()

    print(f"Dataset数  : {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=False,
    )

    backbone = load_backbone(
        device=device,
    )

    embedding_model = create_embedding_model(
        backbone=backbone,
        normalize=NORMALIZE_EMBEDDING,
    )

    freeze_backbone(
        embedding_model
    )

    embedding_model.to(device)
    embedding_model.eval()

    embeddings, labels, indices = (
        extract_embeddings(
            model=embedding_model,
            dataloader=dataloader,
            device=device,
        )
    )

    save_embedding_data(
        embeddings=embeddings,
        labels=labels,
        indices=indices,
    )

    print("-" * 70)
    print(
        f"Embeddings : {embeddings.shape} "
        f"{embeddings.dtype}"
    )
    print(
        f"Labels     : {labels.shape} "
        f"{labels.dtype}"
    )
    print(
        f"Indices    : {indices.shape} "
        f"{indices.dtype}"
    )
    print("-" * 70)
    print(f"保存先: {Path(EMBEDDING_FILE).parent}")
    print("Embedding生成が完了しました。")


if __name__ == "__main__":
    main()