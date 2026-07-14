"""
dataset_utils.py

点群セグメンテーションDataset用の共通処理
"""

from pathlib import Path
import sys

# common/ を import できるようにする
sys.path.append(str(Path(__file__).resolve().parents[2]))
from common.point_io import load_point_cloud

import numpy as np
import torch

POINT_CLOUD_EXTENSIONS = {
    ".ply",
    ".pcd",
    ".las",
    ".laz",
}


def normalize_points_tensor(
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Tensor点群を中心化し、単位球内に正規化する。

    Parameters
    ----------
    points : torch.Tensor
        (N, 3)

    Returns
    -------
    torch.Tensor
        (N, 3)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3), got {tuple(points.shape)}"
        )

    if points.shape[0] == 0:
        raise ValueError("Point cloud has no points.")

    centroid = points.mean(dim=0, keepdim=True)
    # centroid : (1, 3)

    points = points - centroid
    # (N, 3) - (1, 3) -> (N, 3)

    scale = torch.linalg.vector_norm(
        points,
        dim=1,
    ).max()
    # scale : scalar

    if scale > 0:
        points = points / scale
        # (N, 3)

    return points


def normalize_points_numpy(
    points: np.ndarray,
) -> np.ndarray:
    """
    NumPy点群を中心化し、単位球内に正規化する。

    Parameters
    ----------
    points : np.ndarray
        (N, 3)

    Returns
    -------
    np.ndarray
        (N, 3)
    """
    points = np.asarray(
        points,
        dtype=np.float32,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3), got {points.shape}"
        )

    if len(points) == 0:
        raise ValueError("Point cloud has no points.")

    centroid = points.mean(
        axis=0,
        keepdims=True,
    )
    # centroid : (1, 3)

    points = points - centroid
    # (N, 3) - (1, 3) -> (N, 3)

    scale = np.linalg.norm(
        points,
        axis=1,
    ).max()
    # scale : scalar

    if scale > 0:
        points = points / scale
        # (N, 3)

    return points.astype(
        np.float32,
        copy=False,
    )


def sample_points_and_labels_tensor(
    points: torch.Tensor,
    labels: torch.Tensor,
    num_points: int,
    features: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    点群・点単位ラベル・追加特徴を同じindexでサンプリングする。

    Parameters
    ----------
    points : torch.Tensor
        (N, 3)

    labels : torch.Tensor
        (N,)

    num_points : int
        サンプリング後の点数

    features : torch.Tensor | None
        (N, F)

    Returns
    -------
    sampled_points : torch.Tensor
        (num_points, 3)

    sampled_labels : torch.Tensor
        (num_points,)

    sampled_features : torch.Tensor | None
        (num_points, F)
    """
    num_source_points = points.shape[0]

    if num_source_points == 0:
        raise ValueError("Point cloud has no points.")

    if labels.shape[0] != num_source_points:
        raise ValueError(
            "points and labels must contain the same number of entries."
        )

    if features is not None and features.shape[0] != num_source_points:
        raise ValueError(
            "points and features must contain the same number of entries."
        )

    if num_source_points >= num_points:
        indices = torch.randperm(
            num_source_points,
            device=points.device,
        )[:num_points]
        # indices : (num_points,)
    else:
        indices = torch.randint(
            low=0,
            high=num_source_points,
            size=(num_points,),
            device=points.device,
        )
        # indices : (num_points,)

    sampled_points = points[indices]
    # (N, 3) -> (num_points, 3)

    sampled_labels = labels[indices]
    # (N,) -> (num_points,)

    sampled_features = None

    if features is not None:
        sampled_features = features[indices]
        # (N, F) -> (num_points, F)

    return (
        sampled_points,
        sampled_labels,
        sampled_features,
    )


def sample_points_and_labels_numpy(
    points: np.ndarray,
    labels: np.ndarray,
    num_points: int,
    features: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    """
    NumPy点群とラベルを同じindexでサンプリングする。
    """
    points = np.asarray(points)
    labels = np.asarray(labels)

    num_source_points = points.shape[0]

    if num_source_points == 0:
        raise ValueError("Point cloud has no points.")

    if labels.shape[0] != num_source_points:
        raise ValueError(
            "points and labels must contain the same number of entries."
        )

    if features is not None and features.shape[0] != num_source_points:
        raise ValueError(
            "points and features must contain the same number of entries."
        )

    replace = num_source_points < num_points

    indices = np.random.choice(
        num_source_points,
        size=num_points,
        replace=replace,
    )
    # indices : (num_points,)

    sampled_points = points[indices]
    # (N, 3) -> (num_points, 3)

    sampled_labels = labels[indices]
    # (N,) -> (num_points,)

    sampled_features = None

    if features is not None:
        sampled_features = features[indices]
        # (N, F) -> (num_points, F)

    return (
        sampled_points,
        sampled_labels,
        sampled_features,
    )


def load_local_segmentation_sample(
    file_path: str | Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    """
    自前の点群セグメンテーションデータを読み込む。

    対応形式
    ----------
    NPZ:
        points  : (N, 3)
        labels  : (N,)
        features: (N, F) 省略可

    TXT:
        x y z part_label
        または
        x y z feature... part_label

    PLY/PCD/LAS/LAZ:
        点群ファイル本体
        同じ場所に `{stem}_labels.npy` が必要
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    suffix = file_path.suffix.lower()

    if suffix == ".npz":
        return _load_npz_sample(file_path)

    if suffix in {".txt", ".csv"}:
        return _load_text_sample(file_path)

    if suffix in POINT_CLOUD_EXTENSIONS:
        return _load_point_cloud_with_sidecar_labels(
            file_path
        )

    raise ValueError(
        f"Unsupported segmentation data format: {suffix}"
    )


def _load_npz_sample(
    file_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    with np.load(file_path) as data:
        if "points" not in data or "labels" not in data:
            raise KeyError(
                f"{file_path} must contain 'points' and 'labels'."
            )

        points = np.asarray(
            data["points"],
            dtype=np.float32,
        )
        # points : (N, 3)

        labels = np.asarray(
            data["labels"],
            dtype=np.int64,
        ).reshape(-1)
        # labels : (N,)

        features = None

        if "features" in data:
            features = np.asarray(
                data["features"],
                dtype=np.float32,
            )
            # features : (N, F)

    return points, labels, features


def _load_text_sample(
    file_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    delimiter = "," if file_path.suffix.lower() == ".csv" else None

    values = np.loadtxt(
        file_path,
        delimiter=delimiter,
        dtype=np.float32,
    )
    # values : (N, 4+) または1行なら (4+,)

    if values.ndim == 1:
        values = values.reshape(1, -1)
        # (4+,) -> (1, 4+)

    if values.shape[1] < 4:
        raise ValueError(
            f"{file_path} requires at least x, y, z and label columns."
        )

    points = values[:, :3]
    # (N, 4+) -> (N, 3)

    labels = values[:, -1].astype(
        np.int64
    )
    # (N, 4+) -> (N,)

    features = None

    if values.shape[1] > 4:
        features = values[:, 3:-1]
        # (N, 4+) -> (N, F)

    return points, labels, features


def _load_point_cloud_with_sidecar_labels(
    file_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    point_cloud = load_point_cloud(
        file_path
    )

    points = np.asarray(
        point_cloud.points,
        dtype=np.float32,
    )
    # points : (N, 3)

    labels_path = (
        file_path.parent
        / f"{file_path.stem}_labels.npy"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            "Point-level label file not found: "
            f"{labels_path}"
        )

    labels = np.load(
        labels_path
    ).astype(
        np.int64
    ).reshape(-1)
    # labels : (N,)

    features = None

    if point_cloud.has_normals():
        features = np.asarray(
            point_cloud.normals,
            dtype=np.float32,
        )
        # features : (N, 3)

    return points, labels, features