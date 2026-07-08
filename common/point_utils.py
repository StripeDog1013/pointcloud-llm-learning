"""
point_utils.py

点群処理関連の共通ユーティリティ
"""

import numpy as np
import torch


def normalize_points_numpy(points: np.ndarray) -> np.ndarray:
    """
    NumPy点群を中心化し、単位球に正規化する。
    """
    if len(points) == 0:
        raise ValueError("Point cloud has no points.")

    centroid = np.mean(points, axis=0)
    points = points - centroid

    scale = np.max(np.linalg.norm(points, axis=1))

    if scale > 0:
        points = points / scale

    return points


def normalize_points_tensor(points: torch.Tensor) -> torch.Tensor:
    """
    Tensor点群を中心化し、単位球に正規化する。
    """
    if points.size(0) == 0:
        raise ValueError("Point cloud has no points.")

    centroid = points.mean(dim=0)
    points = points - centroid

    scale = points.norm(dim=1).max()

    if scale > 0:
        points = points / scale

    return points


def sample_points_numpy(
    points: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    NumPy点群の点数をnum_pointsに揃える。
    """
    if len(points) == 0:
        raise ValueError("Point cloud has no points.")

    replace = len(points) < num_points

    indices = np.random.choice(
        len(points),
        num_points,
        replace=replace,
    )

    return points[indices]


def square_distance(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """
    2つの点群間の二乗距離を計算する。

    src: (B, N, C)
    dst: (B, M, C)
    return: (B, N, M)
    """
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)

    return dist


def index_points(
    points: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    バッチ対応で点群から指定indexの点を取り出す。

    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    """
    batch_size = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = (
        torch.arange(batch_size, device=points.device)
        .view(view_shape)
        .repeat(repeat_shape)
    )

    return points[batch_indices, idx, :]