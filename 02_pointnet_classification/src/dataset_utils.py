"""
dataset_utils.py

Dataset用の共通処理
"""

import numpy as np
import torch


def normalize_points_tensor(points: torch.Tensor) -> torch.Tensor:
    """
    点群を中心化し、単位球に正規化する。
    """
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


def normalize_points_numpy(points: np.ndarray) -> np.ndarray:
    """
    NumPy点群を中心化し、単位球に正規化する。
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid

    scale = np.max(np.linalg.norm(points, axis=1))

    if scale > 0:
        points = points / scale

    return points