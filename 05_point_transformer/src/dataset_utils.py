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
    NumPy点群をFarthest Point Samplingでサンプリングする。
    """

    return farthest_point_sample_numpy(
        points,
        num_points,
    )

def farthest_point_sample_numpy(
    points: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """
    NumPy版 Farthest Point Sampling

    Parameters
    ----------
    points : (N, 3)

    num_points : int

    Returns
    -------
    (num_points, 3)
    """

    if len(points) == 0:
        raise ValueError("Point cloud has no points.")

    if num_points >= len(points):
        return points.copy()

    sampled_indices = np.zeros(
        num_points,
        dtype=np.int64,
    )

    distances = np.full(
        len(points),
        np.inf,
    )

    farthest = np.random.randint(
        len(points),
    )

    for i in range(num_points):
        sampled_indices[i] = farthest

        centroid = points[farthest]

        dist = np.sum(
            (points - centroid) ** 2,
            axis=1,
        )

        distances = np.minimum(
            distances,
            dist,
        )

        farthest = np.argmax(
            distances,
        )

    return points[sampled_indices]

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