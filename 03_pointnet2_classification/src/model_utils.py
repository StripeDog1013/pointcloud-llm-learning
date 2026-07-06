"""
model_utils.py

PointNet++用の点群操作ユーティリティ
"""

import torch


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
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


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    indexに基づいて点を取り出す。

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


def farthest_point_sample(
    xyz: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Farthest Point Sampling。

    xyz: (B, N, 3)
    return: (B, num_samples)
    """
    device = xyz.device
    batch_size, num_points, _ = xyz.shape

    centroids = torch.zeros(
        batch_size,
        num_samples,
        dtype=torch.long,
        device=device,
    )

    distance = torch.ones(
        batch_size,
        num_points,
        device=device,
    ) * 1e10

    farthest = torch.randint(
        0,
        num_points,
        (batch_size,),
        dtype=torch.long,
        device=device,
    )

    batch_indices = torch.arange(
        batch_size,
        dtype=torch.long,
        device=device,
    )

    for i in range(num_samples):
        centroids[:, i] = farthest

        centroid = xyz[batch_indices, farthest, :].view(
            batch_size,
            1,
            3,
        )

        dist = torch.sum((xyz - centroid) ** 2, dim=-1)

        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(
    radius: float,
    num_samples: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
) -> torch.Tensor:
    """
    Ball Query。

    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    return: (B, S, num_samples)
    """
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    num_centroids = new_xyz.shape[1]

    group_idx = (
        torch.arange(num_points, device=device)
        .view(1, 1, num_points)
        .repeat(batch_size, num_centroids, 1)
    )

    sqrdists = square_distance(new_xyz, xyz)

    group_idx[sqrdists > radius ** 2] = num_points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :num_samples]

    first_group_idx = group_idx[:, :, 0].view(
        batch_size,
        num_centroids,
        1,
    ).repeat(1, 1, num_samples)

    mask = group_idx == num_points
    group_idx[mask] = first_group_idx[mask]

    return group_idx


def sample_and_group(
    num_centroids: int,
    radius: float,
    num_samples: int,
    xyz: torch.Tensor,
    points: torch.Tensor | None,
):
    """
    FPS + Ball Query + Grouping。

    xyz: (B, N, 3)
    points: (B, N, D) or None

    return:
        new_xyz: (B, S, 3)
        new_points: (B, S, K, 3 + D)
    """
    fps_idx = farthest_point_sample(
        xyz,
        num_centroids,
    )

    new_xyz = index_points(
        xyz,
        fps_idx,
    )

    idx = query_ball_point(
        radius,
        num_samples,
        xyz,
        new_xyz,
    )

    grouped_xyz = index_points(
        xyz,
        idx,
    )

    grouped_xyz_norm = grouped_xyz - new_xyz.view(
        xyz.shape[0],
        num_centroids,
        1,
        3,
    )

    if points is not None:
        grouped_points = index_points(
            points,
            idx,
        )

        new_points = torch.cat(
            [grouped_xyz_norm, grouped_points],
            dim=-1,
        )
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(
    xyz: torch.Tensor,
    points: torch.Tensor | None,
):
    """
    全点を1グループにまとめる。

    xyz: (B, N, 3)
    points: (B, N, D) or None
    """
    batch_size, num_points, _ = xyz.shape

    new_xyz = torch.zeros(
        batch_size,
        1,
        3,
        device=xyz.device,
    )

    grouped_xyz = xyz.view(
        batch_size,
        1,
        num_points,
        3,
    )

    if points is not None:
        new_points = torch.cat(
            [
                grouped_xyz,
                points.view(batch_size, 1, num_points, -1),
            ],
            dim=-1,
        )
    else:
        new_points = grouped_xyz

    return new_xyz, new_points