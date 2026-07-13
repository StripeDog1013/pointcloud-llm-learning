"""
model_utils.py

Point Transformer用の点群操作ユーティリティ
"""

import torch


def square_distance(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """
    2つの点群間の二乗距離を計算する。

    src:
        (B, N, 3)

    dst:
        (B, M, 3)

    return:
        (B, N, M)
    """
    
    # (B, N, M)
    dist = -2 * torch.matmul(
        src,
        dst.transpose(1, 2),
    )

    # (B, N, 1)をブロードキャスト
    dist += torch.sum(
        src**2,
        dim=-1,
    ).unsqueeze(-1)

    # (B, 1, M)をブロードキャスト
    dist += torch.sum(
        dst**2,
        dim=-1,
    ).unsqueeze(1)

    return dist


def index_points(
    points: torch.Tensor,
    idx: torch.Tensor,
) -> torch.Tensor:
    """
    バッチ単位で指定インデックスの点を取り出す。

    points:
        (B, N, C)

    idx:
        (B, S)
        または
        (B, S, K)

    return:
        (B, S, C)
        または
        (B, S, K, C)
    """
    batch_size = points.shape[0]

    # [B] のシーケンスを作り、[B, 1] または [B, 1, 1] に変形
    # idx.ndim の数に応じて自動で次元を拡張
    view_shape = [batch_size] + [1] * (idx.ndim - 1)
    batch_indices = torch.arange(points.shape[0], device=points.device).view(view_shape)
    
    # batch_indices と idx が自動でブロードキャスト
    return points[batch_indices, idx, :]

def farthest_point_sample(
    xyz: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Farthest Point Sampling。

    xyz:
        (B, N, 3)

    return:
        選択した点のindex
        (B, num_samples)
    """
    batch_size, num_points, _ = xyz.shape
    device = xyz.device

    if num_samples > num_points:
        raise ValueError(
            f"num_samples ({num_samples}) must not exceed "
            f"num_points ({num_points})."
        )

    centroids = torch.zeros(
        batch_size,
        num_samples,
        dtype=torch.long,
        device=device,
    )

    minimum_distances = torch.full(
        (batch_size, num_points),
        float("inf"),
        device=device,
    )

    farthest = torch.randint(
        low=0,
        high=num_points,
        size=(batch_size,),
        device=device,
    )

    batch_indices = torch.arange(
        batch_size,
        device=device,
    )

    for sample_index in range(num_samples):
        centroids[:, sample_index] = farthest

        centroid = xyz[
            batch_indices,
            farthest,
            :,
        ].unsqueeze(1)

        distances = torch.sum(
            (xyz - centroid) ** 2,
            dim=-1,
        )

        minimum_distances = torch.minimum(
            minimum_distances,
            distances,
        )

        farthest = torch.max(
            minimum_distances,
            dim=-1,
        )[1]

    return centroids


def knn(
    xyz: torch.Tensor,
    k: int,
    exclude_self: bool = True,
) -> torch.Tensor:
    """
    同一点群内でk近傍探索を行う。

    xyz:
        (B, N, 3)

    return:
        (B, N, K)
    """
    num_points = xyz.shape[1]

    required_k = min(
        k + 1,
        num_points,
    )

    if required_k > num_points:
        raise ValueError(
            f"k is too large: required={required_k}, "
            f"num_points={num_points}"
        )

    distances = square_distance(
        xyz,
        xyz,
    )

    indices = distances.topk(
        k=required_k,
        largest=False,
    )[1]

    if exclude_self:
        indices = indices[:, :, 1:]

    return indices


def knn_query(
    query_xyz: torch.Tensor,
    support_xyz: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    query点ごとにsupport点群からk近傍を探す。

    query_xyz:
        (B, S, 3)

    support_xyz:
        (B, N, 3)

    return:
        (B, S, K)
    """
    num_support_points = support_xyz.shape[1]

    if k > num_support_points:
        raise ValueError(
            f"k ({k}) must not exceed support points "
            f"({num_support_points})."
        )

    distances = square_distance(
        query_xyz,
        support_xyz,
    )

    indices = distances.topk(
        k=k,
        dim=-1,
        largest=False,
    )[1]

    return indices


def group_points(
    xyz: torch.Tensor,
    features: torch.Tensor,
    k: int,
):
    """
    Point Transformer Layer用に近傍を取得する。

    xyz:
        (B, N, 3)

    features:
        (B, N, C)

    return:
        grouped_xyz:
            (B, N, K, 3)

        grouped_features:
            (B, N, K, C)
    """
    indices = knn(
        xyz=xyz,
        k=k,
        exclude_self=True,
    )

    grouped_xyz = index_points(
        xyz,
        indices,
    )

    grouped_features = index_points(
        features,
        indices,
    )

    return grouped_xyz, grouped_features