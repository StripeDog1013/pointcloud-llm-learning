"""
model_utils.py

PointNet++ Segmentation用の点群操作ユーティリティ。

主な処理:
- 点間距離計算
- Point indexing
- Farthest Point Sampling
- Ball Query
- Sampling / Grouping
- 3-NN補間
"""

import torch


def square_distance(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """
    2つの点集合間の二乗距離を計算する。

    Parameters
    ----------
    src : torch.Tensor
        shape = (B, N, C)

    dst : torch.Tensor
        shape = (B, M, C)

    Returns
    -------
    torch.Tensor
        shape = (B, N, M)
    """
    # src @ dst^T
    # (B, N, C) @ (B, C, M) -> (B, N, M)
    distance = -2.0 * torch.matmul(
        src,
        dst.transpose(1, 2),
    )

    # ||src||^2
    # (B, N, C) -> (B, N) -> (B, N, 1)
    distance += torch.sum(
        src**2,
        dim=-1,
        keepdim=True,
    )

    # ||dst||^2
    # (B, M, C) -> (B, M) -> (B, 1, M)
    distance += torch.sum(
        dst**2,
        dim=-1,
    ).unsqueeze(1)

    # distance : (B, N, M)
    return distance


def index_points(
    points: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    バッチ単位で指定indexの点や特徴を取り出す。

    Parameters
    ----------
    points : torch.Tensor
        shape = (B, N, C)

    indices : torch.Tensor
        shape = (B, S)
        または (B, S, K)

    Returns
    -------
    torch.Tensor
        shape = (B, S, C)
        または (B, S, K, C)
    """
    batch_size = points.shape[0]

    view_shape = list(indices.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(indices.shape)
    repeat_shape[0] = 1

    # batch_indices : indicesと同じshape
    batch_indices = (
        torch.arange(
            batch_size,
            device=points.device,
        )
        .reshape(view_shape)
        .repeat(repeat_shape)
    )

    sampled_points = points[
        batch_indices,
        indices,
        :,
    ]

    # indicesが(B, S)なら      sampled_points : (B, S, C)
    # indicesが(B, S, K)なら   sampled_points : (B, S, K, C)
    return sampled_points


def farthest_point_sample(
    xyz: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Farthest Point Samplingで代表点を選択する。

    Parameters
    ----------
    xyz : torch.Tensor
        shape = (B, N, 3)

    num_samples : int
        選択する代表点数

    Returns
    -------
    torch.Tensor
        shape = (B, num_samples)
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
    # centroids : (B, num_samples)

    minimum_distances = torch.full(
        (batch_size, num_points),
        float("inf"),
        device=device,
    )
    # minimum_distances : (B, N)

    farthest = torch.randint(
        low=0,
        high=num_points,
        size=(batch_size,),
        device=device,
    )
    # farthest : (B,)

    batch_indices = torch.arange(
        batch_size,
        device=device,
    )
    # batch_indices : (B,)

    for sample_index in range(num_samples):
        centroids[:, sample_index] = farthest

        centroid = xyz[
            batch_indices,
            farthest,
            :,
        ].unsqueeze(1)
        # (B, 3) -> (B, 1, 3)

        distances = torch.sum(
            (xyz - centroid) ** 2,
            dim=-1,
        )
        # (B, N, 3) - (B, 1, 3) -> (B, N, 3)
        # distances : (B, N)

        minimum_distances = torch.minimum(
            minimum_distances,
            distances,
        )
        # minimum_distances : (B, N)

        farthest = torch.max(
            minimum_distances,
            dim=-1,
        )[1]
        # farthest : (B,)

    return centroids


def query_ball_point(
    radius: float,
    num_samples: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
) -> torch.Tensor:
    """
    各代表点について、半径内の近傍点を取得する。

    Parameters
    ----------
    radius : float
        探索半径

    num_samples : int
        各代表点の最大近傍点数

    xyz : torch.Tensor
        元の点群。
        shape = (B, N, 3)

    new_xyz : torch.Tensor
        代表点。
        shape = (B, S, 3)

    Returns
    -------
    torch.Tensor
        近傍点index。
        shape = (B, S, num_samples)
    """
    batch_size, num_points, _ = xyz.shape
    num_centroids = new_xyz.shape[1]
    device = xyz.device

    if num_samples > num_points:
        num_samples = num_points

    group_indices = (
        torch.arange(
            num_points,
            device=device,
        )
        .reshape(1, 1, num_points)
        .repeat(batch_size, num_centroids, 1)
    )
    # group_indices : (B, S, N)

    squared_distances = square_distance(
        new_xyz,
        xyz,
    )
    # (B, S, 3), (B, N, 3) -> (B, S, N)

    group_indices[
        squared_distances > radius**2
    ] = num_points

    group_indices = group_indices.sort(
        dim=-1
    )[0][:, :, :num_samples]
    # (B, S, N) -> (B, S, num_samples)

    first_indices = group_indices[
        :,
        :,
        0,
    ].unsqueeze(-1)
    # (B, S) -> (B, S, 1)

    first_indices = first_indices.repeat(
        1,
        1,
        num_samples,
    )
    # first_indices : (B, S, num_samples)

    # 半径内に点が存在しない場合への安全策
    invalid_first = first_indices == num_points

    if invalid_first.any():
        nearest_indices = squared_distances.argmin(
            dim=-1,
            keepdim=True,
        )
        # nearest_indices : (B, S, 1)

        nearest_indices = nearest_indices.repeat(
            1,
            1,
            num_samples,
        )
        # nearest_indices : (B, S, num_samples)

        first_indices[invalid_first] = nearest_indices[
            invalid_first
        ]

    invalid_mask = group_indices == num_points
    group_indices[invalid_mask] = first_indices[
        invalid_mask
    ]

    return group_indices


def sample_and_group(
    num_centroids: int,
    radius: float,
    num_samples: int,
    xyz: torch.Tensor,
    points: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FPS・Ball Query・Groupingをまとめて実行する。

    Parameters
    ----------
    num_centroids : int
        FPSで選択する代表点数

    radius : float
        Ball Queryの探索半径

    num_samples : int
        各代表点の近傍点数

    xyz : torch.Tensor
        shape = (B, N, 3)

    points : torch.Tensor | None
        shape = (B, N, D)

    Returns
    -------
    new_xyz : torch.Tensor
        shape = (B, S, 3)

    new_points : torch.Tensor
        shape = (B, S, K, 3 + D)
    """
    fps_indices = farthest_point_sample(
        xyz=xyz,
        num_samples=num_centroids,
    )
    # fps_indices : (B, S)

    new_xyz = index_points(
        xyz,
        fps_indices,
    )
    # (B, N, 3), (B, S) -> (B, S, 3)

    neighbor_indices = query_ball_point(
        radius=radius,
        num_samples=num_samples,
        xyz=xyz,
        new_xyz=new_xyz,
    )
    # neighbor_indices : (B, S, K)

    grouped_xyz = index_points(
        xyz,
        neighbor_indices,
    )
    # (B, N, 3), (B, S, K) -> (B, S, K, 3)

    grouped_xyz_normalized = (
        grouped_xyz
        - new_xyz.unsqueeze(2)
    )
    # new_xyz.unsqueeze(2) : (B, S, 1, 3)
    # grouped_xyz_normalized : (B, S, K, 3)

    if points is not None:
        grouped_points = index_points(
            points,
            neighbor_indices,
        )
        # (B, N, D), (B, S, K) -> (B, S, K, D)

        new_points = torch.cat(
            [
                grouped_xyz_normalized,
                grouped_points,
            ],
            dim=-1,
        )
        # (B, S, K, 3) + (B, S, K, D)
        # -> (B, S, K, 3 + D)
    else:
        new_points = grouped_xyz_normalized
        # new_points : (B, S, K, 3)

    return new_xyz, new_points


def sample_and_group_all(
    xyz: torch.Tensor,
    points: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    すべての点を1つのグループにまとめる。

    Parameters
    ----------
    xyz : torch.Tensor
        shape = (B, N, 3)

    points : torch.Tensor | None
        shape = (B, N, D)

    Returns
    -------
    new_xyz : torch.Tensor
        shape = (B, 1, 3)

    new_points : torch.Tensor
        shape = (B, 1, N, 3 + D)
    """
    batch_size, num_points, _ = xyz.shape

    new_xyz = torch.zeros(
        batch_size,
        1,
        3,
        dtype=xyz.dtype,
        device=xyz.device,
    )
    # new_xyz : (B, 1, 3)

    grouped_xyz = xyz.unsqueeze(1)
    # (B, N, 3) -> (B, 1, N, 3)

    if points is not None:
        grouped_points = points.unsqueeze(1)
        # (B, N, D) -> (B, 1, N, D)

        new_points = torch.cat(
            [
                grouped_xyz,
                grouped_points,
            ],
            dim=-1,
        )
        # -> (B, 1, N, 3 + D)
    else:
        new_points = grouped_xyz
        # new_points : (B, 1, N, 3)

    return new_xyz, new_points


def three_nn(
    target_xyz: torch.Tensor,
    source_xyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    各target点について、source点群から近い3点を探す。

    Feature Propagationで使用する。

    Parameters
    ----------
    target_xyz : torch.Tensor
        高解像度側の点。
        shape = (B, N, 3)

    source_xyz : torch.Tensor
        低解像度側の点。
        shape = (B, S, 3)

    Returns
    -------
    distances : torch.Tensor
        shape = (B, N, K)

    indices : torch.Tensor
        shape = (B, N, K)

    Notes
    -----
    source点数が3未満の場合は、利用可能な点数へ自動調整する。
    """
    num_source_points = source_xyz.shape[1]
    interpolation_k = min(
        3,
        num_source_points,
    )

    squared_distances = square_distance(
        target_xyz,
        source_xyz,
    )
    # (B, N, 3), (B, S, 3) -> (B, N, S)

    squared_distances, indices = torch.topk(
        squared_distances,
        k=interpolation_k,
        dim=-1,
        largest=False,
        sorted=True,
    )
    # squared_distances : (B, N, K)
    # indices           : (B, N, K)

    distances = torch.sqrt(
        torch.clamp(
            squared_distances,
            min=1e-10,
        )
    )
    # distances : (B, N, K)

    return distances, indices


def three_interpolate(
    source_points: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    近傍特徴を重み付き補間する。

    Parameters
    ----------
    source_points : torch.Tensor
        低解像度側の特徴。
        shape = (B, S, C)

    indices : torch.Tensor
        shape = (B, N, K)

    weights : torch.Tensor
        shape = (B, N, K)

    Returns
    -------
    torch.Tensor
        補間後の特徴。
        shape = (B, N, C)
    """
    grouped_points = index_points(
        source_points,
        indices,
    )
    # (B, S, C), (B, N, K) -> (B, N, K, C)

    interpolated_points = torch.sum(
        grouped_points
        * weights.unsqueeze(-1),
        dim=2,
    )
    # weights.unsqueeze(-1) : (B, N, K, 1)
    # (B, N, K, C) * (B, N, K, 1)
    # -> sum over K -> (B, N, C)

    return interpolated_points


def interpolate_features(
    target_xyz: torch.Tensor,
    source_xyz: torch.Tensor,
    source_points: torch.Tensor,
) -> torch.Tensor:
    """
    3-NNと逆距離重みを使って特徴を補間する。

    Parameters
    ----------
    target_xyz : torch.Tensor
        高解像度側の座標。
        shape = (B, N, 3)

    source_xyz : torch.Tensor
        低解像度側の座標。
        shape = (B, S, 3)

    source_points : torch.Tensor
        低解像度側の特徴。
        shape = (B, S, C)

    Returns
    -------
    torch.Tensor
        高解像度側へ補間した特徴。
        shape = (B, N, C)
    """
    if source_xyz.shape[1] == 1:
        interpolated_points = source_points.repeat(
            1,
            target_xyz.shape[1],
            1,
        )
        # (B, 1, C) -> (B, N, C)

        return interpolated_points

    distances, indices = three_nn(
        target_xyz=target_xyz,
        source_xyz=source_xyz,
    )
    # distances : (B, N, K)
    # indices   : (B, N, K)

    inverse_distances = 1.0 / (
        distances + 1e-8
    )
    # inverse_distances : (B, N, K)

    weights = inverse_distances / torch.sum(
        inverse_distances,
        dim=-1,
        keepdim=True,
    )
    # weights : (B, N, K)

    interpolated_points = three_interpolate(
        source_points=source_points,
        indices=indices,
        weights=weights,
    )
    # interpolated_points : (B, N, C)

    return interpolated_points