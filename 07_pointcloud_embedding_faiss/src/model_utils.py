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
) -> tuple[torch.Tensor, torch.Tensor]:
    """PointNet++ のサンプリング & グルーピング処理。

    Args:
        num_centroids: サンプリングする中心点の数 (S)
        radius: 球状クエリの半径 (R)
        num_samples: 各グループ内でサンプリングする近傍点数 (K)
        xyz: 入力点群の3次元座標テンソル。Shape: (B, N, 3)
        points: 各点の特徴量テンソル。Shape: (B, N, D) または None

    Returns:
        new_xyz: サンプリングされた中心点の座標。Shape: (B, S, 3)
        new_points: グルーピングされた局所特徴量。Shape: (B, S, K, 3 + D)
    """
    # 1. 最遠点サンプリング (FPS) により中心点のインデックスを抽出
    # fps_idx: (B, S)
    fps_idx = farthest_point_sample(xyz, num_centroids)

    # 2. 中心点の座標を取得
    # new_xyz: (B, S, 3)
    new_xyz = index_points(xyz, fps_idx)

    # 3. 各中心点から半径 radius 内の近傍点を num_samples 個集約
    # idx: (B, S, K)
    idx = query_ball_point(radius, num_samples, xyz, new_xyz)

    # 4. 集約された近傍点の座標を取得
    # grouped_xyz: (B, S, K, 3)
    grouped_xyz = index_points(xyz, idx)

    # 5. 中心点を原点とする相対座標に正規化 (ココを修正)
    # new_xyz を (B, S, 3) -> (B, S, 1, 3) に拡張して引き算
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    # 6. 特徴量 (points) の結合
    if points is not None:
        # 近傍点の特徴量を抽出。grouped_points: (B, S, K, D)
        grouped_points = index_points(points, idx)
        # 相対座標 (3) と特徴量 (D) を最後の次元で結合 -> (B, S, K, 3 + D)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        # 特徴量がない場合は相対座標のみを返す -> (B, S, K, 3)
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(
    xyz: torch.Tensor,
    points: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """点群全体を1つのグループにまとめる（グローバル特徴抽出用）。

    Args:
        xyz: 入力点群の3次元座標テンソル。Shape: (B, N, 3)
        points: 各点の特徴量テンソル。Shape: (B, N, D) または None

    Returns:
        new_xyz: グループの中心点（原点）。Shape: (B, 1, 3)
        new_points: 全点を取り込んだ局所特徴量。Shape: (B, 1, N, 3 + D)
    """
    device = xyz.device
    batch_size, num_points, _ = xyz.shape

    # 1. 中心点を原点 (0, 0, 0) として 1 つだけ作成
    # new_xyz: (B, 1, 3)
    new_xyz = torch.zeros(batch_size, 1, 3, device=device)

    # 2. 全点の座標の次元を拡張 (B, N, 3) -> (B, 1, N, 3)
    # view よりも unsqueeze の方が意図が明確になります
    grouped_xyz = xyz.unsqueeze(1)

    # 3. 中心点からの相対座標を計算
    # (B, 1, N, 3) - (B, 1, 1, 3) -> (B, 1, N, 3)
    # ※ new_xyz が 0 なので値は変わりませんが、PointNet++ の数式（相対座標の入力）を正確に表現します
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    # 4. 特徴量 (points) の結合
    if points is not None:
        # points: (B, N, D) -> (B, 1, N, D) に拡張して結合
        grouped_points = points.unsqueeze(1)
        # 相対座標 (3) と特徴量 (D) を結合 -> (B, 1, N, 3 + D)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        # 特徴量がない場合は相対座標のみを返す -> (B, 1, N, 3)
        new_points = grouped_xyz_norm

    return new_xyz, new_points