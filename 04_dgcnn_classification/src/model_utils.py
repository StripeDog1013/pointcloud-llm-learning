"""
model_utils.py

DGCNN用ユーティリティ
"""

import torch


def knn(
    x: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    k近傍探索を行う。

    Parameters
    ----------
    x : torch.Tensor
        shape = (B, C, N)

    k : int
        近傍点数

    Returns
    -------
    torch.Tensor
        shape = (B, N, k)
    """

    inner = -2 * torch.matmul(x.transpose(2, 1), x)

    xx = torch.sum(x ** 2, dim=1, keepdim=True)

    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]

    return idx


def get_graph_feature(
    x: torch.Tensor,
    k: int,
    idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Edge Featureを生成する。

    Parameters
    ----------
    x : torch.Tensor
        shape = (B, C, N)

    k : int
        k近傍数

    idx : torch.Tensor | None
        shape = (B, N, k)

    Returns
    -------
    torch.Tensor
        shape = (B, 2*C, N, k)
    """

    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)

    # Shape: (B, N, k)
    if idx is None:
        idx = knn(x, k)

    device = x.device

    # Shape: (B * N * k)
    idx_base = torch.arange(0, batch_size, device=device,).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    # Shape: (B, N, k, C)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1,)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    # Shape: (B, N, k, C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # Shape: (B, N, k, 2*C)
    feature = torch.cat((feature - x, x), dim=3)
    
    # Shape: (B, 2*C, N, k)
    feature = feature.permute(0, 3, 1, 2)

    return feature