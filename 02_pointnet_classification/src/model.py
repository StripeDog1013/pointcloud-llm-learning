"""
model.py

PointNet分類モデル（論文準拠版）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES


class TNet(nn.Module):
    """
    T-Net。

    入力特徴から k x k の変換行列を学習する。
    """

    def __init__(self, k: int = 3):
        super().__init__()

        self.k = k

        self.conv1 = nn.Conv1d(k, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            shape = (B, k, N)

        Returns
        -------
        torch.Tensor
            shape = (B, k, k)
        """
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, dim=2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(
            self.k,
            dtype=x.dtype,
            device=x.device,
        ).view(1, self.k * self.k)

        x = x + identity
        x = x.view(batch_size, self.k, self.k)

        return x


class PointNetEncoder(nn.Module):
    """
    PointNet Encoder。

    Input T-Net と Feature T-Net を含む。
    """

    def __init__(self, use_feature_transform: bool = True):
        super().__init__()

        self.use_feature_transform = use_feature_transform

        self.input_transform = TNet(k=3)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.feature_transform = TNet(k=64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            shape = (B, N, 3)

        Returns
        -------
        global_feature : torch.Tensor
            shape = (B, 1024)
        input_transform : torch.Tensor
            shape = (B, 3, 3)
        feature_transform : torch.Tensor | None
            shape = (B, 64, 64)
        """
        x = x.transpose(2, 1)

        input_transform = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, input_transform)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        feature_transform = None

        if self.use_feature_transform:
            feature_transform = self.feature_transform(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, feature_transform)
            x = x.transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        global_feature = torch.max(x, dim=2)[0]

        return global_feature, input_transform, feature_transform


class PointNetClassifier(nn.Module):
    """
    PointNet分類モデル。
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        use_feature_transform: bool = True,
    ):
        super().__init__()

        self.encoder = PointNetEncoder(
            use_feature_transform=use_feature_transform
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            shape = (B, N, 3)

        Returns
        -------
        logits : torch.Tensor
            shape = (B, num_classes)
        input_transform : torch.Tensor
            shape = (B, 3, 3)
        feature_transform : torch.Tensor | None
            shape = (B, 64, 64)
        """
        x, input_transform, feature_transform = self.encoder(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        logits = self.fc3(x)

        return logits, input_transform, feature_transform


def feature_transform_regularizer(
    transform: torch.Tensor,
) -> torch.Tensor:
    """
    Feature Transform Regularizer。

    T-Netの変換行列が直交行列に近づくようにする。

    || I - A A^T || を最小化する。
    """
    if transform is None:
        return torch.tensor(0.0)

    batch_size, k, _ = transform.shape

    identity = torch.eye(
        k,
        dtype=transform.dtype,
        device=transform.device,
    ).unsqueeze(0)

    diff = torch.bmm(transform, transform.transpose(2, 1)) - identity

    loss = torch.mean(torch.norm(diff, dim=(1, 2)))

    return loss


def count_parameters(model: nn.Module) -> int:
    """
    学習対象パラメータ数を返す。
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():
    model = PointNetClassifier(
        num_classes=NUM_CLASSES,
        use_feature_transform=True,
    )

    dummy_points = torch.randn(4, 1024, 3)

    logits, input_transform, feature_transform = model(dummy_points)

    print(model)
    print(f"Input shape             : {dummy_points.shape}")
    print(f"Output shape            : {logits.shape}")
    print(f"Input transform shape   : {input_transform.shape}")
    print(f"Feature transform shape : {feature_transform.shape}")
    print(f"Parameters              : {count_parameters(model):,}")


if __name__ == "__main__":
    main()