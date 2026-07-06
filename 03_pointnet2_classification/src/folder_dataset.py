"""
folder_dataset.py

自前点群データ用Dataset
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import NUM_POINTS
from dataset_utils import (
    normalize_points_numpy,
    sample_points_numpy,
)
from utils import load_point_cloud


class PointCloudFolderDataset(Dataset):
    """
    自前点群データを分類用Datasetとして扱う。

    想定構成:

    data/train/chair/*.ply
    data/train/table/*.pcd
    data/test/chair/*.las
    data/test/table/*.laz
    """

    def __init__(
        self,
        root_dir: str,
        num_points: int = NUM_POINTS,
    ):
        self.root_dir = Path(root_dir)
        self.num_points = num_points

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        self.class_names = sorted(
            [
                path.name
                for path in self.root_dir.iterdir()
                if path.is_dir()
            ]
        )

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.class_names)
        }

        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No point cloud files found: {self.root_dir}"
            )

    def _collect_samples(self):
        samples = []

        extensions = [
            "*.ply",
            "*.pcd",
            "*.las",
            "*.laz",
        ]

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            label = self.class_to_idx[class_name]

            for extension in extensions:
                for file_path in sorted(class_dir.glob(extension)):
                    samples.append((file_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]

        point_cloud = load_point_cloud(file_path)
        points = np.asarray(point_cloud.points)

        points = sample_points_numpy(
            points,
            self.num_points,
        )

        points = normalize_points_numpy(points)

        points = torch.tensor(
            points,
            dtype=torch.float32,
        )

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return points, label