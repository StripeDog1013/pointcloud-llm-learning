"""
folder_dataset.py

自前データ用の点群セグメンテーションDataset
"""

from pathlib import Path

import torch
from torch.utils.data import (
    Dataset,
)

from dataset_utils import (
    load_local_segmentation_sample,
    normalize_points_numpy,
    sample_points_and_labels_numpy,
)

SUPPORTED_LOCAL_EXTENSIONS = {
    ".npz",
    ".txt",
    ".csv",
    ".ply",
    ".pcd",
    ".las",
    ".laz",
}

class LocalPartSegmentationDataset(Dataset):
    """
    自前の点群セグメンテーションDataset。

    フォルダ構成
    ----------
    data/custom/train/
    ├── Chair/
    │   ├── chair_001.npz
    │   └── chair_002.ply
    └── Table/
        └── table_001.txt

    data/custom/test/
    ├── Chair/
    └── Table/

    PLYなどを使う場合:
        chair_002.ply
        chair_002_labels.npy
    """

    def __init__(
        self,
        root_dir: str | Path,
        num_points: int,
        class_names: list[str] | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.num_points = num_points

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        detected_class_names = sorted(
            path.name
            for path in self.root_dir.iterdir()
            if path.is_dir()
        )

        if class_names is None:
            self.class_names = detected_class_names
        else:
            self.class_names = list(class_names)

            unknown_classes = set(
                detected_class_names
            ) - set(self.class_names)

            if unknown_classes:
                raise ValueError(
                    "Unknown classes found in dataset: "
                    f"{sorted(unknown_classes)}"
                )

        if not self.class_names:
            raise RuntimeError(
                f"No category directories found: {self.root_dir}"
            )

        self.class_to_idx = {
            class_name: index
            for index, class_name
            in enumerate(self.class_names)
        }

        self.samples = self._collect_samples()

        if not self.samples:
            raise RuntimeError(
                f"No segmentation samples found: {self.root_dir}"
            )

    def _collect_samples(
        self,
    ) -> list[tuple[Path, int]]:
        samples = []

        for class_name in self.class_names:
            class_dir = (
                self.root_dir
                / class_name
            )

            if not class_dir.exists():
                continue

            category_index = self.class_to_idx[
                class_name
            ]

            for file_path in sorted(
                class_dir.iterdir()
            ):
                if file_path.suffix.lower() not in (
                    SUPPORTED_LOCAL_EXTENSIONS
                ):
                    continue

                # 点群形式のラベル用sidecarは
                # Datasetサンプルとして数えない。
                if file_path.name.endswith(
                    "_labels.npy"
                ):
                    continue

                samples.append(
                    (
                        file_path,
                        category_index,
                    )
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        file_path, category_index = self.samples[
            index
        ]

        (
            points,
            part_labels,
            features,
        ) = load_local_segmentation_sample(
            file_path
        )

        # points      : (N, 3)
        # part_labels : (N,)
        # features    : (N, F) またはNone

        (
            points,
            part_labels,
            features,
        ) = sample_points_and_labels_numpy(
            points=points,
            labels=part_labels,
            num_points=self.num_points,
            features=features,
        )

        # points      : (NUM_POINTS, 3)
        # part_labels : (NUM_POINTS,)
        # features    : (NUM_POINTS, F) またはNone

        points = normalize_points_numpy(
            points
        )
        # points : (NUM_POINTS, 3)

        points_tensor = torch.from_numpy(
            points
        ).float()
        # points_tensor : (NUM_POINTS, 3)

        part_labels_tensor = torch.from_numpy(
            part_labels
        ).long()
        # part_labels_tensor : (NUM_POINTS,)

        category_tensor = torch.tensor(
            category_index,
            dtype=torch.long,
        )
        # category_tensor : scalar

        if features is None:
            features_tensor = torch.empty(
                self.num_points,
                0,
                dtype=torch.float32,
            )
            # features_tensor : (NUM_POINTS, 0)
        else:
            features_tensor = torch.from_numpy(
                features.astype(
                    "float32",
                    copy=False,
                )
            )
            # features_tensor : (NUM_POINTS, F)

        return (
            points_tensor,
            category_tensor,
            part_labels_tensor,
            features_tensor,
        )
