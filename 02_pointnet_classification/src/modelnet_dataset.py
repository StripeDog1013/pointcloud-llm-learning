"""
modelnet_dataset.py

PyTorch GeometricのModelNetをPointNet用Datasetとして扱う
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints

from config import NUM_POINTS
from dataset_utils import normalize_points_tensor

MODELNET10_CLASS_NAMES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]

MODELNET40_CLASS_NAMES = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]

class ModelNetDataset(Dataset):
    """
    PyTorch Geometric版ModelNetをラップするDataset。
    """

    def __init__(
        self,
        root_dir: str = "../data",
        train: bool = True,
        num_points: int = NUM_POINTS,
        name: str = "10",
    ):
        self.root_dir = root_dir
        self.train = train
        self.num_points = num_points
        self.name = name

        transform = SamplePoints(num_points)

        self.dataset = ModelNet(
            root=root_dir,
            name=name,
            train=train,
            transform=transform,
        )

        self.num_classes = int(name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        points = data.pos.float()
        points = normalize_points_tensor(points)

        label = data.y.squeeze().long()

        return points, label
    
    @property
    def class_names(self):
        if self.name == "10":
            return MODELNET10_CLASS_NAMES

        if self.name == "40":
            return MODELNET40_CLASS_NAMES

        raise ValueError(f"Unsupported ModelNet name: {self.name}")