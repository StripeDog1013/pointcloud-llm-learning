"""
utils.py

共通ユーティリティ
"""

import json
import random
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
import torch


# ==============================================================================
# Header
# ==============================================================================

def print_header(title: str):

    print("=" * 60)
    print(title)
    print("=" * 60)


def print_subheader(title: str):

    print("-" * 60)
    print(title)
    print("-" * 60)


# ==============================================================================
# Timestamp
# ==============================================================================

def get_timestamp():

    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ==============================================================================
# Timer
# ==============================================================================

def timer(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        print(f"\nExecution Time : {elapsed:.3f} sec")

        return result

    return wrapper


# ==============================================================================
# Random Seed
# ==============================================================================

def seed_everything(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# Directory
# ==============================================================================

def create_directory(path):

    path = Path(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


# ==============================================================================
# JSON
# ==============================================================================

def save_json(data, prefix, log_dir="../logs"):

    create_directory(log_dir)

    output_path = (
        Path(log_dir)
        / f"{prefix}_{get_timestamp()}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


def load_json(file_path):

    with open(file_path, "r", encoding="utf-8") as f:

        return json.load(f)


# ==============================================================================
# Point Cloud
# ==============================================================================

def load_point_cloud(file_path):

    file_path = Path(file_path)

    suffix = file_path.suffix.lower()

    if suffix in [".ply", ".pcd"]:

        point_cloud = o3d.io.read_point_cloud(str(file_path))

    elif suffix in [".las", ".laz"]:

        las = laspy.read(file_path)

        xyz = np.vstack(
            (
                las.x,
                las.y,
                las.z,
            )
        ).T

        point_cloud = o3d.geometry.PointCloud()

        point_cloud.points = (
            o3d.utility.Vector3dVector(xyz)
        )

    else:

        raise ValueError(
            f"Unsupported file format : {suffix}"
        )

    return point_cloud


def save_point_cloud(
    point_cloud,
    file_path,
):

    file_path = Path(file_path)

    suffix = file_path.suffix.lower()

    if suffix in [".ply", ".pcd"]:

        o3d.io.write_point_cloud(
            str(file_path),
            point_cloud,
        )

    elif suffix in [".las", ".laz"]:

        xyz = np.asarray(point_cloud.points)

        header = laspy.LasHeader(
            point_format=3,
            version="1.2",
        )

        las = laspy.LasData(header)

        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]

        las.write(file_path)

    else:

        raise ValueError(
            f"Unsupported file format : {suffix}"
        )

def print_point_cloud_info(
    point_cloud: o3d.geometry.PointCloud,
) -> None:
    """
    点群情報を表示する。
    """
    print(f"Number of points : {len(point_cloud.points)}")
    print(f"Has normals      : {point_cloud.has_normals()}")
    print(f"Has colors       : {point_cloud.has_colors()}")

def visualize(
    geometry,
    window_name: str = "Open3D",
) -> None:
    """
    Geometryを表示する。

    Parameters
    ----------
    geometry : Geometry または list
    """
    if isinstance(geometry, list):
        geometries = geometry
    else:
        geometries = [geometry]

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
    )

# ==============================================================================
# Checkpoint
# ==============================================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    file_path,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        file_path,
    )


def load_checkpoint(
    model,
    optimizer,
    file_path,
):

    checkpoint = torch.load(
        file_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


# ==============================================================================
# Accuracy
# ==============================================================================

def calculate_accuracy(
    prediction,
    target,
):

    prediction = prediction.argmax(dim=1)

    correct = (
        prediction == target
    ).sum().item()

    return correct / len(target)