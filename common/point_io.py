"""
point_io.py

点群ファイルの読み込み・保存
対応形式:
    .ply
    .pcd
    .las
    .laz
"""

from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

from common.path_utils import ensure_parent_directory


SUPPORTED_POINT_CLOUD_EXTENSIONS = [
    ".ply",
    ".pcd",
    ".las",
    ".laz",
]


def load_point_cloud(file_path: str | Path) -> o3d.geometry.PointCloud:
    """
    点群ファイルを読み込み、Open3D PointCloudとして返す。
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if not file_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")

    if suffix in [".ply", ".pcd"]:
        point_cloud = o3d.io.read_point_cloud(str(file_path))

    elif suffix in [".las", ".laz"]:
        las = laspy.read(file_path)

        xyz = np.vstack(
            [
                las.x,
                las.y,
                las.z,
            ]
        ).T

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz)

        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            rgb = np.vstack(
                [
                    las.red,
                    las.green,
                    las.blue,
                ]
            ).T.astype(np.float64)

            if rgb.max() > 0:
                rgb = rgb / 65535.0
                point_cloud.colors = o3d.utility.Vector3dVector(rgb)

    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")

    if point_cloud.is_empty():
        raise RuntimeError(f"Failed to load point cloud: {file_path}")

    return point_cloud


def save_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    file_path: str | Path,
) -> None:
    """
    Open3D PointCloudを指定形式で保存する。
    """
    file_path = ensure_parent_directory(file_path)
    suffix = file_path.suffix.lower()

    if suffix in [".ply", ".pcd"]:
        success = o3d.io.write_point_cloud(
            str(file_path),
            point_cloud,
        )

        if not success:
            raise RuntimeError(f"Failed to save point cloud: {file_path}")

    elif suffix in [".las", ".laz"]:
        xyz = np.asarray(point_cloud.points)

        if xyz.size == 0:
            raise ValueError("Point cloud has no points.")

        header = laspy.LasHeader(
            point_format=3,
            version="1.2",
        )

        las = laspy.LasData(header)

        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]

        if point_cloud.has_colors():
            colors = np.asarray(point_cloud.colors)
            colors = np.clip(colors, 0.0, 1.0)
            colors = (colors * 65535).astype(np.uint16)

            las.red = colors[:, 0]
            las.green = colors[:, 1]
            las.blue = colors[:, 2]

        las.write(file_path)

    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")
    
def print_point_cloud_info(point_cloud: o3d.geometry.PointCloud) -> None:
    """
    点群情報を表示する。
    """
    print(f"Number of points : {len(point_cloud.points)}")
    print(f"Has normals      : {point_cloud.has_normals()}")
    print(f"Has colors       : {point_cloud.has_colors()}")