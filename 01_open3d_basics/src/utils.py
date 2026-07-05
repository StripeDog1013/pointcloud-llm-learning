"""
utils.py

共通ユーティリティ
"""

from functools import wraps
from pathlib import Path
import platform
import sys
import time

import open3d as o3d


def print_header(title: str) -> None:
    """ヘッダーを表示する。"""
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_subheader(title: str) -> None:
    """サブヘッダーを表示する。"""
    print("-" * 60)
    print(title)
    print("-" * 60)


def print_python_info() -> None:
    """Python実行環境を表示する。"""
    print(f"Python Version    : {sys.version}")
    print(f"Python Executable : {sys.executable}")
    print(f"Platform          : {platform.platform()}")
    print(f"Machine           : {platform.machine()}")
    print(f"Processor         : {platform.processor()}")


def timer(func):
    """関数の実行時間を表示するデコレータ。"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        print(f"\nExecution Time : {elapsed:.3f} sec")

        return result

    return wrapper


def create_output_dir() -> Path:
    """
    outputsディレクトリを作成して返す。
    """
    output_dir = Path("../outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def load_point_cloud(file_path: str) -> o3d.geometry.PointCloud:
    """
    点群を読み込む。

    Parameters
    ----------
    file_path : str
        点群ファイル

    Returns
    -------
    o3d.geometry.PointCloud
    """
    point_cloud = o3d.io.read_point_cloud(file_path)

    if point_cloud.is_empty():
        raise RuntimeError(f"Failed to load point cloud: {file_path}")

    return point_cloud


def save_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    file_path: str,
) -> None:
    """
    点群を保存する。

    Parameters
    ----------
    point_cloud : PointCloud
    file_path : str
    """
    o3d.io.write_point_cloud(file_path, point_cloud)


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