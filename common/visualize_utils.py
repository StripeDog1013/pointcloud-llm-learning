"""
visualize_utils.py

Open3D可視化関連の共通ユーティリティ
"""

import open3d as o3d


def visualize(
    geometry,
    window_name: str = "Open3D",
) -> None:
    """
    Open3D Geometryを可視化する。
    """
    if isinstance(geometry, list):
        geometries = geometry
    else:
        geometries = [geometry]

    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
    )