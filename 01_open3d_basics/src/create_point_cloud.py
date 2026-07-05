"""
create_point_cloud.py

NumPy配列から点群を作成する
"""

from pathlib import Path

import numpy as np
import open3d as o3d

from utils import (
    create_output_dir,
    print_header,
    print_point_cloud_info,
    save_point_cloud,
    visualize,
)


def main():
    print_header("Create Point Cloud")

    output_dir = create_output_dir()
    output_path = output_dir / "created_point_cloud.ply"

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    print_point_cloud_info(point_cloud)

    save_point_cloud(point_cloud, str(output_path))
    print(f"Saved : {output_path}")

    visualize(
        point_cloud,
        window_name="Created Point Cloud",
    )


if __name__ == "__main__":
    main()