"""
icp_registration.py

ICPを使って2つの点群を位置合わせする
"""

from pathlib import Path
import copy

import numpy as np
import open3d as o3d

from config import VOXEL_SIZE
from utils import (
    load_point_cloud,
    print_header,
    print_point_cloud_info,
    print_subheader,
    visualize,
)


def draw_registration_result(
    source,
    target,
    transformation,
    window_name,
):
    """
    位置合わせ結果を可視化する。
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color((1.0, 0.0, 0.0))
    target_temp.paint_uniform_color((0.0, 1.0, 0.0))

    source_temp.transform(transformation)

    visualize(
        [source_temp, target_temp],
        window_name=window_name,
    )


def main():
    print_header("ICP Registration")

    data_dir = Path("../data")
    source_path = data_dir / "bun045.ply"
    target_path = data_dir / "bun000.ply"

    print_subheader("Load Point Clouds")
    source = load_point_cloud(str(source_path))
    target = load_point_cloud(str(target_path))

    print("Source")
    print_point_cloud_info(source)

    print("Target")
    print_point_cloud_info(target)

    print_subheader("Downsample")
    source_down = source.voxel_down_sample(VOXEL_SIZE)
    target_down = target.voxel_down_sample(VOXEL_SIZE)

    print(f"Voxel size : {VOXEL_SIZE}")
    print(f"Source down points : {len(source_down.points)}")
    print(f"Target down points : {len(target_down.points)}")

    threshold = VOXEL_SIZE * 3.0
    initial_transformation = np.identity(4)

    print_subheader("Before ICP")
    draw_registration_result(
        source_down,
        target_down,
        initial_transformation,
        window_name="Before ICP",
    )

    print_subheader("Run ICP")
    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold,
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    print(f"Threshold : {threshold}")
    print(f"Fitness   : {result.fitness:.6f}")
    print(f"RMSE      : {result.inlier_rmse:.6f}")
    print("Transformation:")
    print(result.transformation)

    print_subheader("After ICP")
    draw_registration_result(
        source_down,
        target_down,
        result.transformation,
        window_name="After ICP",
    )


if __name__ == "__main__":
    main()