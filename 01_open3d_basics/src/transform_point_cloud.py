"""
transform_point_cloud.py

点群の座標変換を行う
"""

from pathlib import Path
import copy
import math

import numpy as np
import open3d as o3d

from utils import (
    create_output_dir,
    load_point_cloud,
    print_header,
    print_point_cloud_info,
    print_subheader,
    save_point_cloud,
    visualize,
)


def main():
    print_header("Transform Point Cloud")

    data_dir = Path("../data")
    input_path = data_dir / "bun000.ply"

    output_dir = create_output_dir()
    output_path = output_dir / "bun000_transformed.ply"

    print_subheader("Load Point Cloud")
    point_cloud = load_point_cloud(str(input_path))
    print_point_cloud_info(point_cloud)

    transformed = copy.deepcopy(point_cloud)

    print_subheader("Apply Transformation")

    angle_deg = 45.0
    angle_rad = math.radians(angle_deg)
    translate_x = 0.15
    scale = 1.5

    rotation_matrix = transformed.get_rotation_matrix_from_xyz(
        (0.0, angle_rad, 0.0)
    )

    center = transformed.get_center()

    transformed.rotate(
        rotation_matrix,
        center=center,
    )

    transformed.translate((translate_x, 0.0, 0.0))

    transformed.scale(
        scale,
        center=transformed.get_center(),
    )

    print(f"Rotation      : {angle_deg} deg around Y axis")
    print(f"Translation   : ({translate_x}, 0.0, 0.0)")
    print(f"Scale         : {scale}")
    print_point_cloud_info(transformed)

    print_subheader("Save Transformed Point Cloud")
    save_point_cloud(transformed, str(output_path))
    print(f"Saved : {output_path}")

    original = copy.deepcopy(point_cloud)
    original.paint_uniform_color((0.5, 0.5, 0.5))
    transformed.paint_uniform_color((1.0, 0.0, 0.0))

    visualize(
        [original, transformed],
        window_name="Original and Transformed Point Cloud",
    )


if __name__ == "__main__":
    main()