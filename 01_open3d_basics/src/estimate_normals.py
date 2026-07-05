"""
estimate_normals.py

点群の法線ベクトルを推定する
"""

from pathlib import Path

import open3d as o3d

from config import MAX_NN, NORMAL_RADIUS
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
    print_header("Estimate Normals")

    data_dir = Path("../data")
    input_path = data_dir / "bun000.ply"

    output_dir = create_output_dir()
    output_path = output_dir / "bun000_with_normals.ply"

    print_subheader("Load Point Cloud")
    point_cloud = load_point_cloud(str(input_path))
    print_point_cloud_info(point_cloud)

    print_subheader("Estimate Normals")
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=MAX_NN,
        )
    )

    point_cloud.normalize_normals()

    print(f"Normal radius : {NORMAL_RADIUS}")
    print(f"Max NN        : {MAX_NN}")
    print_point_cloud_info(point_cloud)

    print_subheader("Save Point Cloud")
    save_point_cloud(point_cloud, str(output_path))
    print(f"Saved : {output_path}")

    visualize(
        point_cloud,
        window_name="Point Cloud with Normals",
    )


if __name__ == "__main__":
    main()