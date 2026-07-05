"""
compute_bounding_box.py

Axis Aligned Bounding Box(AABB)と
Oriented Bounding Box(OBB)を計算する
"""

from pathlib import Path

from utils import (
    load_point_cloud,
    print_header,
    print_point_cloud_info,
    print_subheader,
    visualize,
)


def main():
    print_header("Compute Bounding Box")

    data_dir = Path("../data")
    input_path = data_dir / "bun000.ply"

    print_subheader("Load Point Cloud")

    point_cloud = load_point_cloud(str(input_path))
    print_point_cloud_info(point_cloud)

    print_subheader("Axis Aligned Bounding Box")

    aabb = point_cloud.get_axis_aligned_bounding_box()
    aabb.color = (1.0, 0.0, 0.0)

    print(f"Min Bound : {aabb.min_bound}")
    print(f"Max Bound : {aabb.max_bound}")
    print(f"Center    : {aabb.get_center()}")
    print(f"Extent    : {aabb.get_extent()}")
    print(f"Volume    : {aabb.volume():.6f}")

    print_subheader("Oriented Bounding Box")

    obb = point_cloud.get_oriented_bounding_box()
    obb.color = (0.0, 1.0, 0.0)

    print(f"Center : {obb.center}")
    print(f"Extent : {obb.extent}")
    print(f"Volume : {obb.volume():.6f}")

    print_subheader("Visualization")

    print("Gray  : Point Cloud")
    print("Red   : Axis Aligned Bounding Box")
    print("Green : Oriented Bounding Box")

    point_cloud.paint_uniform_color((0.5, 0.5, 0.5))

    visualize(
        [point_cloud, aabb, obb],
        window_name="Bounding Box Comparison",
    )


if __name__ == "__main__":
    main()