"""
crop_point_cloud.py

点群を指定範囲で切り出す
"""

from pathlib import Path

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
    print_header("Crop Point Cloud")

    data_dir = Path("../data")
    input_path = data_dir / "bun000.ply"

    output_dir = create_output_dir()
    output_path = output_dir / "bun000_cropped.ply"

    print_subheader("Load Point Cloud")
    point_cloud = load_point_cloud(str(input_path))
    print_point_cloud_info(point_cloud)

    print_subheader("Create Bounding Box")

    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-0.05, 0.02, -0.05),
        max_bound=(0.08, 0.18, 0.05),
    )
    bbox.color = (1.0, 0.0, 0.0)

    cropped = point_cloud.crop(bbox)

    print("Crop range")
    print(f"min_bound : {bbox.min_bound}")
    print(f"max_bound : {bbox.max_bound}")
    print_point_cloud_info(cropped)

    print_subheader("Save Cropped Point Cloud")
    save_point_cloud(cropped, str(output_path))
    print(f"Saved : {output_path}")

    visualize(
        [point_cloud, bbox],
        window_name="Original Point Cloud with Crop Box",
    )

    visualize(
        [cropped],
        window_name="Cropped Point Cloud",
    )


if __name__ == "__main__":
    main()