"""
voxel_downsample.py

Voxel Down Samplingで点群を間引く
"""

from pathlib import Path

from config import VOXEL_SIZE
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
    print_header("Voxel Down Sampling")

    data_dir = Path("../data")
    input_path = data_dir / "bun000.ply"

    output_dir = create_output_dir()
    output_path = output_dir / "bun000_downsampled.ply"

    print_subheader("Load Point Cloud")
    point_cloud = load_point_cloud(str(input_path))
    print_point_cloud_info(point_cloud)

    print_subheader("Down Sample")
    downsampled = point_cloud.voxel_down_sample(
        voxel_size=VOXEL_SIZE,
    )
    print(f"Voxel size : {VOXEL_SIZE}")
    print_point_cloud_info(downsampled)

    print_subheader("Save Point Cloud")
    save_point_cloud(downsampled, str(output_path))
    print(f"Saved : {output_path}")

    visualize(
        [point_cloud],
        window_name="Original Point Cloud",
    )

    visualize(
        [downsampled],
        window_name="Downsampled Point Cloud",
    )


if __name__ == "__main__":
    main()