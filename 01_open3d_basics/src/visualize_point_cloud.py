"""
visualize_point_cloud.py

点群の可視化
"""

from pathlib import Path

from utils import (
    load_point_cloud,
    print_header,
    print_subheader,
    print_point_cloud_info,
    visualize,
)


POINT_CLOUD_FILES = [
    "bun000.ply",
    "bun045.ply",
    "bun090.ply",
    "bun180.ply",
    "bun270.ply",
    "bun315.ply",
]


def main():
    print_header("Visualize Point Clouds")

    data_dir = Path("../data")

    for file_name in POINT_CLOUD_FILES:
        print_subheader(file_name)

        file_path = data_dir / file_name

        point_cloud = load_point_cloud(str(file_path))

        print_point_cloud_info(point_cloud)

        print(
            "Close the Open3D window to display the next point cloud..."
        )

        visualize(
            point_cloud,
            window_name=file_name,
        )


if __name__ == "__main__":
    main()