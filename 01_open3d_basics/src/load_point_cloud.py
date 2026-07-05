"""
load_point_cloud.py

点群ファイルの読み込み
"""

from pathlib import Path

from utils import (
    load_point_cloud,
    print_header,
    print_point_cloud_info,
    visualize,
)


def main():
    print_header("Load Point Cloud")

    data_dir = Path("../data")
    file_path = data_dir / "bun000.ply"

    print(f"Loading : {file_path}")

    point_cloud = load_point_cloud(str(file_path))

    print_point_cloud_info(point_cloud)

    visualize(
        point_cloud,
        window_name="Loaded Point Cloud",
    )


if __name__ == "__main__":
    main()