from pathlib import Path

import numpy as np
import open3d as o3d

from utils import print_header, print_subheader


OUTPUT_DIR = Path("../outputs")
OUTPUT_PATH = OUTPUT_DIR / "sample_point_cloud.ply"


def main():
    print_header("Check Open3D")

    print(f"Open3D Version : {o3d.__version__}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_subheader("Create Point Cloud")

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    print(point_cloud)
    print(f"Number of points : {len(point_cloud.points)}")

    print_subheader("Save Point Cloud")
    o3d.io.write_point_cloud(str(OUTPUT_PATH), point_cloud)
    print(f"Saved : {OUTPUT_PATH}")

    print_subheader("Load Point Cloud")
    loaded = o3d.io.read_point_cloud(str(OUTPUT_PATH))
    print(loaded)
    print(f"Loaded points : {len(loaded.points)}")

    print("Open3D point cloud test succeeded.")


if __name__ == "__main__":
    main()