"""
run_all.py

01_open3d_basicsのサンプルを順番に実行する
"""

from utils import print_header

import create_point_cloud
import load_point_cloud
import visualize_point_cloud
import voxel_downsample
import estimate_normals
import crop_point_cloud
import transform_point_cloud
import compute_bounding_box
import nearest_neighbor
import icp_registration
from utils import timer

@timer
def main():
    print_header("Run All Open3D Basics")

    create_point_cloud.main()
    print()

    load_point_cloud.main()
    print()

    visualize_point_cloud.main()
    print()

    voxel_downsample.main()
    print()

    estimate_normals.main()
    print()

    crop_point_cloud.main()
    print()

    transform_point_cloud.main()
    print()

    compute_bounding_box.main()
    print()

    nearest_neighbor.main()
    print()

    icp_registration.main()
    print()

    print_header("All Open3D Basics Finished")


if __name__ == "__main__":
    main()