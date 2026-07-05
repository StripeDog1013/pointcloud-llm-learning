"""
nearest_neighbor.py

KDTreeを使って近傍点探索を行う
"""

from pathlib import Path

import open3d as o3d

from utils import (
    load_point_cloud,
    print_header,
    print_point_cloud_info,
    print_subheader,
    visualize,
)


def main():
    print_header("Nearest Neighbor Search")

    data_dir = Path("../data")
    input_path = data_dir / "bun000.ply"

    print_subheader("Load Point Cloud")
    point_cloud = load_point_cloud(str(input_path))
    print_point_cloud_info(point_cloud)

    print_subheader("Build KDTree")
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    query_index = 100
    query_point = point_cloud.points[query_index]

    print(f"Query index : {query_index}")
    print(f"Query point : {query_point}")

    print_subheader("Search K Nearest Neighbors")
    k = 30
    _, indices, distances = kdtree.search_knn_vector_3d(query_point, k,)

    print(f"K : {k}")
    print(f"Neighbor indices : {indices[:10]} ...")
    print(f"Neighbor distances : {distances[:10]} ...")

    point_cloud.paint_uniform_color((0.5, 0.5, 0.5))

    colors = point_cloud.colors
    for index in indices:
        colors[index] = [1.0, 0.0, 0.0]

    colors[query_index] = [0.0, 1.0, 0.0]
    point_cloud.colors = colors

    print_subheader("Visualization")
    print("Gray  : Other points")
    print("Green : Query point")
    print("Red   : Nearest neighbors")

    visualize(
        point_cloud,
        window_name="Nearest Neighbor Search",
    )


if __name__ == "__main__":
    main()