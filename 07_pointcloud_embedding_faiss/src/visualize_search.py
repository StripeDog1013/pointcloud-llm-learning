"""
visualize_search.py

FAISSによる類似検索結果をOpen3Dで可視化する。
"""

import numpy as np
import torch

from config import TOP_K
from dataset import (
    get_all_dataset,
    get_class_names,
)
from search import (
    load_embedding_data,
    load_faiss_index,
    search,
)


POINT_CLOUD_SPACING = 3.0

QUERY_COLOR = [1.0, 0.3, 0.3]
RESULT_COLOR = [0.3, 0.7, 1.0]


def tensor_to_numpy(
    points: torch.Tensor,
) -> np.ndarray:
    """
    点群TensorをNumPy配列へ変換する。

    Parameters
    ----------
    points:
        shape = (num_points, 3)

    Returns
    -------
    np.ndarray
        shape = (num_points, 3)
    """
    points = (
        points
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    # (num_points, 3)

    return points


def create_open3d_point_cloud(
    points: torch.Tensor,
    color: list[float],
    offset_x: float = 0.0,
):
    """
    Open3D用の点群を作成する。

    Open3Dは、この関数が呼ばれたときだけ読み込む。
    """
    import open3d as o3d

    points_np = tensor_to_numpy(
        points
    )
    # (num_points, 3)

    points_np[:, 0] += offset_x

    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = (
        o3d.utility.Vector3dVector(
            points_np
        )
    )

    point_cloud.paint_uniform_color(
        color
    )

    return point_cloud


def remove_query_from_results(
    distances: np.ndarray,
    neighbors: np.ndarray,
    query_position: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    検索結果からクエリ自身を除外する。
    """
    filtered_distances = []
    filtered_neighbors = []

    for distance, neighbor in zip(
        distances,
        neighbors,
    ):
        if neighbor < 0:
            continue

        if int(neighbor) == query_position:
            continue

        filtered_distances.append(
            distance
        )
        filtered_neighbors.append(
            neighbor
        )

        if len(filtered_neighbors) >= top_k:
            break

    return (
        np.asarray(
            filtered_distances,
            dtype=np.float32,
        ),
        np.asarray(
            filtered_neighbors,
            dtype=np.int64,
        ),
    )


def print_search_results(
    query_position: int,
    distances: np.ndarray,
    neighbors: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    class_names: list[str],
) -> None:
    """
    検索結果を表示する。
    """
    query_label = int(
        labels[query_position]
    )

    query_dataset_index = int(
        indices[query_position]
    )

    print()
    print("=" * 70)
    print("類似点群検索結果")
    print("=" * 70)

    print(
        f"Query embedding index : "
        f"{query_position}"
    )
    print(
        f"Query dataset index   : "
        f"{query_dataset_index}"
    )
    print(
        f"Query label           : "
        f"{query_label}"
    )
    print(
        f"Query class           : "
        f"{class_names[query_label]}"
    )

    print("-" * 70)

    for rank, (
        distance,
        neighbor,
    ) in enumerate(
        zip(
            distances,
            neighbors,
        ),
        start=1,
    ):
        neighbor = int(neighbor)
        label = int(labels[neighbor])
        dataset_index = int(
            indices[neighbor]
        )

        print(
            f"{rank:2d}. "
            f"embedding_index={neighbor:5d}  "
            f"dataset_index={dataset_index:5d}  "
            f"label={label:2d}  "
            f"class={class_names[label]:15s}  "
            f"score={distance:.6f}"
        )


def visualize_results(
    dataset,
    query_position: int,
    distances: np.ndarray,
    neighbors: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    class_names: list[str],
) -> None:
    """
    クエリ点群と検索結果を横並びで表示する。
    """
    try:
        import open3d as o3d
    except ImportError as error:
        raise ImportError(
            "Open3Dを読み込めませんでした。"
            "\nOpen3Dのインストール環境を確認してください。"
        ) from error

    geometries = []

    query_dataset_index = int(
        indices[query_position]
    )

    query_points, query_label = dataset[
        query_dataset_index
    ]
    # query_points: (num_points, 3)

    query_geometry = (
        create_open3d_point_cloud(
            points=query_points,
            color=QUERY_COLOR,
            offset_x=0.0,
        )
    )

    geometries.append(
        query_geometry
    )

    print()
    print(
        "[赤] Query: "
        f"{class_names[int(query_label)]}"
    )

    for rank, (
        distance,
        neighbor,
    ) in enumerate(
        zip(
            distances,
            neighbors,
        ),
        start=1,
    ):
        neighbor = int(neighbor)

        dataset_index = int(
            indices[neighbor]
        )

        points, label = dataset[
            dataset_index
        ]
        # points: (num_points, 3)

        offset_x = (
            rank * POINT_CLOUD_SPACING
        )

        geometry = (
            create_open3d_point_cloud(
                points=points,
                color=RESULT_COLOR,
                offset_x=offset_x,
            )
        )

        geometries.append(
            geometry
        )

        print(
            f"[青] Rank {rank}: "
            f"{class_names[int(label)]} "
            f"(score={distance:.6f})"
        )

    o3d.visualization.draw_geometries(
        geometries,
        window_name=(
            "Point Cloud Similarity Search"
        ),
        width=1400,
        height=800,
    )


def validate_data(
    embeddings: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    index,
    dataset,
) -> None:
    """
    各データの件数が一致しているか確認する。
    """
    num_embeddings = len(embeddings)

    if len(labels) != num_embeddings:
        raise ValueError(
            "embeddingsとlabelsの件数が一致しません。"
        )

    if len(indices) != num_embeddings:
        raise ValueError(
            "embeddingsとindicesの件数が一致しません。"
        )

    if index.ntotal != num_embeddings:
        raise ValueError(
            "FAISS IndexとEmbeddingの件数が一致しません。"
            f"\nFAISS : {index.ntotal}"
            f"\nEmbedding: {num_embeddings}"
        )

    if len(indices) > 0:
        max_index = int(
            indices.max()
        )

        if max_index >= len(dataset):
            raise IndexError(
                "indices.npyにデータセット範囲外の"
                "インデックスが含まれています。"
                f"\n最大index: {max_index}"
                f"\nDataset size: {len(dataset)}"
            )


def main() -> None:
    embeddings, labels, indices = (
        load_embedding_data()
    )
    # embeddings: (num_samples, embedding_dim)
    # labels:     (num_samples,)
    # indices:    (num_samples,)

    faiss_index = load_faiss_index()

    dataset = get_all_dataset()
    class_names = get_class_names()

    validate_data(
        embeddings=embeddings,
        labels=labels,
        indices=indices,
        index=faiss_index,
        dataset=dataset,
    )

    print("=" * 70)
    print("類似点群検索・可視化")
    print("=" * 70)
    print(
        f"Embedding shape : {embeddings.shape}"
    )
    print(
        f"Dataset size    : {len(dataset)}"
    )
    print(
        f"TOP_K           : {TOP_K}"
    )

    while True:
        print()

        user_input = input(
            "Query Embedding Index "
            f"(0-{len(embeddings) - 1}, q=終了): "
        ).strip()

        if user_input.lower() == "q":
            break

        try:
            query_position = int(
                user_input
            )
        except ValueError:
            print(
                "整数またはqを入力してください。"
            )
            continue

        if not (
            0
            <= query_position
            < len(embeddings)
        ):
            print(
                "インデックスが範囲外です。"
            )
            continue

        query_embedding = embeddings[
            query_position : query_position + 1
        ]
        # (1, embedding_dim)

        search_count = min(
            TOP_K + 1,
            len(embeddings),
        )

        distances, neighbors = search(
            index=faiss_index,
            query_embedding=query_embedding,
            top_k=search_count,
        )
        # distances: (search_count,)
        # neighbors: (search_count,)

        distances, neighbors = (
            remove_query_from_results(
                distances=distances,
                neighbors=neighbors,
                query_position=query_position,
                top_k=TOP_K,
            )
        )

        print_search_results(
            query_position=query_position,
            distances=distances,
            neighbors=neighbors,
            labels=labels,
            indices=indices,
            class_names=class_names,
        )

        visualize_results(
            dataset=dataset,
            query_position=query_position,
            distances=distances,
            neighbors=neighbors,
            labels=labels,
            indices=indices,
            class_names=class_names,
        )


if __name__ == "__main__":
    main()