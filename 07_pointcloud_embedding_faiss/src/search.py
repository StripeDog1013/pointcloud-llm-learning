"""
search.py

FAISSを用いて類似点群検索を行う。
"""

from pathlib import Path

import faiss
import numpy as np

from config import (
    EMBEDDING_FILE,
    FAISS_INDEX_FILE,
    INDEX_FILE,
    LABEL_FILE,
    TOP_K,
)


def load_embedding_data():
    """
    Embedding関連データを読み込む。
    """

    embeddings = np.load(
        EMBEDDING_FILE,
        allow_pickle=False,
    ).astype(np.float32)

    labels = np.load(
        LABEL_FILE,
        allow_pickle=False,
    ).astype(np.int64)

    indices = np.load(
        INDEX_FILE,
        allow_pickle=False,
    ).astype(np.int64)

    return (
        embeddings,
        labels,
        indices,
    )


def load_faiss_index():
    """
    FAISS Indexを読み込む。
    """

    path = Path(
        FAISS_INDEX_FILE
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return faiss.read_index(
        str(path)
    )


def search(
    index,
    query_embedding,
    top_k,
):
    """
    類似検索を実行する。
    """

    distances, neighbors = index.search(
        query_embedding,
        top_k,
    )

    return (
        distances[0],
        neighbors[0],
    )


def print_result(
    distances,
    neighbors,
    labels,
    indices,
):
    """
    検索結果表示
    """

    print()

    print("=" * 70)
    print("Search Result")
    print("=" * 70)

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
        print(
            f"{rank:2d}"
            f"  index={indices[neighbor]:5d}"
            f"  label={labels[neighbor]:2d}"
            f"  score={distance:.6f}"
        )


def main():

    embeddings, labels, indices = (
        load_embedding_data()
    )

    index = load_faiss_index()

    print(
        f"Embedding shape : {embeddings.shape}"
    )
    print(
        f"Index size      : {index.ntotal}"
    )

    while True:

        print()

        sample = input(
            f"Query Index (0-{len(embeddings)-1}, q=quit): "
        )

        if sample.lower() == "q":
            break

        sample = int(sample)

        if sample < 0 or sample >= len(
            embeddings
        ):
            print("Out of range")
            continue

        query = embeddings[
            sample : sample + 1
        ]
        # (1, embedding_dim)

        distances, neighbors = search(
            index=index,
            query_embedding=query,
            top_k=TOP_K,
        )

        print()

        print(
            f"Query Label : {labels[sample]}"
        )

        print_result(
            distances,
            neighbors,
            labels,
            indices,
        )


if __name__ == "__main__":
    main()