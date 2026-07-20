"""
build_faiss_index.py

保存済みEmbeddingからFAISSインデックスを作成する。
"""

from pathlib import Path

import faiss
import numpy as np

from config import (
    EMBEDDING_FILE,
    FAISS_INDEX_FILE,
    INDEX_TYPE,
    USE_GPU,
)


def load_embeddings(
    file_path: str | Path,
) -> np.ndarray:
    """
    保存済みEmbeddingを読み込む。

    Returns
    -------
    np.ndarray
        shape = (num_samples, embedding_dim)
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            "Embeddingファイルが見つかりません。"
            f"\npath: {path}"
        )

    embeddings = np.load(
        path,
        allow_pickle=False,
    )
    # (num_samples, embedding_dim)

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddingは2次元配列である必要があります。"
            f"\nshape: {embeddings.shape}"
        )

    if len(embeddings) == 0:
        raise ValueError(
            "Embeddingが空です。"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "EmbeddingにNaNまたはInfが含まれています。"
        )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )
    # (num_samples, embedding_dim)

    return embeddings


def create_cpu_index(
    embedding_dim: int,
    index_type: str,
) -> faiss.Index:
    """
    CPU版FAISSインデックスを作成する。

    Parameters
    ----------
    embedding_dim:
        Embeddingの次元数。

    index_type:
        flat_l2:
            ユークリッド距離による検索。

        flat_ip:
            内積による検索。
            L2正規化済みEmbeddingではコサイン類似度相当。
    """
    index_type = index_type.lower()

    if index_type == "flat_l2":
        return faiss.IndexFlatL2(
            embedding_dim
        )

    if index_type == "flat_ip":
        return faiss.IndexFlatIP(
            embedding_dim
        )

    raise ValueError(
        "未対応のINDEX_TYPEです。"
        "\n対応値: flat_l2 / flat_ip"
        f"\n現在値: {index_type}"
    )


def move_index_to_gpu(
    cpu_index: faiss.Index,
) -> tuple[
    faiss.Index,
    object | None,
]:
    """
    利用可能であればFAISSインデックスをGPUへ移動する。

    Returns
    -------
    index:
        CPUまたはGPUのFAISSインデックス。

    resources:
        GPUリソース。
        CPU使用時はNone。
    """
    if not USE_GPU:
        print("FAISS device : CPU")
        return cpu_index, None

    if not hasattr(
        faiss,
        "StandardGpuResources",
    ):
        print(
            "警告: GPU対応FAISSが利用できないため、"
            "CPUを使用します。"
        )
        return cpu_index, None

    try:
        resources = faiss.StandardGpuResources()

        gpu_index = faiss.index_cpu_to_gpu(
            resources,
            0,
            cpu_index,
        )

        print("FAISS device : GPU 0")

        return gpu_index, resources

    except RuntimeError as error:
        print(
            "警告: FAISSインデックスのGPU移動に"
            "失敗したため、CPUを使用します。"
        )
        print(f"詳細: {error}")

        return cpu_index, None


def add_embeddings(
    index: faiss.Index,
    embeddings: np.ndarray,
) -> None:
    """
    EmbeddingをFAISSインデックスへ登録する。
    """
    index.add(
        embeddings
    )
    # embeddings: (num_samples, embedding_dim)

    if index.ntotal != len(embeddings):
        raise RuntimeError(
            "FAISSへ登録されたデータ数が一致しません。"
            f"\nexpected: {len(embeddings)}"
            f"\nactual  : {index.ntotal}"
        )


def convert_to_cpu_index(
    index: faiss.Index,
) -> faiss.Index:
    """
    GPUインデックスを保存可能なCPU形式へ変換する。
    """
    if hasattr(
        faiss,
        "index_gpu_to_cpu",
    ):
        try:
            return faiss.index_gpu_to_cpu(
                index
            )
        except RuntimeError:
            pass

    return index


def save_index(
    index: faiss.Index,
    file_path: str | Path,
) -> Path:
    """
    FAISSインデックスを保存する。
    """
    path = Path(file_path).resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cpu_index = convert_to_cpu_index(
        index
    )

    faiss.write_index(
        cpu_index,
        str(path),
    )

    return path


def main() -> None:
    """
    FAISSインデックスを構築して保存する。
    """
    print("=" * 70)
    print("FAISSインデックス構築")
    print("=" * 70)

    embeddings = load_embeddings(
        EMBEDDING_FILE
    )

    num_samples, embedding_dim = (
        embeddings.shape
    )

    print(
        f"Embedding shape : {embeddings.shape}"
    )
    print(
        f"Embedding dtype : {embeddings.dtype}"
    )
    print(
        f"Index type      : {INDEX_TYPE}"
    )

    cpu_index = create_cpu_index(
        embedding_dim=embedding_dim,
        index_type=INDEX_TYPE,
    )

    index, gpu_resources = (
        move_index_to_gpu(
            cpu_index
        )
    )

    # 処理中にGPUリソースが破棄されないよう参照を保持する
    _ = gpu_resources

    add_embeddings(
        index=index,
        embeddings=embeddings,
    )

    output_path = save_index(
        index=index,
        file_path=FAISS_INDEX_FILE,
    )

    print("-" * 70)
    print(f"登録件数 : {index.ntotal}")
    print(f"次元数   : {embedding_dim}")
    print(f"保存先   : {output_path}")
    print("-" * 70)
    print("FAISSインデックスの構築が完了しました。")


if __name__ == "__main__":
    main()