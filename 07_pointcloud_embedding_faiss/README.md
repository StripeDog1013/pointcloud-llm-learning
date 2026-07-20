# 07 Point Cloud Embedding + FAISS

## 概要

本セクションでは、PointNet++で学習済みの分類モデルを特徴抽出器（Embedding Model）として利用し、点群データの類似検索（Similarity Search）を行います。

分類モデルの最終分類層は使用せず、中間特徴ベクトル（Embedding）を取得し、それらをFAISSへ登録することで高速な近傍探索を実現します。

本章で学ぶ内容

- PointNet++をEmbedding Modelとして利用する方法
- Embeddingの生成
- FAISS Indexの構築
- 類似点群検索
- Open3Dによる検索結果の可視化

---

# フォルダ構成

```text
07_pointcloud_embedding_faiss/
│
├── checkpoints/
├── data/
├── outputs/
├── logs/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── model_utils.py
│   ├── pointnet2_layers.py
│   ├── embedding_model.py
│   ├── build_embeddings.py
│   ├── build_faiss_index.py
│   ├── search.py
│   └── visualize_search.py
│
└── README.md
```

---

# 学習の流れ

```
ModelNet
     │
     ▼
PointNet++
     │
     ▼
Embedding抽出
     │
     ▼
embeddings.npy
     │
     ▼
FAISS Index作成
     │
     ▼
類似点群検索
     │
     ▼
Open3D可視化
```

---

# Embeddingとは

通常の分類モデルでは

```
Point Cloud
      │
      ▼
PointNet++
      │
      ▼
1024次元特徴
      │
      ▼
Linear
      │
      ▼
Class
```

となります。

本章では分類層を使用せず、

```
Point Cloud
      │
      ▼
PointNet++
      │
      ▼
1024次元Embedding
```

のみを取り出します。

Embedding同士の距離を比較することで、

- 形状が似ている
- 特徴が似ている

点群を検索できます。

---

# Embedding生成

```
python build_embeddings.py
```

生成されるファイル

```
outputs/

├── embeddings.npy
├── labels.npy
└── indices.npy
```

### embeddings.npy

各点群のEmbedding

```
(num_samples, embedding_dim)
```

例

```
(3991, 1024)
```

---

### labels.npy

各Embeddingに対応するラベル

```
(num_samples,)
```

---

### indices.npy

元データセットのインデックス

```
(num_samples,)
```

可視化時に元の点群を取得するために利用します。

---

# FAISS Index作成

```
python build_faiss_index.py
```

生成されるファイル

```
outputs/

└── pointcloud.index
```

このIndexにはEmbeddingのみが保存されます。

---

# 類似検索

```
python search.py
```

例

```
Query Label : chair

1  label=chair
2  label=chair
3  label=chair
4  label=chair
5  label=chair
```

---

# 検索結果の可視化

```
python visualize_search.py
```

表示内容

- 赤：検索クエリ
- 青：検索結果

```
Query      Rank1      Rank2      Rank3 ...
```

Open3D上で横並びに表示されます。

---

# FAISSとは

FAISS（Facebook AI Similarity Search）はMetaが開発した高速近傍探索ライブラリです。

大量の特徴ベクトルから、

- 最近傍検索
- 類似検索
- ベクトル検索

を高速に実行できます。

本章では

```
Embedding
      │
      ▼
FAISS
      │
      ▼
Nearest Neighbor Search
```

を体験します。

---

# 使用したIndex

本章では以下のどちらかを利用します。

## IndexFlatL2

ユークリッド距離

```
distance = ||x - y||
```

特徴量同士の距離が近いほど類似していると判断します。

---

## IndexFlatIP

内積

Embeddingを正規化すると

```
Inner Product
    ≒
Cosine Similarity
```

として利用できます。

---

# 学んだこと

- PointNet++を特徴抽出器として利用する方法
- Embeddingの作成
- Embeddingの保存
- FAISSによる高速近傍探索
- 類似点群検索
- Open3Dによる検索結果の可視化

---

# 次のステップ

Embedding検索はRAGやベクトルデータベースの基本技術です。

この技術を発展させることで、

- Point Cloud RAG
- Point Cloud Retrieval
- Shape Search
- 類似3Dモデル検索
- 3Dモデル推薦
- Point Cloud × LLM

などへ応用できます。