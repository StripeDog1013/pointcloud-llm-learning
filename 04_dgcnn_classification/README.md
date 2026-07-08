# 04 DGCNN Classification

## 概要

本セクションでは、**Dynamic Graph CNN（DGCNN）** を用いた3次元点群分類を学習します。

PointNetでは点ごとの特徴を学習し、PointNet++では局所領域を階層的に学習しました。

DGCNNではさらに発展し、**点と点の関係（エッジ）** に着目します。各層でk近傍グラフを動的に再構築し、**EdgeConv** を用いて局所幾何学的特徴を学習します。

本プロジェクトでは、DGCNN論文に準拠した分類モデルを実装します。

---

# PointNet++からの主な変更点

追加された主な技術

* k-Nearest Neighbor (kNN)
* Edge Feature
* EdgeConv
* Dynamic Graph

PointNet++

```text
Point Cloud
      │
      ▼
FPS
      │
      ▼
Ball Query
      │
      ▼
Set Abstraction
      │
      ▼
Classification
```

DGCNN

```text
Point Cloud
      │
      ▼
kNN
      │
      ▼
Edge Feature
      │
      ▼
EdgeConv
      │
      ▼
Dynamic Graph
      │
      ▼
Classification
```

---

# 学習内容

* Dynamic Graph CNN
* k-Nearest Neighbor (kNN)
* Edge Feature
* EdgeConv
* Dynamic Graph
* Feature Concatenation
* Global Pooling
* Checkpoint保存・読込
* 推論

---

# フォルダ構成

```text
04_dgcnn_classification/
├── checkpoints/
├── logs/
├── data/
├── models/
├── outputs/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── dataset_utils.py
│   ├── modelnet_dataset.py
│   ├── folder_dataset.py
│   ├── model_utils.py
│   ├── dgcnn_layers.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── visualize_samples.py
│   └── run_all.py
└── README.md
```

---

# 共通ライブラリ

本セクションから、複数のセクションで利用する処理をレポジトリ直下の `common` フォルダへ集約しています。

```text
common/
├── device.py
├── utils.py
├── path_utils.py
├── log_utils.py
├── train_utils.py
├── point_io.py
├── point_utils.py
├── visualize_utils.py
└── run_utils.py
```

これにより、以降のPoint TransformerやSegmentationなどのセクションでも共通コードを再利用できます。

---

# 使用するデータセット

* ModelNet10
* ModelNet40
* 自前点群データ

対応形式

* `.ply`
* `.pcd`
* `.las`
* `.laz`

PyTorch Geometricを利用してModelNetを自動ダウンロード・前処理します。

---

# 各プログラム

## model_utils.py

DGCNNで使用する点群処理を実装します。

学習内容

* kNN
* Edge Feature生成

---

## dgcnn_layers.py

EdgeConvブロックを実装します。

学習内容

* Shared MLP
* Max Pooling
* EdgeConv

---

## model.py

DGCNN分類モデル本体です。

構成

* EdgeConv ×4
* Feature Concatenation
* Embedding Layer
* Global Max Pooling
* Global Average Pooling
* Classification Head

---

## train.py

DGCNNの学習を行います。

内容

* DataLoader
* CrossEntropyLoss
* Accuracy
* Checkpoint保存
* JSONログ保存

---

## evaluate.py

学習済みモデルの評価を行います。

内容

* Checkpoint読込
* Test Accuracy
* Test Loss

---

## inference.py

1つの点群ファイルを分類します。

対応形式

* OFF
* PLY
* PCD
* LAS
* LAZ

---

## visualize_samples.py

Dataset内の点群サンプルをOpen3Dで表示します。

---

## run_all.py

本セクションをまとめて実行します。

---

# Edge Feature

DGCNNでは、各点と近傍点との関係を特徴量として学習します。

各点に対して

```text
x_i
```

と近傍点

```text
x_j
```

から

```text
[x_j - x_i, x_i]
```

というEdge Featureを生成します。

これにより、点そのものだけでなく、**近傍との相対的位置関係**も学習できます。

---

# Dynamic Graph

DGCNN最大の特徴は、**各EdgeConv層ごとにkNNを再計算する**ことです。

```text
Point Cloud
      │
      ▼
EdgeConv
      │
      ▼
新しい特徴空間
      │
      ▼
kNN再計算
      │
      ▼
EdgeConv
```

入力空間だけでなく、特徴空間上でも近傍関係を更新することで、より高品質な特徴表現を獲得できます。

---

# モデル構成

```text
Point Cloud
      │
      ▼
EdgeConv 1
      │
      ▼
EdgeConv 2
      │
      ▼
EdgeConv 3
      │
      ▼
EdgeConv 4
      │
      ▼
Feature Concatenation
      │
      ▼
Embedding
      │
      ▼
Global Max Pooling
      │
      ├──────────────┐
      ▼              ▼
Global Avg Pooling
      │              │
      └──────┬───────┘
             ▼
      Fully Connected
             ▼
      Classification
```

---

# PointNet++との比較

| PointNet++      | DGCNN         |
| --------------- | ------------- |
| FPS             | kNN           |
| Ball Query      | Edge Feature  |
| Set Abstraction | EdgeConv      |
| 固定した局所領域        | 動的に更新される近傍グラフ |
| 局所点群            | 点間関係          |

---

# 実行方法

`src` ディレクトリへ移動します。

```bash
cd src
```

個別実行

```bash
python model.py
python train.py
python evaluate.py
python inference.py
```

まとめて実行

```bash
python run_all.py
```

---

# 本セクションで学ぶ重要な概念

* Dynamic Graph CNN
* k-Nearest Neighbor
* Edge Feature
* EdgeConv
* Dynamic Graph
* Feature Concatenation
* Global Pooling
* Graph-based Learning

---

# 次のセクション

次の **05_point_transformer_classification** では、Transformerを3次元点群へ適用した **Point Transformer** を学習します。

これまでのCNNベースの手法とは異なり、**Self-Attention** によって点同士の関係を学習します。

Hugging Face学習で学んだTransformerの知識を、3次元点群へ応用する最初のセクションとなります。
