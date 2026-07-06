# 03 PointNet++ Classification

## 概要

本セクションでは、**PointNet++** を用いた3次元点群分類を学習します。

前セクションで学んだPointNetは点群全体から特徴を抽出するモデルでしたが、局所的な形状を十分に捉えることができませんでした。

PointNet++では、点群を階層的にサンプリングし、それぞれの近傍領域から特徴を抽出することで、局所特徴と大域特徴の両方を学習します。

本プロジェクトでは、PointNet++論文に準拠したSingle-Scale Grouping（SSG）版を実装します。

---

# PointNetからの主な変更点

追加された主な技術

* Farthest Point Sampling (FPS)
* Ball Query
* Set Abstraction Layer
* Hierarchical Feature Learning

PointNet

```text
Point Cloud
      │
      ▼
Shared MLP
      │
      ▼
Global Max Pooling
      │
      ▼
Classification
```

PointNet++

```text
Point Cloud
      │
      ▼
Set Abstraction 1
      │
      ▼
Set Abstraction 2
      │
      ▼
Set Abstraction 3
      │
      ▼
Classification
```

---

# 学習内容

* PointNet++のネットワーク構造
* Farthest Point Sampling
* Ball Query
* Grouping
* Set Abstraction
* Shared MLP
* Hierarchical Feature Learning
* PointNet++分類
* Checkpoint保存・読込
* 推論

---

# フォルダ構成

```text
03_pointnet2_classification/
├── checkpoints/
├── logs/
├── data/
├── models/
├── outputs/
├── src/
│   ├── config.py
│   ├── device.py
│   ├── utils.py
│   ├── dataset.py
│   ├── dataset_utils.py
│   ├── modelnet_dataset.py
│   ├── folder_dataset.py
│   ├── model_utils.py
│   ├── pointnet2_layers.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── visualize_samples.py
│   └── run_all.py
└── README.md
```

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

PointNet++の基礎となる点群操作を実装します。

学習内容

* 点群間距離計算
* Farthest Point Sampling
* Ball Query
* Point Indexing
* Grouping

---

## pointnet2_layers.py

PointNet++のSet Abstraction Layerを実装します。

学習内容

* Local Feature Learning
* Shared MLP
* Local Max Pooling

---

## model.py

PointNet++分類モデル本体です。

構成

* SA Layer 1
* SA Layer 2
* Global SA Layer
* Fully Connected Layer
* Classification Head

---

## train.py

PointNet++の学習を行います。

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

# PointNet++の処理フロー

```text
Point Cloud
      │
      ▼
Farthest Point Sampling
      │
      ▼
Ball Query
      │
      ▼
Grouping
      │
      ▼
Shared MLP
      │
      ▼
Max Pooling
      │
      ▼
Local Feature
      │
      ▼
Set Abstraction
      │
      ▼
Classification
```

---

# Set Abstraction

PointNet++の中心となる処理です。

1. FPSで代表点を選択
2. Ball Queryで近傍点を取得
3. Shared MLPで局所特徴抽出
4. Max Poolingで代表特徴を生成

これを複数回繰り返すことで、より大きな範囲の特徴を学習できます。

---

# PointNetとの比較

| PointNet         | PointNet++             |
| ---------------- | ---------------------- |
| Global Featureのみ | Local + Global Feature |
| T-Net            | Set Abstraction        |
| 点群全体を一括処理        | 階層的処理                  |
| 局所形状に弱い          | 局所形状に強い                |

---

# 実行方法

`src` ディレクトリへ移動します。

```bash
cd src
```

個別実行

```bash
python dataset.py
python model.py
python visualize_samples.py
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

* PointNet++
* Set Abstraction
* Farthest Point Sampling
* Ball Query
* Grouping
* Local Feature
* Hierarchical Feature Learning
* Shared MLP
* Max Pooling

---

# 次のセクション

次の **04_dgcnn_classification** では、グラフニューラルネットワーク（GNN）の考え方を取り入れた **Dynamic Graph CNN (DGCNN)** を学習します。

PointNet++が「局所点群」を扱うのに対し、DGCNNでは点群から動的にグラフを構築し、**EdgeConv** を用いて点間の関係性を学習します。これにより、より豊かな局所幾何学的特徴を抽出できるようになります。
