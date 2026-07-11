# 05 Point Transformer Classification

## 概要

本セクションでは、**Point Transformer** を用いた3次元点群分類を学習します。

これまで学習したモデルとの違いは以下の通りです。

* **PointNet**：点ごとの特徴を学習
* **PointNet++**：局所領域を階層的に学習
* **DGCNN**：点同士の関係（Edge Feature）を学習
* **Point Transformer**：Self-Attentionによって重要な近傍点との関係を学習

本セクションでは、Point Transformer（ICCV 2021）論文を参考に、**Transition Downを含む階層型Point Transformer** を実装します。

---

# Point Transformerとは

Transformerは自然言語処理で広く利用されていますが、Point TransformerではそのSelf-Attentionを3次元点群へ適用します。

自然言語

```text
Token
    ↓
Embedding
    ↓
Self Attention
    ↓
Encoder
```

Point Transformer

```text
Point
    ↓
Feature Embedding
    ↓
Point Attention
    ↓
Point Transformer Block
```

Transformerの考え方をそのまま点群へ応用したモデルです。

---

# 学習内容

* Point Transformer
* Relative Position Encoding
* Vector Attention
* Point Transformer Block
* Transition Down
* Farthest Point Sampling (FPS)
* k-Nearest Neighbor (kNN)
* Local Self-Attention
* Global Pooling

---

# フォルダ構成

```text
05_point_transformer_classification/
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
│   ├── positional_encoding.py
│   ├── transformer_layers.py
│   ├── transition_down.py
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

本プロジェクト共通で利用する処理は `common/` フォルダへまとめています。

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

---

# 使用するデータセット

* ModelNet10
* ModelNet40
* 自前点群データ

対応形式

* OFF
* PLY
* PCD
* LAS
* LAZ

PyTorch Geometricを利用してModelNetを自動ダウンロードします。

---

# モデル構成

本セクションではTransition Downを含む階層型Point Transformerを実装します。

```text
Point Cloud
      │
      ▼
Feature Embedding
      │
      ▼
Stage 1
Point Transformer
      │
      ▼
Transition Down
      │
      ▼
Stage 2
Point Transformer
      │
      ▼
Transition Down
      │
      ▼
Stage 3
Point Transformer
      │
      ▼
Transition Down
      │
      ▼
Stage 4
Point Transformer
      │
      ▼
Transition Down
      │
      ▼
Stage 5
Point Transformer
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
Classification Head
             ▼
Class Prediction
```

---

# Point Transformer Block

Point Transformer Blockでは、各点について近傍点のみを対象としたSelf-Attentionを計算します。

```text
Input Feature
      │
      ▼
kNN
      │
      ▼
Relative Position
      │
      ▼
Vector Attention
      │
      ▼
Residual Connection
      │
      ▼
Output Feature
```

通常のTransformerが全Token同士のAttentionを計算するのに対し、Point Transformerでは**近傍点のみ**を対象とします。

---

# Relative Position Encoding

LLMでは絶対位置を利用します。

```text
Token Position
```

Point Transformerでは相対位置を利用します。

```text
Point i
Point j

↓

Point j - Point i
```

この相対位置をMLPで特徴空間へ写像し、Attention計算へ利用します。

---

# Vector Attention

通常のTransformer

```text
Softmax(QKᵀ)
```

Point Transformer

```text
MLP(Q − K + Position)
```

Attention Weightへ相対位置特徴を直接組み込む点が大きな特徴です。

---

# Transition Down

Point数を段階的に削減しながら特徴量を増加させます。

```text
1024 points
      │
      ▼
256 points
      │
      ▼
64 points
      │
      ▼
16 points
      │
      ▼
4 points
```

各Transition Downでは

* Farthest Point Sampling
* kNN
* Shared MLP
* Max Pooling

を行います。

---

# Feature Dimension

各Stageで特徴次元を増加させます。

```text
64
 ↓
128
 ↓
256
 ↓
512
 ↓
1024
```

点数は減りますが、各点が保持する情報量は増加していきます。

---

# 各プログラム

## model_utils.py

Point群操作を実装します。

内容

* Square Distance
* kNN
* FPS
* Grouping

---

## positional_encoding.py

Relative Position Encodingを実装します。

内容

* Relative Coordinate
* LayerNorm
* MLP

---

## transformer_layers.py

Point Transformer Layerを実装します。

内容

* Query
* Key
* Value
* Vector Attention
* Residual Connection

---

## transition_down.py

Transition Downを実装します。

内容

* FPS
* Neighbor Search
* Feature Aggregation

---

## model.py

Point Transformer分類モデル本体です。

---

## train.py

学習を行います。

内容

* Checkpoint保存
* JSONログ保存
* Accuracy表示

---

## evaluate.py

学習済みモデルの評価を行います。

---

## inference.py

1つの点群ファイルを分類します。

---

## visualize_samples.py

データセット内の点群を表示します。

---

## run_all.py

本セクションをまとめて実行します。

---

# PointNet・PointNet++・DGCNNとの比較

| モデル               | 特徴             |
| ----------------- | -------------- |
| PointNet          | 点単位            |
| PointNet++        | 局所領域           |
| DGCNN             | Edge Feature   |
| Point Transformer | Self-Attention |

---

# Hugging Faceとの対応

本セクションは、Hugging Face学習で学んだTransformerとの共通点を理解することが重要です。

| Hugging Face        | Point Transformer          |
| ------------------- | -------------------------- |
| Token               | Point                      |
| Token Embedding     | Point Feature Embedding    |
| Position Encoding   | Relative Position Encoding |
| Self-Attention      | Vector Attention           |
| Transformer Block   | Point Transformer Block    |
| Encoder             | Point Transformer Encoder  |
| Pooling             | Global Pooling             |
| Classification Head | Classification Head        |

Transformerの基本構造は同じであり、入力データが文章から3次元点群へ変わっただけと考えることができます。

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

* Self-Attention
* Vector Attention
* Relative Position Encoding
* Farthest Point Sampling
* Transition Down
* Hierarchical Feature Learning
* Global Pooling
* Point Transformer

---

# 次のセクション

次の **06 Point Transformer Segmentation**（または **Point-BERT** へ進む場合）では、分類ではなく点ごとのラベル予測や自己教師あり学習を扱います。

ここまで学習したPoint Transformerの構造を基盤として、3D Foundation Modelや3D-LLMへと発展させていきます。
