# 06 Point Cloud Segmentation (PointNet++)

## 概要

本セクションでは、**PointNet++** を用いた点群セグメンテーションを学習します。

分類では点群全体に1つのラベルを付与しましたが、セグメンテーションでは**各点ごとにラベルを予測**します。

例

| タスク | 出力 |
|---------|------|
| Classification | Chair |
| Segmentation | Seat, Back, Leg, Arm |

PointNet++では、

- Set Abstraction (Encoder)
- Feature Propagation (Decoder)

から構成されるEncoder-Decoderネットワークにより、高精度な点群セグメンテーションを実現します。

---

# 学習目標

本セクションでは以下を理解することを目標とします。

- PointNet++ Segmentationの構造
- Set Abstraction
- Feature Propagation
- Skip Connection
- 3-NN Feature Interpolation
- Point-wise Classification
- Point Accuracy
- Instance mIoU
- Category mIoU

---

# フォルダ構成

```text
06_pointcloud_segmentation/
│
├── checkpoints/
├── data/
├── logs/
├── outputs/
│
├── src/
│   ├── config.py
│   ├── dataset_utils.py
│   ├── folder_dataset.py
│   ├── dataset.py
│   ├── model_utils.py
│   ├── pointnet2_layers.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── visualize_samples.py
│   ├── visualize_prediction.py
│   ├── run_all.py
│   └── utils.py
│
└── README.md
```

---

# 使用データセット

デフォルトでは **ShapeNet Part** を使用します。

- 16カテゴリ
- 約17,000点群
- 50種類の部品ラベル

例

| カテゴリ | 部品 |
|-----------|------|
| Airplane | Body / Wing / Tail / Engine |
| Chair | Seat / Back / Leg / Arm |
| Table | Top / Leg |

また、自前データセットにも対応できる構成となっています。

---

# ネットワーク構成

```text
Input Point Cloud
        │
        ▼
Set Abstraction 1
        │
        ▼
Set Abstraction 2
        │
        ▼
Global Set Abstraction
        │
        ▼
Feature Propagation 3
        │
        ▼
Feature Propagation 2
        │
        ▼
Feature Propagation 1
        │
        ▼
Point-wise Classification
        │
        ▼
Part Label
```

Encoderで特徴を抽出し、Decoderで元の点数まで特徴を復元します。

---

# 学習

```bash
python train.py
```

実行内容

- Dataset読み込み
- PointNet++生成
- 学習
- Validation
- Checkpoint保存

Validationでは以下を計算します。

- Loss
- Point Accuracy
- Instance mIoU
- Category mIoU

Checkpointは

**Validation Instance mIoU**

が最高となったモデルを保存します。

---

# 評価

```bash
python evaluate.py
```

出力例

```text
Test Loss
Point Accuracy
Instance mIoU
Category mIoU
```

カテゴリ毎のIoUも表示します。

---

# 推論

テストデータ

```bash
python inference.py
```

指定サンプル

```bash
python inference.py --index 10
```

自前点群

```bash
python inference.py \
    --input sample.ply \
    --category Chair
```

対応フォーマット

- PLY
- PCD
- LAS
- LAZ
- OFF

---

# データセット可視化

```bash
python visualize_samples.py
```

表示内容

- 点群
- 部品ラベル
- 部品ごとの点数

---

# 推論結果可視化

```bash
python visualize_prediction.py
```

表示内容

- Ground Truth
- Prediction
- Point Accuracy
- Instance mIoU

Ground TruthとPredictionを色分け表示し、予測結果を視覚的に確認できます。

---

# Point Accuracy

各点が正しく分類された割合です。

```text
Correct Points
-----------------
Total Points
```

単純で分かりやすい指標ですが、部品ごとの品質は評価できません。

---

# IoU (Intersection over Union)

部品ごとの重なり具合を表します。

```text
            Prediction
        +------------------+
        |      ####        |
        |   ##########     |
        | ####GT######     |
        | ###########      |
        |     #####        |
        +------------------+

IoU = Intersection / Union
```

IoUはセグメンテーションで最も一般的な評価指標です。

---

# Instance mIoU

各点群についてIoUを計算し、その平均を求めます。

```text
Point Cloud 1
Point Cloud 2
Point Cloud 3
・・・
      │
      ▼

Average
```

サンプル数の多いカテゴリの影響を受けやすい特徴があります。

---

# Category mIoU

まずカテゴリ毎に平均IoUを求めます。

```text
Chair Average IoU
Table Average IoU
Lamp Average IoU
・・・
```

最後にカテゴリ間で平均します。

```text
Average(Category IoU)
```

各カテゴリを均等に評価できるため、ShapeNet Partでは重要な指標です。

---

# PointNet++ Segmentationの特徴

## Encoder

- Farthest Point Sampling
- Ball Query
- Shared MLP
- Max Pooling

局所特徴を抽出します。

---

## Decoder

- Feature Propagation
- 3-NN Interpolation
- Skip Connection

粗い特徴を元の点群へ戻します。

---

## Skip Connection

Encoderで得られた高解像度特徴をDecoderへ直接渡します。

```text
Encoder --------+
                │
                ▼
             Decoder
```

位置情報を失わずに細かな部品を復元できます。

---

## 3-NN Interpolation

Feature Propagationでは

近傍3点

を利用して特徴を補間します。

```text
Nearest 3 Points
      │
      ▼
Weighted Average
```

---

# 学習の流れ

1. 点群入力
2. Set Abstraction
3. Global Feature抽出
4. Feature Propagation
5. Point-wise Classification
6. Loss計算
7. Backpropagation
8. Validation
9. Instance mIoU更新ならCheckpoint保存

---

# ここまで学んだモデル

| セクション | タスク |
|------------|---------|
| 02 | PointNet |
| 03 | PointNet++ |
| 04 | DGCNN |
| 05 | Point Transformer |
| **06** | **PointNet++ Segmentation** |

---

# 次のセクション

次の **07_pointcloud_embedding_faiss** では、点群からEmbeddingを抽出し、FAISSを用いた類似検索を学習します。

主な内容

* 点群特徴量の抽出
* Embeddingベクトルの保存
* FAISSインデックスの構築
* 類似点群検索
* Top-K検索
* 距離指標の比較
* PointNet++やPoint TransformerのEncoder活用

これにより、

```text
入力点群
   ↓
Embedding
   ↓
FAISS検索
   ↓
類似点群を取得
```

という、今後のPoint Cloud RAGにつながる基盤を構築します。