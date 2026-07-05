# 01 Open3D Basics

## 概要

本セクションでは、**Open3D** を用いた3次元点群処理の基礎を学習します。

今後学習する **PointNet**、**PointNet++**、**Point Transformer**、**3D Object Detection** などのモデルでは、点群データの読み込みや前処理が必須となるため、本セクションで基本操作を身に付けます。

---

# 学習内容

* Open3Dによる点群の読み込み
* 点群の可視化
* NumPy配列から点群の生成
* Voxel Down Sampling
* 法線推定（Normal Estimation）
* 点群の切り出し（Crop）
* 点群の座標変換
* Bounding Boxの計算
* KDTreeによる近傍探索
* ICPによる点群位置合わせ

---

# フォルダ構成

```text
01_open3d_basics/
├── checkpoints/
├── logs/
├── data/
├── models/
├── outputs/
├── src/
│   ├── config.py
│   ├── device.py
│   ├── utils.py
│   ├── create_point_cloud.py
│   ├── load_point_cloud.py
│   ├── visualize_point_cloud.py
│   ├── voxel_downsample.py
│   ├── estimate_normals.py
│   ├── crop_point_cloud.py
│   ├── transform_point_cloud.py
│   ├── compute_bounding_box.py
│   ├── nearest_neighbor.py
│   ├── icp_registration.py
│   └── run_all.py
└── README.md
```

---

# 使用する点群データ

本セクションでは Stanford Bunny を回転させた以下の点群データを使用します。

```text
data/
├── bun000.ply
├── bun045.ply
├── bun090.ply
├── bun180.ply
├── bun270.ply
└── bun315.ply
```

各ファイルは Bunny を Y軸方向へ回転させた点群です。

---

# 各プログラム

## create_point_cloud.py

NumPy配列から点群を生成し、PLY形式で保存します。

学習内容

* PointCloud
* Vector3dVector
* 点群保存

---

## load_point_cloud.py

PLY形式の点群を読み込みます。

学習内容

* 点群ファイルの読み込み
* 点群情報の取得

---

## visualize_point_cloud.py

回転済み点群を順番に表示します。

学習内容

* Open3D Viewer
* 点群の可視化
* カメラ操作

---

## voxel_downsample.py

Voxel Down Sampling を実行します。

学習内容

* 点群のダウンサンプリング
* 点数削減

---

## estimate_normals.py

法線ベクトルを推定します。

学習内容

* Normal Estimation
* KDTree Search
* Hybrid Search

---

## crop_point_cloud.py

Bounding Box を利用して点群を切り出します。

学習内容

* AxisAlignedBoundingBox
* Crop

---

## transform_point_cloud.py

点群の座標変換を行います。

学習内容

* Rotation
* Translation
* Scale

---

## compute_bounding_box.py

AABB と OBB を比較します。

学習内容

* Axis Aligned Bounding Box
* Oriented Bounding Box

---

## nearest_neighbor.py

KDTree による近傍探索を行います。

学習内容

* KDTree
* K-Nearest Neighbor

---

## icp_registration.py

ICPを用いて点群同士の位置合わせを行います。

学習内容

* ICP Registration
* Transformation Matrix
* Fitness
* RMSE

---

## run_all.py

本セクションのすべてのサンプルを順番に実行します。

---

# 実行方法

`src` ディレクトリへ移動します。

```bash
cd src
```

各サンプルは以下のように実行できます。

```bash
python create_point_cloud.py
python load_point_cloud.py
python visualize_point_cloud.py
python voxel_downsample.py
python estimate_normals.py
python crop_point_cloud.py
python transform_point_cloud.py
python compute_bounding_box.py
python nearest_neighbor.py
python icp_registration.py
```

すべてまとめて実行する場合は、

```bash
python run_all.py
```

---

# Open3D Viewer の操作

| 操作       | 内容           |
| -------- | ------------ |
| 左ドラッグ    | 回転           |
| 右ドラッグ    | 平行移動         |
| マウスホイール  | ズーム          |
| R        | カメラリセット      |
| Ctrl + C | カメラパラメータコピー  |
| Ctrl + V | カメラパラメータ貼り付け |

---

# 本セクションで学ぶ重要な概念

```text
Point Cloud
      │
      ▼
Open3D
      │
      ▼
Visualization
      │
      ▼
Down Sampling
      │
      ▼
Normal Estimation
      │
      ▼
Bounding Box
      │
      ▼
KDTree
      │
      ▼
ICP Registration
```

---

# 次のセクション

次の **02_pointnet_classification** では、本セクションで扱った点群データを入力として使用し、PointNetによる点群分類を学習します。

本セクションで学んだ以下の内容は、今後のすべての3D点群処理で基礎となります。

* 点群データの読み込み
* 点群の可視化
* 前処理
* 法線推定
* 座標変換
* KDTree
* ICP
