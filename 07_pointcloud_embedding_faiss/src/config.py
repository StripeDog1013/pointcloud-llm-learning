"""
config.py

プロジェクト共通設定
"""

# ==============================================================================
# GPU設定
# ==============================================================================

# 使用する論理CUDAデバイスID
CUDA_ID = 0

# CUDA_VISIBLE_DEVICESで指定する物理GPU ID
PHYSICAL_CUDA_ID = 0

# Trueの場合、CUDA_VISIBLE_DEVICESを設定してGPUを固定する
USE_CUDA_VISIBLE_DEVICES = True

# ==============================================================================
# Dataset
# ==============================================================================

DATASET_TYPE = "modelnet40"      # modelnet10 / modelnet40 / folder

NUM_CLASSES = (
    10
    if DATASET_TYPE == "modelnet10"
    else 40
)

TRAIN_DIR =  "../../02_pointnet_classification/data/ModelNet40/" # "../data/train"

TEST_DIR = "../../02_pointnet_classification/data/ModelNet40/" # "../data/test"

NUM_POINTS = 1024

BATCH_SIZE = 32

NUM_WORKERS = 4

RANDOM_SEED = 42

# ==============================================================================
# Checkpoint
# ==============================================================================

CHECKPOINT_DIR = "../checkpoints"

CHECKPOINT_NAME = (
    "pointnet2_best_10.pth"
    if DATASET_TYPE == "modelnet10"
    else "pointnet2_best_40.pth"
)

# ==============================================================================
# Embedding
# ==============================================================================

# 使用するバックボーン
BACKBONE = "pointnet2"

POINTNET2_CHECKPOINT_PATH = "../../03_pointnet2_classification/checkpoints/" + CHECKPOINT_NAME

# PointNet++ Global Feature次元
EMBEDDING_DIM = 1024

# EmbeddingをL2正規化する
NORMALIZE_EMBEDDING = True

# ------------------------------------------------------------------------------
# Set Abstraction Layer 1
# ------------------------------------------------------------------------------

SA1_NUM_POINTS = 512

SA1_RADIUS = 0.20

SA1_NUM_SAMPLES = 32

# ------------------------------------------------------------------------------
# Set Abstraction Layer 2
# ------------------------------------------------------------------------------

SA2_NUM_POINTS = 128

SA2_RADIUS = 0.40

SA2_NUM_SAMPLES = 64

# ==============================================================================
# FAISS
# ==============================================================================

# flat_l2
# flat_ip
INDEX_TYPE = "flat_l2"

# faiss-gpu使用
USE_GPU = False

TOP_K = 5

# ==============================================================================
# Embedding保存
# ==============================================================================

EMBEDDING_DIR = "../embeddings"

EMBEDDING_FILE = (
    f"{EMBEDDING_DIR}/embeddings.npy"
)

LABEL_FILE = (
    f"{EMBEDDING_DIR}/labels.npy"
)

INDEX_FILE = (
    f"{EMBEDDING_DIR}/indices.npy"
)

# ==============================================================================
# FAISS Index
# ==============================================================================

FAISS_INDEX_DIR = "../faiss_index"

FAISS_INDEX_FILE = (
    f"{FAISS_INDEX_DIR}/{INDEX_TYPE}.index"
)

# ==============================================================================
# Log
# ==============================================================================

LOG_DIR = "../logs"

# ==============================================================================
# Output
# ==============================================================================

OUTPUT_DIR = "../outputs"