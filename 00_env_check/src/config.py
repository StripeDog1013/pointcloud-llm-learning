"""
config.py

プロジェクト共通設定
"""

# ------------------------------------------------------------------------------
# GPU設定
# ------------------------------------------------------------------------------

# 使用する論理CUDAデバイスID
CUDA_ID = 0

# CUDA_VISIBLE_DEVICESで指定する物理GPU ID
PHYSICAL_CUDA_ID = 0

# Trueの場合、CUDA_VISIBLE_DEVICESを設定してGPUを固定する
USE_CUDA_VISIBLE_DEVICES = True