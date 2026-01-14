#!/usr/bin/env python3
"""测试所有依赖是否正确安装"""

print("测试基础依赖...")
import torch
import numpy as np
import cv2
from PIL import Image
import tqdm
print("✓ 基础依赖OK")

print("\n测试transformers...")
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
print("✓ transformers OK")

print("\n测试supervision...")
import supervision as sv
print("✓ supervision OK")

print("\n测试decord...")
import decord
print("✓ decord OK")

print("\n测试sklearn...")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
print("✓ sklearn OK")

print("\n测试SAM2...")
try:
    from thirdparty.grounded_sam_2.sam2.build_sam import (
        build_sam2_video_predictor,
        build_sam2,
    )
    print("✓ SAM2导入OK")
except ImportError as e:
    print(f"✗ SAM2导入失败: {e}")
    print("  请检查thirdparty/grounded_sam_2是否正确安装")

print("\n测试模型权重...")
import os

# 检查SAM2权重
sam2_checkpoint = "thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt"
if os.path.exists(sam2_checkpoint):
    print(f"✓ SAM2权重存在: {sam2_checkpoint}")
else:
    print(f"✗ SAM2权重不存在: {sam2_checkpoint}")
    print("  请下载: wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt")

print("\n所有测试完成！")

