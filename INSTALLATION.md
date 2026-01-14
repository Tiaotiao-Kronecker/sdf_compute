# 安装指南

本指南说明如何安装 `brdige_dataset_process_depth.py` 脚本所需的所有依赖。

## 一、Python包依赖

### 1. 安装基础依赖

```bash
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import supervision; print('supervision:', supervision.__version__)"
python -c "import decord; print('decord:', decord.__version__)"
python -c "import sklearn; print('sklearn:', sklearn.__version__)"
python -c "import tiktoken; print('tiktoken:', tiktoken.__version__)"
```

### 3. 常见依赖问题

**Qwen-VL 需要的额外依赖**：

1. **tiktoken**：
   - 如果遇到 `ImportError: ... tiktoken`
   - 运行：`pip install tiktoken`

2. **accelerate**（必需）：
   - 如果遇到 `ValueError: Using a device_map ... requires accelerate`
   - 运行：`pip install accelerate`
   - 这是使用 `device_map` 参数所必需的

3. **optimum、auto-gptq 和 gptqmodel**（必需，用于GPTQ量化模型）：
   - 如果遇到 `ImportError: Loading a GPTQ quantized model requires optimum`
   - 运行：`pip install optimum auto-gptq gptqmodel -i https://pypi.org/simple`
   - Qwen-VL-Chat-Int4 使用GPTQ量化，需要这些库
   - **版本兼容性修复**：
     - `QuantizeConfig` 问题：代码已自动修复（将 `BaseQuantizeConfig` 映射为 `QuantizeConfig`）
     - `FORMAT` 问题：需要安装 `gptqmodel`，代码已添加兼容性处理
   - **numpy版本冲突**：
     - `gptqmodel` 需要 `numpy==2.2.6`
     - `depth-anything-3` 需要 `numpy<2`
     - 当前使用 `numpy<2`，如果遇到问题可以尝试：`pip install "numpy==2.2.6"`（可能影响depth-anything-3）

3. **transformers_stream_generator**（可选，但推荐）：
   - 如果遇到 `ImportError: ... transformers_stream_generator`
   - 尝试：`pip install transformers_stream_generator -i https://pypi.org/simple`
   - 如果安装后出现兼容性问题（如BeamSearchScorer错误），可以：
     - 升级transformers：`pip install --upgrade transformers`
     - 或者暂时跳过此依赖（代码会尝试使用mock，但可能影响功能）
   - 注意：代码已添加兼容性处理，即使没有此依赖也可能运行（但功能可能受限）

**注意**：Qwen-VL模型在首次加载时会自动检查依赖，如果缺少会提示安装。建议直接运行 `pip install -r requirements.txt` 安装所有依赖。

## 二、模型权重下载

### 1. Qwen-VL-Chat-Int4 模型

**自动下载**：首次运行时，transformers会自动从Hugging Face下载模型。

**手动下载**（可选）：
```bash
# 使用Hugging Face CLI
huggingface-cli download Qwen/Qwen-VL-Chat-Int4

# 或使用Python
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen-VL-Chat-Int4')"
```

**模型大小**：约 7-8 GB  
**存储位置**：`~/.cache/huggingface/hub/`

### 2. GroundingDINO 模型

**自动下载**：首次运行时自动下载。

**手动下载**（可选）：
```bash
python -c "from transformers import AutoModelForZeroShotObjectDetection; AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')"
```

**模型大小**：约 1-2 GB

### 3. Sentence Transformers 模型

**自动下载**：首次运行时自动下载。

**手动下载**（可选）：
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
```

**模型大小**：约 90 MB

### 4. SAM2 模型权重和代码

**重要**：SAM2需要手动下载模型权重和代码库。

#### 4.1 克隆SAM2代码库

**必须先克隆SAM2代码库**，否则无法导入SAM2模块：

```bash
# 如果thirdparty/grounded_sam_2目录已存在但只有checkpoints，需要先备份
cd thirdparty
if [ -d "grounded_sam_2" ] && [ ! -f "grounded_sam_2/setup.py" ]; then
    # 备份checkpoints目录
    if [ -d "grounded_sam_2/checkpoints" ]; then
        mv grounded_sam_2/checkpoints /tmp/sam2_checkpoints_backup
    fi
    rm -rf grounded_sam_2
fi

# 克隆SAM2代码库（使用Grounded-SAM-2仓库，包含SAM2和GroundingDINO集成）
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git grounded_sam_2

# 或者如果只需要SAM2，可以从官方仓库克隆
# git clone https://github.com/facebookresearch/segment-anything-2.git grounded_sam_2
```

**注意**：根据代码中的导入路径 `from thirdparty.grounded_sam_2.sam2.build_sam import ...`，应该使用 `Grounded-SAM-2` 仓库，因为它包含了正确的目录结构。

#### 4.2 下载SAM2模型权重

```bash
# 创建checkpoints目录（如果不存在）
mkdir -p thirdparty/grounded_sam_2/checkpoints

# 下载SAM2.1 Hiera Large模型权重
cd thirdparty/grounded_sam_2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

**模型大小**：约 856 MB

#### 4.3 安装SAM2

```bash
cd thirdparty/grounded_sam_2
pip install -e .
```

**注意**：如果安装失败，可能需要安装额外的依赖，如 `fvcore`、`iopath` 等。

## 三、SAM2第三方库安装

### 方法1：如果thirdparty/grounded_sam_2是完整的仓库

```bash
cd thirdparty/grounded_sam_2
pip install -e .
```

### 方法2：如果只是子模块，需要安装依赖

SAM2通常需要以下依赖：
- `torch` (已安装)
- `torchvision` (已安装)
- `fvcore`
- `iopath`
- `iopath`
- `detectron2` (可选，用于某些功能)

```bash
pip install fvcore iopath
```

## 四、验证完整安装

创建测试脚本 `test_imports.py`：

```python
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
```

运行测试：
```bash
python test_imports.py
```

## 五、常见问题

### 1. decord安装失败

**问题**：`pip install decord` 可能失败，因为需要编译。

**解决方案**：
```bash
# 使用conda安装（推荐）
conda install -c conda-forge decord

# 或从源码安装
pip install git+https://github.com/dmlc/decord
```

### 2. SAM2导入错误

**问题**：`ModuleNotFoundError: No module named 'thirdparty.grounded_sam_2'`

**解决方案**：
- 确保 `thirdparty/grounded_sam_2` 目录存在
- 如果是从git子模块，运行：`git submodule update --init --recursive`
- 或者手动安装SAM2：`cd thirdparty/grounded_sam_2 && pip install -e .`

### 3. 模型下载慢

**问题**：从Hugging Face下载模型很慢。

**解决方案**：
- 使用镜像站点（如果在中国）
- 设置环境变量：`export HF_ENDPOINT=https://hf-mirror.com`
- 或手动下载后放到 `~/.cache/huggingface/hub/` 目录

### 4. CUDA内存不足

**问题**：运行Qwen-VL时显存不足。

**解决方案**：
- 使用更小的模型（如果可用）
- 减少batch size
- 使用CPU模式（会很慢）：`device="cpu"`

## 六、快速安装脚本

创建 `install_dependencies.sh`：

```bash
#!/bin/bash
set -e

echo "=== 安装Python依赖 ==="
pip install -r requirements.txt

echo ""
echo "=== 下载SAM2模型权重 ==="
mkdir -p thirdparty/grounded_sam_2/checkpoints
if [ ! -f "thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "下载SAM2权重..."
    wget -P thirdparty/grounded_sam_2/checkpoints \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
else
    echo "SAM2权重已存在，跳过下载"
fi

echo ""
echo "=== 安装SAM2（如果需要） ==="
if [ -f "thirdparty/grounded_sam_2/setup.py" ]; then
    cd thirdparty/grounded_sam_2
    pip install -e .
    cd ../..
fi

echo ""
echo "=== 验证安装 ==="
python test_imports.py

echo ""
echo "安装完成！"
```

运行：
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

## 七、总结

**必需的Python包**：
- ✅ transformers (已安装)
- ✅ supervision (需要安装)
- ✅ decord (需要安装)
- ✅ scikit-learn (已安装)

**必需的模型权重**：
- ✅ Qwen-VL (自动下载)
- ✅ GroundingDINO (自动下载)
- ✅ Sentence Transformers (自动下载)
- ⚠️ SAM2 (需要手动下载)

**总存储空间需求**：约 10-15 GB（包括所有模型权重）

