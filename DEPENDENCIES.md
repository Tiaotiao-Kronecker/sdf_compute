# 完整依赖列表

本文档列出 `brdige_dataset_process_depth.py` 所需的所有依赖。

## 快速安装

```bash
pip install -r requirements.txt
```

## 依赖分类

### 1. 基础依赖

| 包名 | 用途 | 必需 |
|------|------|------|
| torch>=2.0 | PyTorch深度学习框架 | ✓ |
| torchvision | 计算机视觉工具 | ✓ |
| numpy | 数值计算 | ✓ |
| opencv-python | 图像处理 | ✓ |
| Pillow | 图像处理 | ✓ |
| tqdm | 进度条 | ✓ |
| scipy | 科学计算 | ✓ |
| scikit-learn | 机器学习（KMeans, PCA） | ✓ |

### 2. Qwen-VL-Chat-Int4 模型依赖

| 包名 | 用途 | 必需 |
|------|------|------|
| transformers>=4.30.0 | Hugging Face transformers库 | ✓ |
| accelerate | 设备管理和device_map支持 | ✓ |
| optimum | GPTQ量化模型加载 | ✓ |
| auto-gptq | GPTQ量化支持 | ✓ |
| sentencepiece | Tokenizer依赖 | ✓ |
| tiktoken | 快速tokenization | ✓ |
| transformers_stream_generator | 流式生成（可选） | ⚠ |

**注意**：Qwen-VL-Chat-Int4 使用GPTQ 4-bit量化，需要 `optimum` 和 `auto-gptq`。

**版本兼容性修复**：
- `optimum 2.1.0` 和 `auto-gptq 0.7.1` 之间存在 `QuantizeConfig` 命名不匹配问题
- 代码已自动修复：将 `auto_gptq.BaseQuantizeConfig` 映射为 `QuantizeConfig`
- 如果遇到 `NameError: name 'QuantizeConfig' is not defined`，代码会自动处理

### 3. GroundingDINO 模型依赖

| 包名 | 用途 | 必需 |
|------|------|------|
| transformers | 已包含在Qwen-VL依赖中 | ✓ |

### 4. Sentence Transformers 依赖

| 包名 | 用途 | 必需 |
|------|------|------|
| transformers | 已包含在Qwen-VL依赖中 | ✓ |

### 5. 其他依赖

| 包名 | 用途 | 必需 |
|------|------|------|
| supervision>=0.18.0 | 标注和可视化 | ✓ |
| decord>=0.6.0 | 视频读取 | ✓ |

### 6. SAM2 依赖

SAM2需要单独的代码库和模型权重，见 `INSTALLATION.md`。

## 依赖检查

运行依赖检查脚本：

```bash
python test_imports.py
```

## 常见问题

### 1. Qwen-VL依赖问题

**问题1**: `ImportError: ... tiktoken`
```bash
pip install tiktoken
```

**问题2**: `ValueError: ... requires accelerate`
```bash
pip install accelerate
```

**问题3**: `ImportError: Loading a GPTQ quantized model requires optimum`
```bash
pip install optimum auto-gptq
```

**问题4**: `ImportError: ... transformers_stream_generator`
- 这是可选依赖，代码已处理兼容性
- 如果需要流式生成功能，可以尝试安装：
```bash
pip install transformers_stream_generator -i https://pypi.org/simple
```

### 2. 安装顺序建议

1. 先安装基础依赖：`pip install torch torchvision numpy opencv-python Pillow tqdm scipy scikit-learn`
2. 再安装transformers相关：`pip install transformers accelerate`
3. 然后安装量化支持：`pip install optimum auto-gptq`
4. 最后安装其他：`pip install supervision decord sentencepiece tiktoken`

### 3. 验证安装

```bash
# 检查所有依赖
python test_imports.py

# 测试Qwen-VL导入
python -c "from transformers import AutoModelForCausalLM; print('✓ Qwen-VL依赖OK')"

# 测试GPTQ支持
python -c "from optimum.gptq import GPTQQuantizer; print('✓ GPTQ支持OK')"
```

## 依赖大小估算

- transformers: ~50 MB
- optimum + auto-gptq: ~20 MB
- accelerate: ~5 MB
- 其他: ~10 MB
- **总计**: ~85 MB（不包括模型权重）

模型权重（首次下载）：
- Qwen-VL-Chat-Int4: ~7-8 GB
- GroundingDINO: ~1-2 GB
- Sentence Transformers: ~90 MB
- SAM2: ~856 MB

## 更新日志

- 2025-01-13: 添加 `optimum` 和 `auto-gptq` 支持GPTQ量化模型
- 2025-01-13: 添加 `accelerate` 支持device_map
- 2025-01-13: 添加 `tiktoken` 支持快速tokenization

