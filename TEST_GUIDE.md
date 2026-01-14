# 测试指南

## 快速开始

### 单个Case测试

测试从bridge数据集处理到SDF计算的完整流程：

```bash
python test_single_case.py \
    --input_dir bridge_depth \
    --episode_id 00000 \
    --stream_id 0 \
    --output_base output/test \
    --device cuda:0 \
    --max_frames 10
```

### 参数说明

- `--input_dir`: bridge数据集输入目录（包含episode目录，如00000/）
- `--episode_id`: 要测试的episode ID（例如：00000）
- `--stream_id`: 要处理的stream ID（默认0，对应images0）
- `--output_base`: 输出基础目录（默认：output/test）
- `--device`: 设备（默认：cuda:0）
- `--max_frames`: 最大处理帧数（默认10，用于快速测试）
- `--skip_dataset_process`: 跳过数据集处理步骤（如果已经处理过）
- `--skip_sdf`: 跳过SDF计算步骤

### 分步测试

#### 步骤1：数据集处理

```bash
python brdige_dataset_process_depth.py \
    --input_dir bridge_depth \
    --output_dir output/test/dataset_processed \
    --max_videos 1 \
    --device cuda:0
```

这会生成：
- `output/test/dataset_processed/{episode_id}/images{stream_id}/rgb.mp4`
- `output/test/dataset_processed/{episode_id}/images{stream_id}/frame_XXXX.npz` (分割掩码)
- `output/test/dataset_processed/labels.txt` (标签文件)

**注意**：episode_id 会保留原始格式（如 `00000`），不会转换为整数格式。

#### 步骤2：SDF计算

**重要**：使用实际的输出路径，episode_id应该与步骤1的输出一致。

```bash
# 如果episode_id是00000
python process_sdf.py \
    --input_dir output/test/dataset_processed/00000/images0 \
    --output_dir output/test/sdf_processed/00000 \
    --device cuda \
    --max_frames 10

# 如果episode_id是0（旧版本代码生成）
python process_sdf.py \
    --input_dir output/test/dataset_processed/0/images0 \
    --output_dir output/test/sdf_processed/0 \
    --device cuda \
    --max_frames 10
```

这会生成：
- `output/test/sdf_processed/00000/raw/` (原始深度图模式)
- `output/test/sdf_processed/00000/filtered/` (滤波后深度图模式)

## 输出结构

```
output/test/
├── dataset_processed/          # 数据集处理输出
│   ├── 00000/
│   │   └── images0/
│   │       ├── rgb.mp4         # RGB视频
│   │       ├── frame_0000.npz # 分割掩码
│   │       ├── frame_0001.npz
│   │       ├── labels.txt     # 标签文件（复制）
│   │       └── ...
│   └── labels.txt              # 全局标签文件
└── sdf_processed/              # SDF计算输出
    └── 00000/
        ├── raw/                # 原始深度图模式
        │   ├── depths/
        │   ├── sdf_npy/
        │   ├── sdf_vis/
        │   └── ...
        └── filtered/            # 滤波后深度图模式
            ├── depths/
            ├── sdf_npy/
            ├── sdf_vis/
            └── ...
```

## 常见问题

### 1. SAM2导入失败

如果遇到 `ModuleNotFoundError: No module named 'thirdparty.grounded_sam_2.sam2'`，请运行：

```bash
./install_dependencies.sh
```

### 2. 模型下载慢

首次运行会自动下载模型，如果下载慢，可以设置Hugging Face镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 显存不足

如果显存不足，可以：
- 减少 `--max_frames` 数量
- 使用更小的模型（如 `--model_name depth-anything/DA3-BASE`）
- 使用CPU模式（会很慢）：`--device cpu`

### 4. 跳过已完成的步骤

如果数据集处理已完成，可以跳过：

```bash
python test_single_case.py \
    --input_dir bridge_depth \
    --episode_id 00000 \
    --skip_dataset_process \
    --max_frames 10
```

## 验证输出

### 检查数据集处理输出

```bash
# 检查RGB视频
ls -lh output/test/dataset_processed/00000/images0/rgb.mp4

# 检查分割掩码
ls output/test/dataset_processed/00000/images0/frame_*.npz

# 检查标签文件
cat output/test/dataset_processed/labels.txt
```

### 检查SDF计算输出

```bash
# 检查原始模式输出
ls output/test/sdf_processed/00000/raw/sdf_vis/

# 检查滤波模式输出
ls output/test/sdf_processed/00000/filtered/sdf_vis/

# 查看GIF动画
ls output/test/sdf_processed/00000/raw/*.gif
```

