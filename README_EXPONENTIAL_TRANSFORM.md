# 指数变换SDF和视频生成功能

## 功能说明

### 1. 指数变换SDF

对原始SDF值应用指数变换，将接近0的部分量程放大，接近1的部分压缩。

**变换公式：**
```
normalized_sdf = (sdf - sdf_min) / (sdf_max - sdf_min)  # 归一化到[0,1]
exp_sdf = 1 - exp(-k * normalized_sdf)  # 指数变换
```

**参数：**
- `k`: 指数变换参数（默认5.0）
  - k越大，变换越强
  - k=5.0时，接近0的值会被显著放大，接近1的值会被压缩

**效果：**
- 接近0的SDF值（距离其他物体很近）→ 被放大，更容易观察
- 接近1的SDF值（距离其他物体很远）→ 被压缩，减少视觉差异

### 2. 视频生成

自动生成两个视频文件：
- `sdf_video.mp4`: 原始SDF值的可视化视频
- `sdf_exp_video.mp4`: 指数变换SDF值的可视化视频

**视频参数：**
- 帧率：30 FPS
- 格式：MP4
- 编码：MP4V

## 输出文件结构

```
output/0/
├── sdf_npy/              # 原始SDF值 (sdf_0000.npy, ...)
├── sdf_vis/              # 原始SDF可视化 (sdf_0000.png, ...)
├── sdf_exp_npy/          # 指数变换SDF值 (sdf_exp_0000.npy, ...)
├── sdf_exp_vis/          # 指数变换SDF可视化 (sdf_exp_0000.png, ...)
├── sdf_video.mp4         # 原始SDF视频
└── sdf_exp_video.mp4     # 指数变换SDF视频
```

## 使用方法

功能已集成到主程序中，无需额外参数：

```bash
python process_sdf.py \
    --input_dir /home/user/projects/SDF/brige_post_process/test/0 \
    --output_dir output/0 \
    --model_name depth-anything/DA3-BASE \
    --device cuda \
    --max_frames 10
```

处理完成后会自动生成：
1. 所有帧的指数变换SDF值（.npy文件）
2. 所有帧的指数变换SDF可视化（.png文件）
3. 原始SDF视频（sdf_video.mp4）
4. 指数变换SDF视频（sdf_exp_video.mp4）

## 调整指数变换参数

如果需要调整指数变换的强度，可以修改 `process_sdf.py` 中的参数：

```python
# 在 process_single_episode 函数中，找到这一行：
sdf_exp = apply_exponential_transform(sdf_map, k=5.0)

# 修改k值：
# k=3.0  # 较弱的变换
# k=5.0  # 默认值
# k=10.0 # 较强的变换
```

## 技术细节

### 指数变换原理

指数变换 `1 - exp(-k * x)` 的特性：
- 当 x=0 时：exp_sdf = 0
- 当 x=1 时：exp_sdf = 1 - exp(-k)
- 当 k=5.0 时：exp_sdf(1) ≈ 0.993
- 变换曲线在x接近0时斜率较大（放大效果），在x接近1时斜率较小（压缩效果）

### 视频生成

使用OpenCV的VideoWriter生成MP4视频：
- 自动读取所有匹配的图像文件
- 按文件名排序确保帧顺序正确
- 自动处理图像尺寸不一致的情况

