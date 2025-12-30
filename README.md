# SDF Processor

基于Depth-Anything-3的SDF（Surface Distance Field，表面距离场）计算项目。

## 功能

1. **视频帧提取**: 从`rgb.mp4`视频文件中提取每一帧图像
2. **深度估计**: 使用Depth-Anything-3模型为每一帧图像生成深度图
3. **3D点云生成**: 将深度图与相机参数结合，生成每一帧的3D点云，并保存为PLY文件
4. **SDF计算**: 对于每一帧，计算每个像素的3D点到场景中其他物体表面的最近距离（Surface Distance Field）
5. **指数变换SDF**: 对SDF值应用指数变换，将接近0的部分量程放大，接近1的部分压缩
6. **SDF存储与可视化**: 
   - 将原始SDF值和指数变换SDF值保存为`.npy`文件
   - 将其可视化为灰度图保存为`.png`文件
7. **视频生成**: 生成原始SDF和指数变换SDF的视频文件
8. **GIF生成**: 为RGB、标记RGB、深度图、SDF可视化等生成GIF动画
9. **像素跟踪**（可选）: 记录指定像素的完整SDF计算过程
10. **RGB标记可视化**（可选）: 在RGB图像上标记追踪像素和最近点像素，生成标记图像和视频

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python process_sdf.py \
    --input_dir /home/user/projects/SDF/brige_post_process/test/0 \
    --output_dir output/0 \
    --model_name depth-anything/DA3-BASE \
    --device cuda \
    --max_frames 10
```

### 记录特定像素的详细信息

```bash
python process_sdf.py \
    --input_dir /home/user/projects/SDF/brige_post_process/test/0 \
    --output_dir output/0 \
    --model_name depth-anything/DA3-BASE \
    --device cuda \
    --max_frames 10 \
    --track_pixels "100,200;300,400"
```

## 输出结果

程序将在`output_dir`下创建以下子目录和文件：

```
output/0/
├── frames/              # 提取的视频帧 (frame_0000.png, ...)
├── depths/              # 深度图 (depth_0000.npy, depth_0000_vis.png, ...)
├── pointcloud.ply       # 合并的3D点云文件
├── sdf_npy/             # 原始SDF值 (sdf_0000.npy, ...)
├── sdf_vis/             # 原始SDF可视化灰度图 (sdf_0000.png, ...)
├── sdf_exp_npy/         # 指数变换SDF值 (sdf_exp_0000.npy, ...)
├── sdf_exp_vis/         # 指数变换SDF可视化灰度图 (sdf_exp_0000.png, ...)
├── sdf_video.mp4        # 原始SDF视频
├── sdf_exp_video.mp4    # 指数变换SDF视频
├── depth_video.mp4      # 深度图视频
├── rgb.gif              # RGB帧GIF动画
├── depth.gif            # 深度图GIF动画
├── sdf_vis.gif          # SDF可视化GIF动画
├── sdf_exp_vis.gif      # 指数变换SDF可视化GIF动画
├── pixel_details/        # 像素详细信息（如果使用--track_pixels）
│   ├── pixel_details_0000.json
│   └── pixel_details_summary.json
└── rgb_marked/           # 标记后的RGB图像（如果使用--track_pixels）
    ├── rgb_marked_0000.png
    ├── rgb_marked_video.mp4  # 标记RGB视频
    └── rgb_marked.gif         # 标记RGB GIF动画
```

## 参数说明

- `--input_dir`: 输入目录（包含rgb.mp4和frame_*/目录）
- `--output_dir`: 输出目录
- `--model_name`: Depth-Anything-3模型名称（默认：depth-anything/DA3-LARGE）
- `--device`: 设备（cuda/cpu，默认：cuda）
- `--max_frames`: 最大处理帧数（None表示全部）
- `--track_pixels`: 要记录的像素坐标，格式：'x1,y1;x2,y2'（可选）
- `--video_fps`: 生成视频的帧率（默认10.0）

## 指数变换说明

指数变换公式：`exp_sdf = 1 - exp(-k * normalized_sdf)`

- 将SDF值归一化到[0,1]
- 应用指数变换，使接近0的部分量程放大，接近1的部分压缩
- 默认参数k=5.0，可以通过修改代码中的`apply_exponential_transform`函数调用调整

## RGB标记可视化功能

当使用`--track_pixels`参数时，程序会生成标记后的RGB图像和视频：

- **追踪像素标记**: 用红色方块标记（半透明填充 + 3像素边框）
- **最近点标记**: 用不同颜色的圆圈标记每个物体距离最近的像素点
  - 绿色：第一个物体的最近点
  - 蓝色：第二个物体的最近点
  - 黄色：第三个物体的最近点
  - 洋红色：第四个物体的最近点
  - 青色：第五个物体的最近点
  - （如果物体超过5个，颜色会循环使用）

标记后的图像保存在`rgb_marked/`目录，并会生成`rgb_marked_video.mp4`视频和`rgb_marked.gif`动画。

## 详细文档

- `README_PIXEL_TRACKING.md` - 像素跟踪功能详细说明
- `QUICK_START_PIXEL_TRACKING.md` - 像素跟踪快速开始指南

