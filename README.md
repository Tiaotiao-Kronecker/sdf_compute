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

程序会生成**两套结果**：原始深度图模式（`raw/`）和滤波后深度图模式（`filtered/`）。

### 输出目录结构

```
output/0/
├── raw/                  # 原始深度图模式（未滤波）
│   ├── depths/          # 原始深度图可视化
│   ├── frames/          # RGB帧
│   ├── sdf_npy/         # 原始SDF值
│   ├── sdf_vis/         # 原始SDF可视化
│   ├── sdf_exp_npy/     # 指数变换SDF值
│   ├── sdf_exp_vis/     # 指数变换SDF可视化
│   ├── pixel_details/   # 像素详细信息（如果使用--track_pixels）
│   ├── rgb_marked/      # 标记后的RGB图像（如果使用--track_pixels）
│   ├── depth.gif        # 原始深度图GIF
│   ├── sdf_vis.gif      # SDF可视化GIF
│   ├── sdf_exp_vis.gif  # 指数变换SDF可视化GIF
│   └── pointcloud.ply   # 合并的点云
└── filtered/            # 滤波后深度图模式（自适应滤波，减少闪烁和残影）
    ├── depths/          # 滤波后深度图可视化
    ├── frames/          # RGB帧
    ├── sdf_npy/         # 原始SDF值（基于滤波后深度图）
    ├── sdf_vis/         # 原始SDF可视化
    ├── sdf_exp_npy/     # 指数变换SDF值
    ├── sdf_exp_vis/     # 指数变换SDF可视化
    ├── pixel_details/   # 像素详细信息（如果使用--track_pixels）
    ├── rgb_marked/      # 标记后的RGB图像（如果使用--track_pixels）
    ├── depth.gif        # 滤波后深度图GIF
    ├── sdf_vis.gif      # SDF可视化GIF
    ├── sdf_exp_vis.gif  # 指数变换SDF可视化GIF
    └── pointcloud.ply   # 合并的点云
```

## 参数说明

- `--input_dir`: 输入目录（包含rgb.mp4和frame_*/目录）
- `--output_dir`: 输出目录
- `--model_name`: Depth-Anything-3模型名称（默认：depth-anything/DA3-LARGE）
- `--device`: 设备（cuda/cpu，默认：cuda）
- `--max_frames`: 最大处理帧数（None表示全部）
- `--track_pixels`: 要记录的像素坐标，格式：'x1,y1;x2,y2'（可选）
- `--video_fps`: 生成视频的帧率（默认10.0）

## 双模式处理说明

程序会自动生成两套结果，便于对比：

### 1. 原始深度图模式（`raw/`）
- 使用Depth-Anything-3直接生成的深度图，不做任何滤波处理
- 保留原始深度值，适合分析模型输出
- 可能包含深度闪烁和噪声

### 2. 滤波后深度图模式（`filtered/`）
- 使用自适应滤波处理深度图
- **静态区域**：应用5帧窗口的时间加权平均滤波（减少闪烁）
- **运动区域**：只应用3x3空间中值滤波（避免残影）
- 既减少闪烁，又避免移动物体产生残影

**自适应滤波原理**：
- 通过帧差检测运动区域（阈值：深度值变化>5%）
- 静态区域应用强时间滤波（5帧加权平均）
- 运动区域只应用空间滤波（避免残影）

## 指数变换说明

指数变换用于增强SDF可视化效果，突出近距离物体：

**变换公式**：`exp_sdf = 1 - exp(-k * normalized_sdf)`

- 将SDF值归一化到[0,1]
- 应用指数变换，使接近0的部分量程放大，接近1的部分压缩
- 默认参数k=5.0，可以通过修改代码中的`apply_exponential_transform`函数调用调整
- 效果：近距离物体的SDF值会被放大显示，远距离物体被压缩

## 像素跟踪功能

### 功能说明

当使用`--track_pixels`参数时，程序会记录指定像素的详细SDF计算信息：

1. **像素详细信息**：保存在`pixel_details/`目录
   - 每帧的详细信息：`pixel_details_XXXX.json`
   - 汇总信息：`pixel_details_summary.json`
   - 包含：像素坐标、所属物体ID、SDF值、到各物体的距离、最近点坐标等

2. **RGB标记可视化**：保存在`rgb_marked/`目录
   - **追踪像素标记**：用红色方块标记（半透明填充 + 边框）
   - **最近点标记**：用不同颜色的圆圈标记每个物体距离最近的像素点
     - 支持最多20种不同颜色，避免颜色重复
     - 颜色包括：绿色、蓝色、黄色、洋红色、青色、棕色、橙色等
   - 生成`rgb_marked_video.mp4`视频和`rgb_marked.gif`动画

### 像素坐标格式

支持多种格式：
- `"100,200"` - 单个像素
- `"100,200;300,400"` - 多个像素（分号分隔）
- `"100,200 300,400"` - 多个像素（空格分隔）

### 像素详细信息内容

每个像素的详细信息包含：
- `pixel_coords`: 像素坐标 (x, y)
- `object_id`: 所属物体ID
- `object_name`: 物体名称（如果有labels.txt）
- `depth_value`: 深度值
- `rgb_value`: RGB颜色值
- `sdf_value`: SDF值
- `distances_to_objects`: 到各个物体的距离和最近点信息
- `final_sdf_value`: 最终SDF值

## 相机参数说明

### 内参（Intrinsics）

- Depth-Anything-3会自动估计每帧的相机内参
- 程序统一使用第一帧的估计内参（假设相机固定，内参不变）
- 内参包括：焦距（fx, fy）和主点（cx, cy）

### 外参（Extrinsics）

- Depth-Anything-3会自动估计每帧的相机外参（世界到相机坐标系变换）
- 外参用于将3D点从相机坐标系转换到世界坐标系

### 点云生成

使用深度图、内参和外参生成3D点云：
- 深度图 → 相机坐标系3D点
- 相机坐标系 → 世界坐标系3D点（使用外参）
- 所有帧的点云合并保存为`pointcloud.ply`文件

