# 像素SDF计算过程记录功能

## 功能说明

本功能允许你指定几个像素坐标，完整记录这些像素在SDF计算过程中的所有中间量，包括：

1. **像素坐标** (x, y)
2. **RGB值** (r, g, b)
3. **物体ID和名称** (从annotated_frame_index.npy和labels.txt获取)
4. **深度值** (depth)
5. **相机内外参数** (intrinsics和extrinsics)
6. **坐标变换过程** (从像素坐标到3D世界坐标的完整计算步骤)
7. **3D世界坐标** (x, y, z)
8. **到各个其他物体的距离** (每个物体的最近距离，包括最近点的3D坐标和像素坐标)
9. **最终SDF值** (到所有其他物体的最近距离)

## 使用方法

### 命令行参数

添加 `--track_pixels` 参数来指定要记录的像素坐标：

```bash
python process_sdf.py \
    --input_dir /path/to/input \
    --output_dir /path/to/output \
    --track_pixels "x1,y1;x2,y2;x3,y3"
```

### 像素坐标格式

支持多种格式：
- 单个像素：`"100,200"`
- 多个像素（分号分隔）：`"100,200;300,400;500,600"`
- 多个像素（空格分隔）：`"100,200 300,400 500,600"`

### 示例

```bash
# 记录两个像素
python process_sdf.py \
    --input_dir ../brige_post_process/test/0 \
    --output_dir output/0 \
    --model_name depth-anything/DA3-BASE \
    --device cuda \
    --max_frames 5 \
    --track_pixels "100,200;300,400"
```

## 输出结果

### 文件结构

```
output/0/
├── pixel_details/
│   ├── pixel_details_0000.json  # 第0帧的像素详细信息
│   ├── pixel_details_0001.json  # 第1帧的像素详细信息
│   ├── ...
│   └── pixel_details_summary.json  # 所有帧的汇总
└── rgb_marked/                    # 标记后的RGB图像
    ├── rgb_marked_0000.png        # 第0帧的标记RGB图像
    ├── rgb_marked_0001.png        # 第1帧的标记RGB图像
    ├── ...
    ├── rgb_marked_video.mp4        # 标记RGB视频
    └── rgb_marked.gif              # 标记RGB GIF动画
```

### JSON格式示例

每个 `pixel_details_XXXX.json` 文件包含：

```json
{
  "pixel_100_200": {
    "pixel_coords": {
      "x": 100,
      "y": 200
    },
    "rgb_value": {
      "r": 128,
      "g": 64,
      "b": 32
    },
    "object_id": 5,
    "object_name": "car",
    "depth_value": 12.345,
    "camera_parameters": {
      "intrinsics": {
        "matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "fx": 500.0,
        "fy": 500.0,
        "cx": 320.0,
        "cy": 240.0,
        "format": "3x3 camera intrinsic matrix K",
        "description": "fx, fy: focal lengths; cx, cy: principal point"
      },
      "extrinsics": {
        "w2c": [[...], [...], [...]],
        "c2w": [[...], [...], [...], [...]],
        "format": "w2c: world-to-camera [3x4], c2w: camera-to-world [4x4]",
        "description": "w2c transforms world coordinates to camera coordinates"
      }
    },
    "coordinate_transformation": {
      "step_1_pixel_coords": {
        "description": "像素坐标（齐次坐标）",
        "value": [100, 200, 1],
        "format": "[u, v, 1]"
      },
      "step_2_camera_ray": {
        "description": "反投影到相机坐标系（归一化射线方向）",
        "formula": "K_inv @ [u, v, 1]",
        "K_inv": [[...], [...], [...]],
        "value": [0.1, 0.2, 1.0],
        "format": "[X_c/Z_c, Y_c/Z_c, 1] (归一化)"
      },
      "step_3_camera_coords": {
        "description": "相机坐标系3D点",
        "formula": "camera_ray * depth",
        "depth": 12.345,
        "value": [1.23, 2.47, 12.345],
        "format": "[X_c, Y_c, Z_c] in camera frame"
      },
      "step_4_world_coords": {
        "description": "世界坐标系3D点",
        "formula": "c2w @ [X_c, Y_c, Z_c, 1]",
        "c2w_matrix": [[...], [...], [...], [...]],
        "value": [1.23, 4.56, 7.89],
        "format": "[X_w, Y_w, Z_w] in world frame"
      },
      "verification": {
        "description": "验证：计算值应与points_3d中的值一致",
        "calculated": [1.23, 4.56, 7.89],
        "from_points_3d": [1.23, 4.56, 7.89],
        "difference": [0.0, 0.0, 0.0],
        "max_error": 0.0
      }
    },
    "3d_world_coords": {
      "x": 1.23,
      "y": 4.56,
      "z": 7.89
    },
    "distances_to_objects": {
      "person": {
        "object_id": 3,
        "distance": 2.5,
        "nearest_point_3d": {
          "x": 1.23,
          "y": 4.56,
          "z": 7.89
        },
        "nearest_pixel_coords": {
          "x": 150,
          "y": 250
        }
      },
      "building": {
        "object_id": 1,
        "distance": 15.8,
        "nearest_point_3d": {
          "x": 2.34,
          "y": 5.67,
          "z": 8.90
        },
        "nearest_pixel_coords": {
          "x": 200,
          "y": 300
        }
      }
    },
    "final_sdf_value": 2.5
  }
}
```

`pixel_details_summary.json` 包含所有帧的汇总：

```json
{
  "tracked_pixels": [
    {"x": 100, "y": 200},
    {"x": 300, "y": 400}
  ],
  "total_frames": 5,
  "frame_details": {
    "frame_0000": { ... },
    "frame_0001": { ... },
    ...
  }
}
```

## 物体标签映射

如果输入目录中有 `labels.txt` 文件，格式如下：

```
1:building
3:person
5:car
```

脚本会自动加载并使用这些标签名称。如果没有标签文件，将使用 `object_{id}` 作为默认名称。

## distances_to_objects 字段说明

`distances_to_objects` 记录了当前像素到每个其他物体的最近距离信息，包括：

- **object_id**: 物体ID
- **distance**: 到该物体的最近距离（3D欧氏距离）
- **nearest_point_3d**: 最近点的3D世界坐标 `{x, y, z}`
- **nearest_pixel_coords**: 最近点在图像中的像素坐标 `{x, y}`

**注意**：
- 只记录到"其他物体"的距离，不包括像素自身所属的物体
- `nearest_point_3d` 是距离当前像素3D点最近的其他物体上的点的3D坐标
- `nearest_pixel_coords` 是该最近点在图像中的像素位置

## 坐标变换过程说明

从像素坐标到3D世界坐标的计算过程分为4个步骤：

1. **像素坐标（齐次坐标）**: `[u, v, 1]`
   - u, v 是像素在图像中的列和行坐标

2. **反投影到相机坐标系（归一化射线方向）**: `K_inv @ [u, v, 1]`
   - K_inv 是相机内参矩阵的逆矩阵
   - 得到归一化的射线方向 `[X_c/Z_c, Y_c/Z_c, 1]`

3. **相机坐标系3D点**: `camera_ray * depth`
   - 将归一化射线方向乘以深度值
   - 得到相机坐标系中的3D点 `[X_c, Y_c, Z_c]`

4. **世界坐标系3D点**: `c2w @ [X_c, Y_c, Z_c, 1]`
   - c2w 是相机到世界的变换矩阵（从w2c计算得到）
   - 得到世界坐标系中的3D点 `[X_w, Y_w, Z_w]`

所有中间计算步骤和矩阵都会记录在JSON文件中，方便验证和调试。

## RGB标记可视化

当使用像素跟踪功能时，程序会自动生成标记后的RGB图像和视频，用于可视化追踪像素和最近点：

### 标记说明

- **追踪像素标记**：
  - 红色方块（半透明填充）
  - 3像素宽的红色边框
  - 标记大小为 5×5 像素（可调整）

- **最近点像素标记**：
  - 不同颜色的圆圈（每个物体使用不同颜色）
  - 半透明填充
  - 3像素宽的边框
  - 颜色方案：
    - 绿色：第一个物体的最近点
    - 蓝色：第二个物体的最近点
    - 黄色：第三个物体的最近点
    - 洋红色：第四个物体的最近点
    - 青色：第五个物体的最近点
    - （如果物体超过5个，颜色会循环使用）

### 输出文件

- **标记RGB图像**：保存在 `rgb_marked/` 目录
  - 文件名格式：`rgb_marked_XXXX.png`
- **标记RGB视频**：`rgb_marked_video.mp4`
  - 使用与其他视频相同的帧率（可通过 `--video_fps` 调整）
- **标记RGB GIF**：`rgb_marked.gif`
  - 帧持续时间根据视频帧率自动计算

## 注意事项

1. **像素坐标范围**：确保指定的像素坐标在图像范围内（0 <= x < width, 0 <= y < height）
2. **性能影响**：记录像素详细信息会增加少量计算开销，但影响很小
3. **内存使用**：如果记录大量像素或处理很多帧，JSON文件可能会较大
4. **坐标系统**：像素坐标使用 (x, y) 格式，其中 x 是列（宽度方向），y 是行（高度方向）
5. **相机参数**：相机内外参数会自动从Depth-Anything-3的预测结果中获取并记录
6. **RGB标记**：只有在使用 `--track_pixels` 参数时才会生成标记RGB图像和视频

