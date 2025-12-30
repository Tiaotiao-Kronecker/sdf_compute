# 像素SDF计算过程记录 - 快速开始

## 功能已完成 ✅

脚本已支持记录指定像素的完整SDF计算过程。

## 快速使用

### 1. 基本用法

记录一个或多个像素的详细信息：

```bash
cd ~/projects/SDF/sdf_processor
conda activate depth_anything_v3

python process_sdf.py \
    --input_dir /home/user/projects/SDF/brige_post_process/test/0 \
    --output_dir output/0 \
    --model_name depth-anything/DA3-BASE \
    --device cuda \
    --max_frames 5 \
    --track_pixels "100,200;300,400"
```

### 2. 像素坐标格式

- 单个像素：`"100,200"`
- 多个像素（分号）：`"100,200;300,400;500,600"`
- 多个像素（空格）：`"100,200 300,400 500,600"`

### 3. 查看结果

处理完成后，查看输出目录：

```bash
ls output/0/pixel_details/
# 输出：
# pixel_details_0000.json
# pixel_details_0001.json
# ...
# pixel_details_summary.json
```

查看汇总文件：

```bash
cat output/0/pixel_details/pixel_details_summary.json
```

## 记录的信息

每个像素的JSON文件包含：

1. ✅ **像素坐标** (x, y)
2. ✅ **RGB值** (r, g, b)
3. ✅ **物体ID和名称** (从annotated_frame_index.npy和labels.txt)
4. ✅ **深度值** (depth)
5. ✅ **3D世界坐标** (x, y, z)
6. ✅ **到各个其他物体的距离** (每个物体的最近距离)
7. ✅ **最终SDF值** (到所有其他物体的最近距离的最小值)

## 示例输出

```json
{
  "pixel_100_200": {
    "pixel_coords": {"x": 100, "y": 200},
    "rgb_value": {"r": 128, "g": 64, "b": 32},
    "object_id": 5,
    "object_name": "car",
    "depth_value": 12.345,
    "3d_world_coords": {"x": 1.23, "y": 4.56, "z": 7.89},
    "distances_to_objects": {
      "person": {"object_id": 3, "distance": 2.5},
      "building": {"object_id": 1, "distance": 15.8}
    },
    "final_sdf_value": 2.5
  }
}
```

## 注意事项

- 像素坐标范围：确保坐标在图像范围内
- 物体标签：如果有 `labels.txt` 文件，会自动加载物体名称
- 性能：记录功能对性能影响很小

## 完整文档

详细说明请查看：`README_PIXEL_TRACKING.md`

