# 相机内外参数的获取方式

## 概述

在当前的SDF处理流程中，相机内外参数（intrinsics 和 extrinsics）是从 **Depth-Anything-3 模型的推理结果**中获取的。

## Depth-Anything-3 如何获取相机参数？

Depth-Anything-3 **可以自动估计相机内外参数**，它提供了两种估计方式：

### 方式1：相机解码器估计（默认方式）

**代码位置**：`Depth-Anything-3/src/depth_anything_3/model/da3.py` 的 `_process_camera_estimation` 方法

**工作原理**：
1. 模型从输入图像中提取特征
2. 使用**相机解码器（cam_dec）**从特征中估计相机参数
3. 将姿态编码转换为外参（extrinsics）和内参（intrinsics）

**代码片段**：
```python
def _process_camera_estimation(
    self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Process camera pose estimation if camera decoder is available."""
    if self.cam_dec is not None:
        pose_enc = self.cam_dec(feats[-1][1])  # 从特征中估计姿态编码
        
        # 将姿态编码转换为外参和内参
        c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
        output.extrinsics = affine_inverse(c2w)  # 转换为w2c格式
        output.intrinsics = ixt
```

**特点**：
- 这是**默认方式**（当 `use_ray_pose=False` 时）
- 不需要提供相机参数，模型会自动估计
- 适用于单张图像或图像序列

### 方式2：射线姿态估计（可选方式）

**代码位置**：`Depth-Anything-3/src/depth_anything_3/model/da3.py` 的 `_process_ray_pose_estimation` 方法

**工作原理**：
1. 模型预测每个像素的射线方向（ray）和置信度（ray_conf）
2. 使用 `get_extrinsic_from_camray` 函数从射线信息中估计外参
3. 从射线信息中估计焦距（focal lengths）和主点（principal points）
4. 构建内参矩阵

**代码片段**：
```python
def _process_ray_pose_estimation(
    self, output: Dict[str, torch.Tensor], height: int, width: int
) -> Dict[str, torch.Tensor]:
    """Process ray pose estimation if ray pose decoder is available."""
    if "ray" in output and "ray_conf" in output:
        # 从射线信息中估计外参、焦距和主点
        pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
            output.ray,
            output.ray_conf,
            output.ray.shape[-3],
            output.ray.shape[-2],
        )
        
        # 构建内参矩阵
        pred_intrinsic = torch.eye(3, 3)[None, None].repeat(...)
        pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
        pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
        pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
        pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
        
        output.extrinsics = pred_extrinsic
        output.intrinsics = pred_intrinsic
```

**特点**：
- 需要设置 `use_ray_pose=True` 才会使用
- 基于射线方向估计，可能在某些场景下更准确
- 适用于需要更精确姿态估计的场景

## 在我们的代码中如何使用

在 `process_sdf.py` 中，我们直接使用 Depth-Anything-3 估计的相机参数：

```python
# 运行推理
prediction = model.inference(
    [frame_path],
    export_format="mini_npz",
    export_dir=None,
)

# 获取相机参数（模型自动估计的）
intrinsics = prediction.intrinsics[0]  # [3, 3] 内参矩阵
extrinsics = prediction.extrinsics[0]  # [3, 4] 外参矩阵 (w2c格式)
```

## 相机参数的含义

### 内参（Intrinsics）- [3, 3] 矩阵

内参矩阵 K 描述相机的光学特性：

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

- **fx, fy**：焦距（focal lengths），单位通常是像素
- **cx, cy**：主点（principal point），图像中心或光轴与图像平面的交点

### 外参（Extrinsics）- [3, 4] 矩阵（w2c格式）

外参矩阵描述相机在世界坐标系中的位置和方向：

```
w2c = [[r11, r12, r13, tx],
       [r21, r22, r23, ty],
       [r31, r32, r33, tz]]
```

- **R**：旋转矩阵（前3x3部分）
- **t**：平移向量（最后一列）
- **w2c**：world-to-camera 变换，将世界坐标转换为相机坐标

## 是否可以手动提供相机参数？

**可以！** Depth-Anything-3 的 `inference` 方法支持手动提供相机参数：

```python
prediction = model.inference(
    image_paths,
    extrinsics=your_extrinsics,  # 可选：手动提供外参
    intrinsics=your_intrinsics,  # 可选：手动提供内参
    ...
)
```

**使用场景**：
- 如果你有通过相机标定获得的精确参数
- 如果你有从其他系统（如SLAM、NeRF等）获得的相机参数
- 如果你需要确保相机参数的一致性

**注意事项**：
- 如果提供了相机参数，模型可能不会重新估计，而是直接使用你提供的参数
- 参数格式需要匹配：`extrinsics` 为 `[B, N, 4, 4]` 或 `[B, N, 3, 4]`，`intrinsics` 为 `[B, N, 3, 3]`

## 估计精度如何？

Depth-Anything-3 的相机参数估计是基于**深度学习模型**的，它：

1. **在训练时学习**了从图像特征推断相机参数的能力
2. **适用于大多数常见场景**，但可能不如专业相机标定精确
3. **对于SDF计算**，通常精度足够，因为：
   - SDF计算主要关注相对距离
   - 深度估计和相机参数估计是联合优化的
   - 误差会在一定程度上相互抵消

## 总结

- ✅ **Depth-Anything-3 可以自动估计相机内外参数**
- ✅ **默认使用相机解码器估计**（不需要额外设置）
- ✅ **也可以手动提供相机参数**（如果有更精确的标定结果）
- ✅ **估计的精度对于SDF计算通常足够**

在我们的项目中，我们直接使用模型估计的参数，这对于大多数应用场景是合适的。如果你有更精确的相机标定参数，也可以手动提供。

