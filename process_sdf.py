#!/usr/bin/env python3
"""
基于Depth-Anything-3的Surface Distance Field (SDF)计算
计算每个像素的3D点到其他物体表面的最近距离
"""
import argparse
import os
import sys
from pathlib import Path

# 修复PyTorch Intel库符号问题（必须在import torch之前设置）
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial import cKDTree
import torch
import json
from collections import defaultdict
from PIL import Image
import glob

# 添加Depth-Anything-3到路径
# 尝试多个可能的路径
da3_paths = [
    Path(__file__).parent.parent / "Depth-Anything-3",
    Path.home() / "projects" / "Depth-Anything-3",
    Path("/home/user/projects/Depth-Anything-3"),
]

DA3_AVAILABLE = False
DepthAnything3 = None
unproject_depth = None

# 先检查基础依赖
try:
    import torch
    import torchvision
except ImportError as e:
    print(f"错误: 缺少基础依赖: {e}")
    print("请先安装: pip install torch torchvision")
    sys.exit(1)

for da3_path in da3_paths:
    if da3_path.exists():
        src_path = da3_path / "src"
        if src_path.exists():
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            try:
                # 先测试基础模块导入
                import depth_anything_3
                from depth_anything_3.api import DepthAnything3
                from depth_anything_3.utils.geometry import unproject_depth
                DA3_AVAILABLE = True
                print(f"成功从 {da3_path} 导入Depth-Anything-3")
                break
            except ImportError as e:
                print(f"从 {da3_path} 导入失败: {e}")
                import traceback
                traceback.print_exc()
                continue

if not DA3_AVAILABLE:
    print("\n错误: 无法导入Depth-Anything-3")
    print("请运行诊断脚本获取详细信息:")
    print("  python diagnose.py")
    print("\n常见解决方案:")
    print("  1. 确保torch和torchvision已安装:")
    print("     pip install torch torchvision")
    print("  2. 安装Depth-Anything-3:")
    print("     cd ~/projects/Depth-Anything-3")
    print("     pip install -e .")
    print("  3. 如果使用conda环境，确保激活了正确的环境")
    sys.exit(1)


def extract_frames_from_video(video_path, output_dir, max_frames=None):
    """
    从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        max_frames: 最大提取帧数（None表示全部）
    
    Returns:
        提取的帧文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    frame_paths = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames is not None and frame_count >= max_frames:
            break
        
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_count += 1
    
    cap.release()
    print(f"从视频中提取了 {len(frame_paths)} 帧")
    return frame_paths


def generate_depth_and_pointcloud(model, frame_paths, output_dir, device):
    """
    使用Depth-Anything-3生成深度图和点云
    
    Args:
        model: DepthAnything3模型
        frame_paths: 帧文件路径列表
        output_dir: 输出目录
        device: 设备
    
    Returns:
        depth_maps: 深度图列表
        point_clouds: 点云列表（每帧的点云）
        intrinsics_list: 内参列表
        extrinsics_list: 外参列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 将模型移到设备
    model = model.to(device)
    
    depth_maps = []
    point_clouds = []
    intrinsics_list = []
    extrinsics_list = []
    
    print("正在生成深度图和点云...")
    for i, frame_path in enumerate(tqdm(frame_paths, desc="处理帧")):
        # 运行推理
        prediction = model.inference(
            [frame_path],
            export_format="mini_npz",
            export_dir=None,
        )
        
        # 获取深度图
        depth = prediction.depth[0]  # [H, W]
        depth_maps.append(depth)
        
        # 获取相机参数
        intrinsics = prediction.intrinsics[0]  # [3, 3]
        extrinsics = prediction.extrinsics[0]  # [3, 4] (w2c格式)
        intrinsics_list.append(intrinsics)
        extrinsics_list.append(extrinsics)
        
        # 生成点云
        H, W = depth.shape
        points_3d = depth_to_pointcloud(depth, intrinsics, extrinsics)
        point_clouds.append(points_3d)
        
        # 保存深度图
        depth_vis = (depth / depth.max() * 255).astype(np.uint8)
        depth_path = os.path.join(output_dir, f"depth_{i:04d}.png")
        cv2.imwrite(depth_path, depth_vis)
    
    # 保存合并的点云
    all_points = np.concatenate([pc.reshape(-1, 3) for pc in point_clouds], axis=0)
    all_colors = None  # 可以添加颜色信息
    
    # 保存为PLY文件
    save_pointcloud_ply(all_points, all_colors, os.path.join(output_dir, "pointcloud.ply"))
    
    return depth_maps, point_clouds, intrinsics_list, extrinsics_list


def depth_to_pointcloud(depth, intrinsics, extrinsics):
    """
    将深度图转换为3D点云
    
    Args:
        depth: [H, W] 深度图
        intrinsics: [3, 3] 内参矩阵
        extrinsics: [3, 4] 外参矩阵 (w2c格式)
    
    Returns:
        points_3d: [H, W, 3] 3D点云
    """
    H, W = depth.shape
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(u)
    pixel_coords = np.stack([u, v, ones], axis=-1)  # [H, W, 3]
    
    # 转换为齐次坐标
    pixel_coords_flat = pixel_coords.reshape(-1, 3).T  # [3, H*W]
    
    # 反投影到相机坐标系
    K_inv = np.linalg.inv(intrinsics)
    camera_coords = K_inv @ pixel_coords_flat  # [3, H*W]
    camera_coords = camera_coords * depth.reshape(-1)[None, :]  # [3, H*W]
    
    # 转换为世界坐标系
    # extrinsics是w2c，需要转换为c2w
    w2c = np.vstack([extrinsics, [0, 0, 0, 1]])  # [4, 4]
    c2w = np.linalg.inv(w2c)
    
    camera_coords_homo = np.vstack([camera_coords, np.ones((1, camera_coords.shape[1]))])  # [4, H*W]
    world_coords = (c2w @ camera_coords_homo)[:3].T  # [H*W, 3]
    
    points_3d = world_coords.reshape(H, W, 3)
    return points_3d


def save_pointcloud_ply(points, colors, output_path):
    """
    保存点云为PLY文件
    """
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
"""
    if colors is not None:
        header += """property uchar red
property uchar green
property uchar blue
"""
    header += "end_header\n"
    
    with open(output_path, 'w') as f:
        f.write(header)
        for i, point in enumerate(points):
            if colors is not None:
                f.write(f"{point[0]} {point[1]} {point[2]} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
            else:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    print(f"点云已保存到: {output_path}")


def load_annotated_frame_index(index_path):
    """
    加载annotated_frame_index.npy文件
    
    Returns:
        object_ids: [H, W] 每个像素对应的对象ID
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"找不到文件: {index_path}")
    
    object_ids = np.load(index_path)
    return object_ids


def load_object_labels(labels_path):
    """
    加载物体标签映射（如果有labels.txt文件）
    
    Returns:
        dict: {object_id: object_name}
    """
    label_map = {}
    if labels_path and os.path.exists(labels_path):
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            obj_id = int(parts[0].strip())
                            obj_name = parts[1].strip()
                            label_map[obj_id] = obj_name
        except Exception as e:
            print(f"警告: 无法加载标签文件 {labels_path}: {e}")
    return label_map


def compute_sdf_for_frame(points_3d, object_ids, frame_idx, output_dir, 
                          rgb_image=None, depth_map=None, 
                          tracked_pixels=None, label_map=None,
                          intrinsics=None, extrinsics=None):
    """
    计算单帧的Surface Distance Field (SDF)
    
    SDF值定义为：每个像素的3D点到其他物体（除去自身）的最近距离
    
    Args:
        points_3d: [H, W, 3] 3D点云
        object_ids: [H, W] 每个像素的对象ID
        frame_idx: 帧索引
        output_dir: 输出目录
        rgb_image: [H, W, 3] RGB图像（可选，用于记录像素RGB值）
        depth_map: [H, W] 深度图（可选，用于记录像素深度值）
        tracked_pixels: list of (x, y) tuples，要记录的像素坐标
        label_map: dict，物体ID到名称的映射
        intrinsics: [3, 3] 相机内参矩阵（可选，用于记录计算过程）
        extrinsics: [3, 4] 相机外参矩阵w2c格式（可选，用于记录计算过程）
    
    Returns:
        sdf_map: [H, W] SDF值
        pixel_details: dict，记录的像素详细信息（如果tracked_pixels不为None）
    """
    H, W = points_3d.shape[:2]
    sdf_map = np.full((H, W), np.inf, dtype=np.float32)
    
    # 获取所有唯一的对象ID（排除背景255）
    unique_objects = np.unique(object_ids)
    unique_objects = unique_objects[unique_objects != 255]
    
    if len(unique_objects) == 0:
        print(f"警告: 帧 {frame_idx} 没有有效的对象")
        pixel_details = {} if tracked_pixels else None
        return sdf_map, pixel_details
    
    if len(unique_objects) > 10:
        print(f"帧 {frame_idx}: 找到 {len(unique_objects)} 个对象（处理中...）")
    
    # 为每个对象构建点云，同时保存每个点对应的像素坐标
    object_point_clouds = {}
    object_pixel_coords = {}  # 保存每个点对应的像素坐标 (y, x)
    for obj_id in unique_objects:
        mask = (object_ids == obj_id)
        y_coords, x_coords = np.where(mask)
        obj_points = points_3d[y_coords, x_coords]  # [N, 3]
        
        # 过滤无效点（NaN或Inf）
        valid_mask = np.isfinite(obj_points).all(axis=1)
        obj_points = obj_points[valid_mask]
        valid_y = y_coords[valid_mask]
        valid_x = x_coords[valid_mask]
        
        if len(obj_points) > 0:
            object_point_clouds[obj_id] = obj_points
            # 保存每个点对应的像素坐标
            object_pixel_coords[obj_id] = np.column_stack([valid_y, valid_x])  # [N, 2]
    
    if len(object_point_clouds) == 0:
        print(f"警告: 帧 {frame_idx} 没有有效的点云")
        pixel_details = {} if tracked_pixels else None
        return sdf_map, pixel_details
    
    # 为每个对象构建KDTree（用于快速最近邻搜索）
    object_trees = {}
    for obj_id, obj_points in object_point_clouds.items():
        if len(obj_points) > 0:
            object_trees[obj_id] = cKDTree(obj_points)
    
    # 初始化像素详细信息记录
    pixel_details = {}
    if tracked_pixels is not None:
        # 准备相机参数（所有像素共享）
        camera_params = {}
        if intrinsics is not None:
            camera_params["intrinsics"] = {
                "matrix": intrinsics.tolist(),
                "format": "3x3 camera intrinsic matrix K",
                "description": "fx, fy: focal lengths; cx, cy: principal point",
                "fx": float(intrinsics[0, 0]),
                "fy": float(intrinsics[1, 1]),
                "cx": float(intrinsics[0, 2]),
                "cy": float(intrinsics[1, 2])
            }
        if extrinsics is not None:
            # 计算c2w矩阵
            w2c = np.vstack([extrinsics, [0, 0, 0, 1]])
            c2w = np.linalg.inv(w2c)
            camera_params["extrinsics"] = {
                "w2c": extrinsics.tolist(),
                "c2w": c2w.tolist(),
                "format": "w2c: world-to-camera [3x4], c2w: camera-to-world [4x4]",
                "description": "w2c transforms world coordinates to camera coordinates"
            }
        
        for x, y in tracked_pixels:
            if 0 <= y < H and 0 <= x < W:
                pixel_key = f"pixel_{x}_{y}"
                pixel_details[pixel_key] = {
                    "pixel_coords": {"x": int(x), "y": int(y)},
                    "rgb_value": None,
                    "object_id": int(object_ids[y, x]),
                    "object_name": None,
                    "depth_value": None,
                    "3d_world_coords": None,
                    "camera_parameters": camera_params.copy() if camera_params else None,
                    "coordinate_transformation": {},
                    "distances_to_objects": {},
                    "final_sdf_value": None
                }
                
                # 记录RGB值
                if rgb_image is not None and rgb_image.shape[:2] == (H, W):
                    if len(rgb_image.shape) == 3:
                        pixel_details[pixel_key]["rgb_value"] = {
                            "r": int(rgb_image[y, x, 2]),
                            "g": int(rgb_image[y, x, 1]),
                            "b": int(rgb_image[y, x, 0])
                        }
                
                # 记录深度值
                if depth_map is not None and depth_map.shape == (H, W):
                    pixel_details[pixel_key]["depth_value"] = float(depth_map[y, x])
                
                # 记录3D世界坐标和计算过程
                point_3d = points_3d[y, x]
                if np.isfinite(point_3d).all():
                    pixel_details[pixel_key]["3d_world_coords"] = {
                        "x": float(point_3d[0]),
                        "y": float(point_3d[1]),
                        "z": float(point_3d[2])
                    }
                    
                    # 记录从像素坐标到3D世界坐标的计算过程
                    if intrinsics is not None and extrinsics is not None and depth_map is not None:
                        depth_val = depth_map[y, x]
                        if depth_val > 0 and np.isfinite(depth_val):
                            # 步骤1: 像素坐标（齐次坐标）
                            pixel_homo = np.array([x, y, 1.0])
                            
                            # 步骤2: 反投影到相机坐标系
                            K_inv = np.linalg.inv(intrinsics)
                            camera_ray = K_inv @ pixel_homo  # 归一化射线方向
                            camera_coords = camera_ray * depth_val  # 乘以深度得到相机坐标
                            
                            # 步骤3: 转换到世界坐标系
                            w2c = np.vstack([extrinsics, [0, 0, 0, 1]])
                            c2w = np.linalg.inv(w2c)
                            camera_coords_homo = np.append(camera_coords, 1.0)
                            world_coords_calc = (c2w @ camera_coords_homo)[:3]
                            
                            pixel_details[pixel_key]["coordinate_transformation"] = {
                                "step_1_pixel_coords": {
                                    "description": "像素坐标（齐次坐标）",
                                    "value": pixel_homo.tolist(),
                                    "format": "[u, v, 1]"
                                },
                                "step_2_camera_ray": {
                                    "description": "反投影到相机坐标系（归一化射线方向）",
                                    "formula": "K_inv @ [u, v, 1]",
                                    "K_inv": K_inv.tolist(),
                                    "value": camera_ray.tolist(),
                                    "format": "[X_c/Z_c, Y_c/Z_c, 1] (归一化)"
                                },
                                "step_3_camera_coords": {
                                    "description": "相机坐标系3D点",
                                    "formula": "camera_ray * depth",
                                    "depth": float(depth_val),
                                    "value": camera_coords.tolist(),
                                    "format": "[X_c, Y_c, Z_c] in camera frame"
                                },
                                "step_4_world_coords": {
                                    "description": "世界坐标系3D点",
                                    "formula": "c2w @ [X_c, Y_c, Z_c, 1]",
                                    "c2w_matrix": c2w.tolist(),
                                    "value": world_coords_calc.tolist(),
                                    "format": "[X_w, Y_w, Z_w] in world frame"
                                },
                                "verification": {
                                    "description": "验证：计算值应与points_3d中的值一致",
                                    "calculated": world_coords_calc.tolist(),
                                    "from_points_3d": point_3d.tolist(),
                                    "difference": (world_coords_calc - point_3d).tolist(),
                                    "max_error": float(np.abs(world_coords_calc - point_3d).max())
                                }
                            }
                
                # 记录物体名称
                obj_id = object_ids[y, x]
                if label_map and obj_id in label_map:
                    pixel_details[pixel_key]["object_name"] = label_map[obj_id]
                else:
                    pixel_details[pixel_key]["object_name"] = f"object_{obj_id}"
    
    # 对每个像素计算SDF值
    # 优化：批量处理每个对象的点
    for obj_id in unique_objects:
        # 获取当前对象的所有像素位置
        current_mask = (object_ids == obj_id)
        y_coords, x_coords = np.where(current_mask)
        current_points = points_3d[y_coords, x_coords]  # [N, 3]
        
        # 过滤无效点
        valid_mask = np.isfinite(current_points).all(axis=1)
        if valid_mask.sum() == 0:
            continue
        
        current_points = current_points[valid_mask]
        valid_y = y_coords[valid_mask]
        valid_x = x_coords[valid_mask]
        
        # 对当前对象的每个点，计算到其他对象的最近距离
        for point, y, x in zip(current_points, valid_y, valid_x):
            min_dist = np.inf
            
            # 检查是否是需要记录的像素
            pixel_key = None
            if tracked_pixels is not None and (x, y) in tracked_pixels:
                pixel_key = f"pixel_{x}_{y}"
            
            # 遍历所有其他对象，记录到每个对象的距离
            distances_to_objects = {}
            for other_obj_id, other_tree in object_trees.items():
                if other_obj_id == obj_id:
                    continue  # 跳过自身
                
                # 计算到其他对象最近点的距离和索引
                dist, idx = other_tree.query(point, k=1)
                min_dist = min(min_dist, dist)
                
                # 如果是要记录的像素，保存到各个对象的距离、3D坐标和像素坐标
                if pixel_key is not None:
                    other_obj_name = label_map.get(other_obj_id, f"object_{other_obj_id}") if label_map else f"object_{other_obj_id}"
                    
                    # 确保idx是整数索引（query可能返回标量或数组）
                    idx = int(idx) if np.isscalar(idx) else int(idx[0])
                    
                    # 获取最近点的3D坐标和像素坐标
                    nearest_point_3d = object_point_clouds[other_obj_id][idx]
                    nearest_pixel_coords = object_pixel_coords[other_obj_id][idx]
                    
                    distances_to_objects[other_obj_name] = {
                        "object_id": int(other_obj_id),
                        "distance": float(dist),
                        "nearest_point_3d": {
                            "x": float(nearest_point_3d[0]),
                            "y": float(nearest_point_3d[1]),
                            "z": float(nearest_point_3d[2])
                        },
                        "nearest_pixel_coords": {
                            "x": int(nearest_pixel_coords[1]),  # x坐标（列）
                            "y": int(nearest_pixel_coords[0])   # y坐标（行）
                        }
                    }
            
            if min_dist < np.inf:
                sdf_map[y, x] = min_dist
            
            # 保存记录的像素信息
            if pixel_key is not None and pixel_key in pixel_details:
                pixel_details[pixel_key]["distances_to_objects"] = distances_to_objects
                pixel_details[pixel_key]["final_sdf_value"] = float(min_dist) if min_dist < np.inf else None
    
    # 处理无效值（将inf设为0或最大有效值）
    if np.isfinite(sdf_map).any():
        max_valid = sdf_map[np.isfinite(sdf_map)].max()
        sdf_map[sdf_map == np.inf] = max_valid if max_valid > 0 else 0
    else:
        sdf_map[sdf_map == np.inf] = 0
    
    return sdf_map, pixel_details


def mark_pixels_on_rgb(rgb_image, tracked_pixels, pixel_details, marker_size=5):
    """
    在RGB图像上标记追踪像素和最近点像素
    
    Args:
        rgb_image: [H, W, 3] RGB图像（BGR格式，OpenCV）
        tracked_pixels: list of (x, y) tuples，追踪的像素坐标
        pixel_details: dict，像素详细信息（包含distances_to_objects）
        marker_size: 标记大小（像素）
    
    Returns:
        marked_image: [H, W, 3] 标记后的RGB图像
    """
    if rgb_image is None:
        return None
    
    marked_image = rgb_image.copy()
    H, W = marked_image.shape[:2]
    
    # 定义颜色（BGR格式）
    TRACKED_COLOR = (0, 0, 255)  # 红色 - 追踪像素
    NEAREST_COLORS = [
        (0, 255, 0),    # 绿色 - 最近点1
        (255, 0, 0),    # 蓝色 - 最近点2
        (0, 255, 255),  # 黄色 - 最近点3
        (255, 0, 255),  # 洋红色 - 最近点4
        (255, 255, 0),  # 青色 - 最近点5
    ]
    
    # 标记追踪像素（用红色方块）
    for x, y in tracked_pixels:
        if 0 <= x < W and 0 <= y < H:
            pixel_key = f"pixel_{x}_{y}"
            
            # 画追踪像素的标记（红色方块，稍微扩大以便观察）
            # 先画一个半透明的填充方块
            overlay = marked_image.copy()
            cv2.rectangle(overlay,
                         (max(0, x - marker_size), max(0, y - marker_size)),
                         (min(W-1, x + marker_size), min(H-1, y + marker_size)),
                         TRACKED_COLOR, -1)  # -1表示填充
            cv2.addWeighted(overlay, 0.5, marked_image, 0.5, 0, marked_image)
            
            # 在追踪像素周围画一个外框（更明显）
            cv2.rectangle(marked_image,
                         (max(0, x - marker_size - 2), max(0, y - marker_size - 2)),
                         (min(W-1, x + marker_size + 2), min(H-1, y + marker_size + 2)),
                         TRACKED_COLOR, 3)  # 3像素宽的边框，更明显
            
            # 标记该像素对应的最近点（从distances_to_objects中获取）
            if pixel_key in pixel_details:
                distances_to_objects = pixel_details[pixel_key].get("distances_to_objects", {})
                color_idx = 0
                
                for obj_name, obj_info in distances_to_objects.items():
                    if "nearest_pixel_coords" in obj_info:
                        nx = obj_info["nearest_pixel_coords"]["x"]
                        ny = obj_info["nearest_pixel_coords"]["y"]
                        
                        if 0 <= nx < W and 0 <= ny < H:
                            # 使用不同颜色标记不同物体的最近点
                            nearest_color = NEAREST_COLORS[color_idx % len(NEAREST_COLORS)]
                            
                            # 画最近点的标记（圆圈，半透明填充）
                            overlay = marked_image.copy()
                            cv2.circle(overlay, (nx, ny), marker_size, nearest_color, -1)
                            cv2.addWeighted(overlay, 0.6, marked_image, 0.4, 0, marked_image)
                            
                            # 画外框（更明显）
                            cv2.circle(marked_image, (nx, ny), marker_size + 2, nearest_color, 3)
                            
                            color_idx += 1
    
    return marked_image


def apply_exponential_transform(sdf_map, k=5.0):
    """
    对SDF值应用指数变换，将接近0的部分量程放大，接近1的部分压缩
    
    变换公式: exp_sdf = 1 - exp(-k * normalized_sdf)
    其中 normalized_sdf 是归一化到[0,1]的SDF值
    
    Args:
        sdf_map: [H, W] SDF值
        k: 指数变换参数，越大则变换越强（默认5.0）
    
    Returns:
        exp_sdf: [H, W] 指数变换后的SDF值
    """
    # 归一化SDF到[0,1]
    valid_mask = (sdf_map > 0) & np.isfinite(sdf_map)
    
    if valid_mask.sum() == 0:
        return np.zeros_like(sdf_map)
    
    sdf_valid = sdf_map[valid_mask]
    sdf_min = sdf_valid.min()
    sdf_max = sdf_valid.max()
    
    if sdf_max > sdf_min:
        normalized_sdf = (sdf_map - sdf_min) / (sdf_max - sdf_min)
    else:
        normalized_sdf = np.zeros_like(sdf_map)
    
    # 应用指数变换：1 - exp(-k * normalized_sdf)
    # 这样接近0的部分会被放大，接近1的部分会被压缩
    exp_sdf = 1.0 - np.exp(-k * normalized_sdf)
    
    # 将无效值设为0
    exp_sdf[~valid_mask] = 0.0
    
    return exp_sdf


def visualize_sdf(sdf_map, output_path):
    """
    将SDF值可视化为灰度图
    
    Args:
        sdf_map: [H, W] SDF值
        output_path: 输出路径
    """
    # 归一化到0-255
    valid_mask = (sdf_map > 0) & np.isfinite(sdf_map)
    
    if valid_mask.sum() == 0:
        print("警告: 没有有效的SDF值")
        gray_image = np.zeros_like(sdf_map, dtype=np.uint8)
    else:
        sdf_valid = sdf_map[valid_mask]
        sdf_min = sdf_valid.min()
        sdf_max = sdf_valid.max()
        
        if sdf_max > sdf_min:
            sdf_normalized = (sdf_map - sdf_min) / (sdf_max - sdf_min) * 255
        else:
            sdf_normalized = np.zeros_like(sdf_map)
        
        sdf_normalized = np.clip(sdf_normalized, 0, 255).astype(np.uint8)
        gray_image = sdf_normalized
    
    cv2.imwrite(output_path, gray_image)
    # 注释掉每帧的打印，避免输出过多信息
    # print(f"SDF可视化已保存到: {output_path}")


def create_gif_from_images(image_dir, output_gif_path, pattern="*.png", duration=100, loop=0):
    """
    从图像序列创建GIF文件
    
    Args:
        image_dir: 图像目录
        output_gif_path: 输出GIF路径
        pattern: 图像文件模式（默认"*.png"）
        duration: 每帧持续时间（毫秒，默认100ms）
        loop: 循环次数（0表示无限循环，默认0）
    """
    image_dir = Path(image_dir)
    image_files = sorted(glob.glob(str(image_dir / pattern)))
    
    if len(image_files) == 0:
        print(f"警告: 在 {image_dir} 中未找到匹配 {pattern} 的图像文件")
        return
    
    print(f"正在创建GIF: {output_gif_path} ({len(image_files)} 帧)")
    
    # 读取所有图像
    images = []
    for img_path in tqdm(image_files, desc="读取图像"):
        try:
            img = Image.open(img_path)
            # 转换为RGB模式（GIF需要）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"警告: 无法读取图像 {img_path}: {e}")
    
    if len(images) == 0:
        print(f"错误: 没有有效的图像")
        return
    
    # 保存为GIF
    try:
        images[0].save(
            str(output_gif_path),
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop
        )
        print(f"✓ GIF已保存到: {output_gif_path}")
    except Exception as e:
        print(f"错误: 无法创建GIF文件 {output_gif_path}: {e}")


def create_video_from_images(image_dir, output_video_path, fps=10.0, pattern="*.png"):
    """
    从图像序列创建视频文件
    
    Args:
        image_dir: 图像目录
        output_video_path: 输出视频路径
        fps: 帧率（默认10.0）
        pattern: 图像文件模式（默认"*.png"）
    """
    import glob
    
    image_dir = Path(image_dir)
    image_files = sorted(glob.glob(str(image_dir / pattern)))
    
    if len(image_files) == 0:
        print(f"警告: 在 {image_dir} 中未找到匹配 {pattern} 的图像文件")
        return
    
    # 读取第一张图像获取尺寸
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"错误: 无法读取图像 {image_files[0]}")
        return
    
    height, width = first_image.shape[:2]
    
    # 创建视频写入器（使用H.264编码，兼容性更好）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"错误: 无法创建视频文件 {output_video_path}")
        return
    
    print(f"正在创建视频: {output_video_path} ({len(image_files)} 帧)")
    for img_path in tqdm(image_files, desc="写入视频帧"):
        img = cv2.imread(img_path)
        if img is not None:
            # 确保图像尺寸一致
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            out.write(img)
        else:
            print(f"警告: 无法读取图像 {img_path}")
    
    out.release()
    print(f"✓ 视频已保存到: {output_video_path}")


def process_single_episode(input_dir, output_dir, model, device, max_frames=None, tracked_pixels=None, video_fps=10.0):
    """
    处理单个episode的数据
    
    Args:
        input_dir: 输入目录（包含rgb.mp4和frame_*/目录）
        output_dir: 输出目录
        model: DepthAnything3模型
        device: 设备
        max_frames: 最大处理帧数
        tracked_pixels: list of (x, y) tuples，要记录的像素坐标（格式：[(x1, y1), (x2, y2), ...]）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 从视频提取帧
    video_path = input_path / "rgb.mp4"
    # if not video_path.exists():
    #     raise FileNotFoundError(f"找不到视频文件: {video_path}")
    
    frames_dir = output_path / "frames"
    frame_paths = extract_frames_from_video(str(video_path), str(frames_dir), max_frames)
    
    # 2. 生成深度图和点云
    depth_dir = output_path / "depths"
    depth_maps, point_clouds, intrinsics_list, extrinsics_list = generate_depth_and_pointcloud(
        model, frame_paths, str(depth_dir), device
    )
    
    # 3. 加载标签映射（如果有）
    labels_path = input_path / "labels.txt"
    label_map = load_object_labels(str(labels_path))
    if label_map:
        print(f"已加载 {len(label_map)} 个物体标签")
    
    # 4. 计算SDF
    sdf_dir = output_path / "sdf"
    sdf_npy_dir = output_path / "sdf_npy"
    sdf_vis_dir = output_path / "sdf_vis"
    sdf_exp_npy_dir = output_path / "sdf_exp_npy"  # 指数变换SDF的npy文件
    sdf_exp_vis_dir = output_path / "sdf_exp_vis"  # 指数变换SDF的可视化
    pixel_details_dir = output_path / "pixel_details"
    rgb_marked_dir = output_path / "rgb_marked"  # 标记后的RGB图像
    
    os.makedirs(sdf_dir, exist_ok=True)
    os.makedirs(sdf_npy_dir, exist_ok=True)
    os.makedirs(sdf_vis_dir, exist_ok=True)
    os.makedirs(sdf_exp_npy_dir, exist_ok=True)
    os.makedirs(sdf_exp_vis_dir, exist_ok=True)
    if tracked_pixels:
        os.makedirs(pixel_details_dir, exist_ok=True)
        os.makedirs(rgb_marked_dir, exist_ok=True)
    
    print("正在计算SDF值...")
    all_pixel_details = {}  # 存储所有帧的像素详细信息
    
    for i in tqdm(range(len(frame_paths)), desc="计算SDF"):
        # 加载annotated_frame_index
        frame_dir = input_path / f"frame_{i:04d}"
        index_path = frame_dir / "annotated_frame_index.npy"
        
        if not index_path.exists():
            # 尝试从npz文件加载
            npz_path = input_path / f"frame_{i:04d}.npz"
            if npz_path.exists():
                data = np.load(npz_path)
                if 'annotated_frame_index' in data:
                    object_ids = data['annotated_frame_index']
                else:
                    print(f"警告: 帧 {i} 没有annotated_frame_index，跳过")
                    continue
            else:
                print(f"警告: 帧 {i} 没有annotated_frame_index，跳过")
                continue
        else:
            object_ids = load_annotated_frame_index(str(index_path))
        
        # 获取3D点云
        points_3d = point_clouds[i]
        
        # 调整object_ids的尺寸以匹配points_3d
        if object_ids.shape != points_3d.shape[:2]:
            # 需要调整尺寸
            object_ids = cv2.resize(object_ids.astype(np.float32), 
                                   (points_3d.shape[1], points_3d.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(object_ids.dtype)
        
        # 加载RGB图像（如果存在且需要记录像素信息）
        rgb_image = None
        if tracked_pixels:
            frame_image_path = frames_dir / f"frame_{i:04d}.png"
            if frame_image_path.exists():
                rgb_image = cv2.imread(str(frame_image_path))
                if rgb_image is not None:
                    # 调整尺寸以匹配points_3d
                    if rgb_image.shape[:2] != points_3d.shape[:2]:
                        rgb_image = cv2.resize(rgb_image, 
                                              (points_3d.shape[1], points_3d.shape[0]))
        
        # 获取深度图
        depth_map = depth_maps[i] if i < len(depth_maps) else None
        
        # 获取当前帧的相机参数
        current_intrinsics = intrinsics_list[i] if i < len(intrinsics_list) else None
        current_extrinsics = extrinsics_list[i] if i < len(extrinsics_list) else None
        
        # 计算SDF（带详细记录）
        sdf_map, pixel_details = compute_sdf_for_frame(
            points_3d, object_ids, i, str(sdf_dir),
            rgb_image=rgb_image,
            depth_map=depth_map,
            tracked_pixels=tracked_pixels,
            label_map=label_map,
            intrinsics=current_intrinsics,
            extrinsics=current_extrinsics
        )
        
        # 保存像素详细信息
        if tracked_pixels and pixel_details:
            all_pixel_details[f"frame_{i:04d}"] = pixel_details
            pixel_details_path = pixel_details_dir / f"pixel_details_{i:04d}.json"
            with open(pixel_details_path, 'w', encoding='utf-8') as f:
                json.dump(pixel_details, f, indent=2, ensure_ascii=False)
            
            # 生成标记后的RGB图像
            if rgb_image is not None:
                marked_rgb = mark_pixels_on_rgb(rgb_image, tracked_pixels, pixel_details, marker_size=5)
                if marked_rgb is not None:
                    marked_rgb_path = rgb_marked_dir / f"rgb_marked_{i:04d}.png"
                    cv2.imwrite(str(marked_rgb_path), marked_rgb)
        
        # 保存原始SDF为npy文件
        sdf_npy_path = sdf_npy_dir / f"sdf_{i:04d}.npy"
        np.save(sdf_npy_path, sdf_map)
        
        # 可视化原始SDF
        sdf_vis_path = sdf_vis_dir / f"sdf_{i:04d}.png"
        visualize_sdf(sdf_map, str(sdf_vis_path))
        
        # 计算指数变换的SDF
        sdf_exp = apply_exponential_transform(sdf_map, k=5.0)
        
        # 保存指数变换SDF为npy文件
        sdf_exp_npy_path = sdf_exp_npy_dir / f"sdf_exp_{i:04d}.npy"
        np.save(sdf_exp_npy_path, sdf_exp)
        
        # 可视化指数变换SDF
        sdf_exp_vis_path = sdf_exp_vis_dir / f"sdf_exp_{i:04d}.png"
        visualize_sdf(sdf_exp, str(sdf_exp_vis_path))
    
    # 保存所有帧的像素详细信息汇总
    if tracked_pixels and all_pixel_details:
        summary_path = pixel_details_dir / "pixel_details_summary.json"
        summary = {
            "tracked_pixels": [{"x": int(x), "y": int(y)} for x, y in tracked_pixels],
            "total_frames": len(all_pixel_details),
            "frame_details": all_pixel_details
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n像素详细信息已保存到: {summary_path}")
    
    # 5. 生成视频
    print("\n正在生成视频...")
    if len(frame_paths) > 0:
        # 生成深度图视频
        depth_video_path = output_path / "depth_video.mp4"
        create_video_from_images(
            str(depth_dir),
            str(depth_video_path),
            fps=video_fps,
            pattern="depth_*.png"
        )
        
        # 生成原始SDF视频
        sdf_video_path = output_path / "sdf_video.mp4"
        create_video_from_images(
            str(sdf_vis_dir),
            str(sdf_video_path),
            fps=video_fps,
            pattern="sdf_*.png"
        )
        
        # 生成指数变换SDF视频
        sdf_exp_video_path = output_path / "sdf_exp_video.mp4"
        create_video_from_images(
            str(sdf_exp_vis_dir),
            str(sdf_exp_video_path),
            fps=video_fps,
            pattern="sdf_exp_*.png"
        )
        
        # 生成标记RGB视频（如果有追踪像素）
        if tracked_pixels:
            rgb_marked_video_path = output_path / "rgb_marked_video.mp4"
            create_video_from_images(
                str(rgb_marked_dir),
                str(rgb_marked_video_path),
                fps=video_fps,
                pattern="rgb_marked_*.png"
            )
    
    # 6. 生成GIF
    print("\n正在生成GIF...")
    if len(frame_paths) > 0:
        # 计算每帧持续时间（毫秒），基于视频帧率
        gif_duration = int(1000 / video_fps)  # 转换为毫秒
        
        # 生成RGB GIF
        rgb_gif_path = output_path / "rgb.gif"
        create_gif_from_images(
            str(frames_dir),
            str(rgb_gif_path),
            pattern="frame_*.png",
            duration=gif_duration,
            loop=0
        )
        
        # 生成标记RGB GIF（如果有追踪像素）
        if tracked_pixels:
            rgb_marked_gif_path = output_path / "rgb_marked.gif"
            create_gif_from_images(
                str(rgb_marked_dir),
                str(rgb_marked_gif_path),
                pattern="rgb_marked_*.png",
                duration=gif_duration,
                loop=0
            )
        
        # 生成深度图GIF
        depth_gif_path = output_path / "depth.gif"
        create_gif_from_images(
            str(depth_dir),
            str(depth_gif_path),
            pattern="depth_*.png",
            duration=gif_duration,
            loop=0
        )
        
        # 生成SDF可视化GIF
        sdf_vis_gif_path = output_path / "sdf_vis.gif"
        create_gif_from_images(
            str(sdf_vis_dir),
            str(sdf_vis_gif_path),
            pattern="sdf_*.png",
            duration=gif_duration,
            loop=0
        )
        
        # 生成指数变换SDF可视化GIF
        sdf_exp_vis_gif_path = output_path / "sdf_exp_vis.gif"
        create_gif_from_images(
            str(sdf_exp_vis_dir),
            str(sdf_exp_vis_gif_path),
            pattern="sdf_exp_*.png",
            duration=gif_duration,
            loop=0
        )
    
    print(f"\n处理完成！结果保存在: {output_dir}")


def parse_pixel_coords(pixel_str):
    """
    解析像素坐标字符串，支持多种格式：
    - "100,200" -> [(100, 200)]
    - "100,200;300,400" -> [(100, 200), (300, 400)]
    - "100,200 300,400" -> [(100, 200), (300, 400)]
    """
    if not pixel_str:
        return None
    
    pixels = []
    # 支持分号或空格分隔
    for part in pixel_str.replace(';', ' ').split():
        if ',' in part:
            try:
                x, y = map(int, part.split(','))
                pixels.append((x, y))
            except ValueError:
                print(f"警告: 无法解析像素坐标 '{part}'，跳过")
    
    return pixels if pixels else None


def main():
    parser = argparse.ArgumentParser(description="计算Surface Distance Field (SDF)")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="输入目录（包含rgb.mp4和frame_*/目录）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    parser.add_argument("--model_name", type=str, default="depth-anything/DA3-LARGE",
                       help="Depth-Anything-3模型名称")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="最大处理帧数（None表示全部）")
    parser.add_argument("--track_pixels", type=str, default=None,
                       help="要记录的像素坐标，格式：'x1,y1;x2,y2' 或 'x1,y1 x2,y2'，例如：'100,200;300,400'")
    parser.add_argument("--video_fps", type=float, default=10.0,
                       help="生成视频的帧率（默认10.0）")
    
    args = parser.parse_args()
    
    # 解析要记录的像素坐标
    tracked_pixels = parse_pixel_coords(args.track_pixels)
    if tracked_pixels:
        print(f"将记录 {len(tracked_pixels)} 个像素的详细信息: {tracked_pixels}")
    
    # 检查Depth-Anything-3是否可用
    if not DA3_AVAILABLE:
        print("错误: Depth-Anything-3未正确安装，无法继续")
        sys.exit(1)
    
    # 加载模型
    print(f"正在加载模型: {args.model_name}")
    try:
        model = DepthAnything3.from_pretrained(args.model_name)
    except Exception as e:
        print(f"无法加载模型: {e}")
        print("请确保Depth-Anything-3已正确安装，或使用本地模型路径")
        sys.exit(1)
    
    device = torch.device(args.device)
    
    # 处理数据
    process_single_episode(
        args.input_dir,
        args.output_dir,
        model,
        device,
        args.max_frames,
        tracked_pixels=tracked_pixels,
        video_fps=args.video_fps
    )


if __name__ == "__main__":
    main()

