#!/usr/bin/env python3
"""
单个case测试脚本
测试从bridge数据集处理到SDF计算的完整流程
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"{'='*60}")
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"错误: {description} 失败")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="测试单个case：从bridge数据集到SDF计算")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="输入目录（bridge_depth格式，包含episode目录）")
    parser.add_argument("--episode_id", type=str, required=True,
                       help="要测试的episode ID（例如：00000）")
    parser.add_argument("--stream_id", type=int, default=0,
                       help="要处理的stream ID（默认0，对应images0）")
    parser.add_argument("--output_base", type=str, default="output/test",
                       help="输出基础目录（默认：output/test）")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="设备 (cuda:0/cpu)")
    parser.add_argument("--max_frames", type=int, default=10,
                       help="最大处理帧数（默认10，用于快速测试）")
    parser.add_argument("--skip_dataset_process", action="store_true",
                       help="跳过数据集处理步骤（如果已经处理过）")
    parser.add_argument("--skip_sdf", action="store_true",
                       help="跳过SDF计算步骤")
    parser.add_argument("--track_pixels", type=str, default=None,
                       help="要记录的像素坐标，格式：'x1,y1;x2,y2'，例如：'320,240;160,120'")
    
    args = parser.parse_args()
    
    # 路径设置
    input_dir = Path(args.input_dir)
    episode_id = args.episode_id
    stream_id = args.stream_id
    output_base = Path(args.output_base)
    
    # Step 1: 数据集处理（生成RGB视频、描述、分割掩码）
    dataset_output_dir = output_base / "dataset_processed"
    episode_output_dir = dataset_output_dir / episode_id / f"images{stream_id}"
    
    if not args.skip_dataset_process:
        print("\n" + "="*60)
        print("步骤1: 数据集处理（生成RGB视频、描述、分割掩码）")
        print("="*60)
        
        cmd = [
            sys.executable, "brdige_dataset_process_depth.py",
            "--input_dir", str(input_dir),
            "--output_dir", str(dataset_output_dir),
            "--max_videos", "1",  # 只处理一个episode
            "--device", args.device,
        ]
        
        # 检查是否有特定的episode
        episode_path = input_dir / episode_id
        if not episode_path.exists():
            print(f"错误: Episode {episode_id} 不存在于 {input_dir}")
            sys.exit(1)
        
        if not run_command(cmd, "数据集处理"):
            print("数据集处理失败，请检查错误信息")
            sys.exit(1)
        
        # 检查输出
        rgb_video = episode_output_dir / "rgb.mp4"
        if not rgb_video.exists():
            print(f"错误: 未找到生成的RGB视频: {rgb_video}")
            sys.exit(1)
        print(f"✓ RGB视频已生成: {rgb_video}")
        
        # 检查labels.txt
        labels_file = dataset_output_dir / "labels.txt"
        if labels_file.exists():
            print(f"✓ 标签文件已生成: {labels_file}")
        else:
            print(f"⚠ 警告: 未找到标签文件: {labels_file}")
    else:
        print("跳过数据集处理步骤")
        if not episode_output_dir.exists():
            print(f"错误: 输出目录不存在: {episode_output_dir}")
            print("请先运行数据集处理步骤，或移除 --skip_dataset_process 参数")
            sys.exit(1)
    
    # Step 2: SDF计算
    if not args.skip_sdf:
        print("\n" + "="*60)
        print("步骤2: SDF计算")
        print("="*60)
        
        sdf_output_dir = output_base / "sdf_processed" / episode_id
        
        # 准备输入目录（需要rgb.mp4和labels.txt）
        # 注意：episode_id可能被转换为整数格式（如"00000" -> "0"）
        # 需要检查实际存在的目录
        if not episode_output_dir.exists():
            # 尝试使用整数格式的episode_id
            episode_id_int = str(int(episode_id)) if episode_id.isdigit() else episode_id
            episode_output_dir_int = dataset_output_dir / episode_id_int / f"images{stream_id}"
            if episode_output_dir_int.exists():
                episode_output_dir = episode_output_dir_int
                print(f"[INFO] 使用实际路径: {episode_output_dir}")
            else:
                print(f"错误: 找不到输出目录: {episode_output_dir} 或 {episode_output_dir_int}")
                sys.exit(1)
        
        sdf_input_dir = episode_output_dir
        
        # 检查labels.txt是否存在
        labels_file = dataset_output_dir / "labels.txt"
        if labels_file.exists():
            # 复制labels.txt到episode目录（process_sdf.py期望在这里）
            import shutil
            target_labels = sdf_input_dir / "labels.txt"
            if not target_labels.exists():
                shutil.copy(labels_file, target_labels)
                print(f"✓ 已复制labels.txt到: {target_labels}")
        
        cmd = [
            sys.executable, "process_sdf.py",
            "--input_dir", str(sdf_input_dir),
            "--output_dir", str(sdf_output_dir),
            "--device", args.device.split(":")[0],  # process_sdf.py使用cuda而不是cuda:0
            "--max_frames", str(args.max_frames),
        ]
        
        # 添加 track_pixels 参数（如果提供）
        if hasattr(args, 'track_pixels') and args.track_pixels:
            cmd.extend(["--track_pixels", args.track_pixels])
        
        if not run_command(cmd, "SDF计算"):
            print("SDF计算失败，请检查错误信息")
            sys.exit(1)
        
        # 检查输出
        raw_output = sdf_output_dir / "raw"
        filtered_output = sdf_output_dir / "filtered"
        if raw_output.exists() and filtered_output.exists():
            print(f"✓ SDF计算结果已生成:")
            print(f"  - 原始模式: {raw_output}")
            print(f"  - 滤波模式: {filtered_output}")
        else:
            print(f"⚠ 警告: 部分输出目录不存在")
    else:
        print("跳过SDF计算步骤")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print(f"\n输出目录:")
    print(f"  - 数据集处理: {dataset_output_dir}")
    if not args.skip_sdf:
        print(f"  - SDF计算: {output_base / 'sdf_processed' / episode_id}")
    print()

if __name__ == "__main__":
    main()

