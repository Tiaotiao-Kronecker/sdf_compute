#!/bin/bash
# 分步批处理脚本
# 步骤1：批量数据集处理
# 步骤2：批量SDF计算

set -e  # 遇到错误立即退出

# ==================== 配置参数 ====================
INPUT_DIR="bridge_depth"           # 输入目录（bridge数据集格式）
OUTPUT_BASE="output/batch"          # 输出基础目录
DEVICE="cuda:0"                    # 设备
MAX_FRAMES=10                      # 最大处理帧数（用于快速测试，None表示全部）
STREAM_ID=0                         # stream ID（默认0，对应images0）

# 像素跟踪配置（可选，仅用于步骤2）
TRACK_PIXELS="320,240;160,120;480,360"  # 要跟踪的像素坐标，留空则不跟踪
# TRACK_PIXELS=""                    # 不使用像素跟踪

# Episode列表（可以手动指定，或从输入目录自动获取）
# 方式1：手动指定
EPISODE_IDS=("00000" "00001" "00002")
# 方式2：自动获取（取消下面的注释）
# EPISODE_IDS=($(ls -d ${INPUT_DIR}/*/ 2>/dev/null | xargs -n1 basename | sort))

# 并行处理数量（0表示串行处理，>0表示并行处理N个episode）
PARALLEL_JOBS=0

# 步骤控制
SKIP_STEP1=false                    # 跳过步骤1（如果已经处理过）
SKIP_STEP2=true                    # 跳过步骤2

# ==================== 函数定义 ====================

# 步骤1：批量数据集处理（一次性处理所有episode）
process_dataset_step1_batch() {
    local input_dir=$1
    local output_dir=$2
    local device=$3
    
    echo ""
    echo "  [步骤1] 批量数据集处理"
    
    # 构建命令
    local cmd="python brdige_dataset_process_depth.py"
    cmd="$cmd --input_dir $input_dir"
    cmd="$cmd --output_dir $output_dir"
    cmd="$cmd --device $device"
    # 不设置max_videos，处理所有episode
    
    echo "  执行: $cmd"
    echo ""
    
    if eval $cmd; then
        echo ""
        echo "  ✓ 批量数据集处理完成"
        return 0
    else
        echo ""
        echo "  ✗ 批量数据集处理失败"
        return 1
    fi
}

# 步骤2：处理单个episode的SDF计算
process_sdf_step2() {
    local episode_id=$1
    local dataset_output_dir=$2
    local sdf_output_dir=$3
    local device=$4
    local max_frames=$5
    local stream_id=$6
    local track_pixels=$7
    
    echo ""
    echo "  [步骤2] Episode: $episode_id"
    
    # 检查输入目录
    local episode_input_dir="${dataset_output_dir}/${episode_id}/images${stream_id}"
    if [ ! -d "$episode_input_dir" ]; then
        # 尝试整数格式
        local episode_id_int=$(echo "$episode_id" | sed 's/^0*//')
        if [ -z "$episode_id_int" ]; then
            episode_id_int="0"
        fi
        episode_input_dir="${dataset_output_dir}/${episode_id_int}/images${stream_id}"
        
        if [ ! -d "$episode_input_dir" ]; then
            echo "  ⚠ 警告: 输入目录不存在: ${dataset_output_dir}/${episode_id}/images${stream_id}"
            return 1
        fi
    fi
    
    # 检查rgb.mp4是否存在
    if [ ! -f "${episode_input_dir}/rgb.mp4" ]; then
        echo "  ⚠ 警告: 未找到rgb.mp4: ${episode_input_dir}/rgb.mp4"
        return 1
    fi
    
    # 复制labels.txt（如果存在）
    local labels_file="${dataset_output_dir}/labels.txt"
    if [ -f "$labels_file" ]; then
        cp -f "$labels_file" "${episode_input_dir}/labels.txt" 2>/dev/null || true
    fi
    
    # 构建命令
    local cmd="python process_sdf.py"
    cmd="$cmd --input_dir $episode_input_dir"
    cmd="$cmd --output_dir $sdf_output_dir"
    cmd="$cmd --device $(echo $device | cut -d: -f1)"  # process_sdf.py使用cuda而不是cuda:0
    cmd="$cmd --max_frames $max_frames"
    
    if [ -n "$track_pixels" ]; then
        cmd="$cmd --track_pixels \"$track_pixels\""
    fi
    
    echo "  执行: $cmd"
    
    if eval $cmd; then
        echo "  ✓ Episode $episode_id SDF计算完成"
        return 0
    else
        echo "  ✗ Episode $episode_id SDF计算失败"
        return 1
    fi
}

# ==================== 主程序 ====================

echo "=========================================="
echo "分步批处理"
echo "=========================================="
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_BASE"
echo "设备: $DEVICE"
echo "最大帧数: $MAX_FRAMES"
echo "Stream ID: $STREAM_ID"
if [ -n "$TRACK_PIXELS" ]; then
    echo "跟踪像素: $TRACK_PIXELS"
else
    echo "跟踪像素: 无"
fi
echo "Episode列表: ${EPISODE_IDS[@]}"
echo "并行任务数: ${PARALLEL_JOBS:-串行}"
echo "跳过步骤1: $SKIP_STEP1"
echo "跳过步骤2: $SKIP_STEP2"
echo "=========================================="
echo ""

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 检查Python脚本
if [ ! -f "brdige_dataset_process_depth.py" ] || [ ! -f "process_sdf.py" ]; then
    echo "错误: 找不到必要的Python脚本"
    exit 1
fi

# 设置输出目录
DATASET_OUTPUT_DIR="${OUTPUT_BASE}/dataset_processed"
SDF_OUTPUT_DIR="${OUTPUT_BASE}/sdf_processed"

# ==================== 步骤1：数据集处理 ====================
if [ "$SKIP_STEP1" = false ]; then
    echo ""
    echo "=========================================="
    echo "步骤1: 批量数据集处理"
    echo "=========================================="
    echo "将处理以下Episode: ${EPISODE_IDS[@]}"
    echo ""
    
    start_time=$(date +%s)
    
    # 执行批量处理（一次性处理所有episode）
    if process_dataset_step1_batch "$INPUT_DIR" "$DATASET_OUTPUT_DIR" "$DEVICE"; then
        # 验证输出文件
        success_count=0
        fail_count=0
        
        for episode_id in "${EPISODE_IDS[@]}"; do
            local rgb_video="${DATASET_OUTPUT_DIR}/${episode_id}/images${STREAM_ID}/rgb.mp4"
            # 尝试整数格式
            if [ ! -f "$rgb_video" ]; then
                local episode_id_int=$(echo "$episode_id" | sed 's/^0*//')
                if [ -z "$episode_id_int" ]; then
                    episode_id_int="0"
                fi
                rgb_video="${DATASET_OUTPUT_DIR}/${episode_id_int}/images${STREAM_ID}/rgb.mp4"
            fi
            
            if [ -f "$rgb_video" ]; then
                echo "  ✓ Episode $episode_id: $rgb_video"
                success_count=$((success_count + 1))
            else
                echo "  ⚠ Episode $episode_id: 未找到输出文件"
                fail_count=$((fail_count + 1))
            fi
        done
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        echo ""
        echo "步骤1完成:"
        echo "  成功: $success_count"
        echo "  失败: $fail_count"
        echo "  耗时: ${duration}秒"
    else
        echo ""
        echo "步骤1失败"
        exit 1
    fi
    echo ""
else
    echo "跳过步骤1（数据集处理）"
fi

# ==================== 步骤2：SDF计算 ====================
if [ "$SKIP_STEP2" = false ]; then
    echo ""
    echo "=========================================="
    echo "步骤2: 批量SDF计算"
    echo "=========================================="
    
    start_time=$(date +%s)
    success_count=0
    fail_count=0
    
    # 执行批处理
    for episode_id in "${EPISODE_IDS[@]}"; do
        if process_sdf_step2 "$episode_id" "$DATASET_OUTPUT_DIR" "${SDF_OUTPUT_DIR}/${episode_id}" "$DEVICE" "$MAX_FRAMES" "$STREAM_ID" "$TRACK_PIXELS"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    done
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "步骤2完成:"
    echo "  成功: $success_count"
    echo "  失败: $fail_count"
    echo "  耗时: ${duration}秒"
    echo ""
else
    echo "跳过步骤2（SDF计算）"
fi

# ==================== 总结 ====================
echo ""
echo "=========================================="
echo "批处理完成"
echo "=========================================="
echo "输出目录:"
echo "  - 数据集处理: $DATASET_OUTPUT_DIR"
echo "  - SDF计算: $SDF_OUTPUT_DIR"
echo "=========================================="

