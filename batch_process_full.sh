#!/bin/bash
# 完整流程批处理脚本
# 一次性处理多个episode：数据集处理 + SDF计算

set -e  # 遇到错误立即退出

# ==================== 配置参数 ====================
INPUT_DIR="bridge_depth"           # 输入目录（bridge数据集格式）
OUTPUT_BASE="output/batch"          # 输出基础目录
DEVICE="cuda:0"                    # 设备
MAX_FRAMES=10                      # 最大处理帧数（用于快速测试，None表示全部）
STREAM_ID=0                         # stream ID（默认0，对应images0）

# 像素跟踪配置（可选）
TRACK_PIXELS="320,240"  # 要跟踪的像素坐标，留空则不跟踪
# TRACK_PIXELS=""                    # 不使用像素跟踪

# Episode列表（可以手动指定，或从输入目录自动获取）
# 方式1：手动指定
EPISODE_IDS=("00000" "00001" "00002")
# 方式2：自动获取（取消下面的注释）
# EPISODE_IDS=($(ls -d ${INPUT_DIR}/*/ 2>/dev/null | xargs -n1 basename | sort))

# 并行处理数量（0表示串行处理，>0表示并行处理N个episode）
PARALLEL_JOBS=0

# ==================== 函数定义 ====================

# 处理单个episode的完整流程
process_episode() {
    local episode_id=$1
    local input_dir=$2
    local output_base=$3
    local device=$4
    local max_frames=$5
    local stream_id=$6
    local track_pixels=$7
    
    echo ""
    echo "=========================================="
    echo "处理 Episode: $episode_id"
    echo "=========================================="
    
    # 检查输入目录是否存在
    local episode_path="${input_dir}/${episode_id}"
    if [ ! -d "$episode_path" ]; then
        echo "⚠ 警告: Episode $episode_id 不存在，跳过"
        return 1
    fi
    
    # 构建命令
    local cmd="python test_single_case.py"
    cmd="$cmd --input_dir $input_dir"
    cmd="$cmd --episode_id $episode_id"
    cmd="$cmd --stream_id $stream_id"
    cmd="$cmd --output_base $output_base"
    cmd="$cmd --device $device"
    cmd="$cmd --max_frames $max_frames"
    
    if [ -n "$track_pixels" ]; then
        cmd="$cmd --track_pixels \"$track_pixels\""
    fi
    
    # 执行命令
    echo "执行命令: $cmd"
    echo ""
    
    if eval $cmd; then
        echo "✓ Episode $episode_id 处理完成"
        return 0
    else
        echo "✗ Episode $episode_id 处理失败"
        return 1
    fi
}

# 并行处理函数
process_parallel() {
    local episodes=("$@")
    local max_jobs=$PARALLEL_JOBS
    local running=0
    local pids=()
    
    for episode_id in "${episodes[@]}"; do
        # 等待有空闲槽位
        while [ $running -ge $max_jobs ]; do
            for pid in "${pids[@]}"; do
                if ! kill -0 $pid 2>/dev/null; then
                    # 进程已结束，移除
                    pids=("${pids[@]/$pid}")
                    running=$((running - 1))
                fi
            done
            sleep 0.1
        done
        
        # 启动新任务
        (
            process_episode "$episode_id" "$INPUT_DIR" "$OUTPUT_BASE" "$DEVICE" "$MAX_FRAMES" "$STREAM_ID" "$TRACK_PIXELS"
        ) &
        
        local pid=$!
        pids+=($pid)
        running=$((running + 1))
        echo "启动 Episode $episode_id (PID: $pid)"
    done
    
    # 等待所有任务完成
    for pid in "${pids[@]}"; do
        wait $pid
    done
}

# ==================== 主程序 ====================

echo "=========================================="
echo "完整流程批处理"
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
echo "=========================================="
echo ""

# 检查输入目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 检查Python脚本
if [ ! -f "test_single_case.py" ]; then
    echo "错误: 找不到 test_single_case.py"
    exit 1
fi

# 记录开始时间
start_time=$(date +%s)

# 执行批处理
if [ "$PARALLEL_JOBS" -gt 0 ]; then
    echo "使用并行处理（${PARALLEL_JOBS}个任务）"
    process_parallel "${EPISODE_IDS[@]}"
else
    echo "使用串行处理"
    success_count=0
    fail_count=0
    
    for episode_id in "${EPISODE_IDS[@]}"; do
        if process_episode "$episode_id" "$INPUT_DIR" "$OUTPUT_BASE" "$DEVICE" "$MAX_FRAMES" "$STREAM_ID" "$TRACK_PIXELS"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    done
fi

# 计算耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "=========================================="
echo "批处理完成"
echo "=========================================="
echo "总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
echo "处理Episode数: ${#EPISODE_IDS[@]}"
if [ "$PARALLEL_JOBS" -eq 0 ]; then
    echo "成功: $success_count"
    echo "失败: $fail_count"
fi
echo ""
echo "输出目录:"
echo "  - 数据集处理: $OUTPUT_BASE/dataset_processed/"
echo "  - SDF计算: $OUTPUT_BASE/sdf_processed/"
echo "=========================================="

