#!/bin/bash
set -e

echo "=== 安装Python依赖 ==="
pip install -r requirements.txt

echo ""
echo "=== 安装SAM2代码库 ==="
if [ ! -f "thirdparty/grounded_sam_2/setup.py" ]; then
    echo "SAM2代码库不存在，正在克隆..."
    # 确保thirdparty目录存在
    mkdir -p thirdparty
    cd thirdparty
    # 如果目录存在但只有checkpoints，先备份
    if [ -d "grounded_sam_2" ] && [ ! -f "grounded_sam_2/setup.py" ]; then
        echo "备份现有的checkpoints目录..."
        if [ -d "grounded_sam_2/checkpoints" ]; then
            mv grounded_sam_2/checkpoints /tmp/sam2_checkpoints_backup_$$ 2>/dev/null || true
        fi
        rm -rf grounded_sam_2
    fi
    # 克隆Grounded-SAM-2仓库（包含SAM2和正确的目录结构）
    if [ ! -d "grounded_sam_2" ]; then
        git clone https://github.com/IDEA-Research/Grounded-SAM-2.git grounded_sam_2
    fi
    cd ..
    # 恢复checkpoints
    if [ -d "/tmp/sam2_checkpoints_backup_$$" ]; then
        mkdir -p thirdparty/grounded_sam_2/checkpoints
        mv /tmp/sam2_checkpoints_backup_$$/* thirdparty/grounded_sam_2/checkpoints/ 2>/dev/null || true
        rm -rf /tmp/sam2_checkpoints_backup_$$ 2>/dev/null || true
    fi
else
    echo "SAM2代码库已存在，跳过克隆"
fi

echo ""
echo "=== 下载SAM2模型权重 ==="
mkdir -p thirdparty/grounded_sam_2/checkpoints
if [ ! -f "thirdparty/grounded_sam_2/checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "下载SAM2权重..."
    wget -P thirdparty/grounded_sam_2/checkpoints \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
else
    echo "SAM2权重已存在，跳过下载"
fi

echo ""
echo "=== 安装SAM2 ==="
if [ -f "thirdparty/grounded_sam_2/setup.py" ]; then
    cd thirdparty/grounded_sam_2
    pip install -e .
    cd ../..
    echo "✓ SAM2安装完成"
else
    echo "✗ 警告: 未找到SAM2的setup.py，请手动克隆SAM2代码库"
    echo "  运行: cd thirdparty && git clone https://github.com/IDEA-Research/Grounded-SAM-2.git grounded_sam_2"
fi

echo ""
echo "=== 验证安装 ==="
if [ -f "test_imports.py" ]; then
    python test_imports.py
else
    echo "test_imports.py 不存在，跳过验证"
fi

echo ""
echo "安装完成！"

