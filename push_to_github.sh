#!/bin/bash
# Git推送脚本
# 使用方法: ./push_to_github.sh YOUR_USERNAME REPO_NAME

set -e

# 检查参数
if [ $# -lt 2 ]; then
    echo "使用方法: $0 <GitHub用户名> <仓库名>"
    echo "示例: $0 albertyanyy sdf-processor"
    exit 1
fi

USERNAME=$1
REPO_NAME=$2
REPO_URL="https://github.com/${USERNAME}/${REPO_NAME}.git"

echo "=========================================="
echo "准备推送到GitHub仓库"
echo "=========================================="
echo "用户名: $USERNAME"
echo "仓库名: $REPO_NAME"
echo "仓库URL: $REPO_URL"
echo ""

# 检查是否已经在正确的目录
if [ ! -f "process_sdf.py" ]; then
    echo "错误: 请在 sdf_processor 目录下运行此脚本"
    exit 1
fi

# 检查是否已有远程仓库
if git remote get-url origin > /dev/null 2>&1; then
    CURRENT_URL=$(git remote get-url origin)
    echo "检测到已有远程仓库: $CURRENT_URL"
    read -p "是否要更新为新的URL? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote set-url origin "$REPO_URL"
        echo "✓ 已更新远程仓库URL"
    else
        echo "使用现有远程仓库: $CURRENT_URL"
        REPO_URL=$CURRENT_URL
    fi
else
    echo "添加远程仓库..."
    git remote add origin "$REPO_URL"
    echo "✓ 已添加远程仓库"
fi

# 将分支重命名为main（如果当前是master）
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" = "master" ]; then
    echo "将分支从 master 重命名为 main..."
    git branch -M main
    echo "✓ 分支已重命名"
fi

# 检查是否有未提交的更改
if ! git diff-index --quiet HEAD --; then
    echo "警告: 检测到未提交的更改"
    read -p "是否要提交这些更改? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        read -p "请输入提交信息: " COMMIT_MSG
        git commit -m "${COMMIT_MSG:-Update code}"
        echo "✓ 已提交更改"
    fi
fi

# 推送代码
echo ""
echo "正在推送到远程仓库..."
echo "如果提示输入用户名和密码，请使用GitHub用户名和Personal Access Token"
echo ""

git push -u origin main

echo ""
echo "=========================================="
echo "✓ 推送完成！"
echo "=========================================="
echo "访问你的仓库: https://github.com/${USERNAME}/${REPO_NAME}"
echo ""

