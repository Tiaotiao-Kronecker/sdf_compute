# Git 仓库创建和推送完整步骤

## 步骤1：在GitHub上创建新仓库

1. 登录你的GitHub账号：https://github.com
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `sdf-processor` (或你喜欢的名称)
   - **Description**: `Surface Distance Field (SDF) processor based on Depth-Anything-3`
   - **Visibility**: 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为我们已经有了）
   - **不要**添加 .gitignore 或 license（我们已经有了）
4. 点击 "Create repository"

## 步骤2：获取仓库URL

创建完成后，GitHub会显示仓库URL，格式如下：
- HTTPS: `https://github.com/你的用户名/sdf-processor.git`
- SSH: `git@github.com:你的用户名/sdf-processor.git`

**推荐使用HTTPS**（更简单，不需要配置SSH密钥）

## 步骤3：在本地设置远程仓库并推送

### 方法A：使用HTTPS（推荐）

```bash
cd /home/user/projects/SDF/sdf_processor

# 添加远程仓库（将 YOUR_USERNAME 和 REPO_NAME 替换为你的实际值）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 将分支重命名为 main（GitHub默认使用main）
git branch -M main

# 推送代码到远程仓库
git push -u origin main
```

**示例**（假设你的用户名是 `albertyanyy`，仓库名是 `sdf-processor`）：
```bash
git remote add origin https://github.com/albertyanyy/sdf-processor.git
git branch -M main
git push -u origin main
```

### 方法B：使用SSH（需要配置SSH密钥）

如果你已经配置了SSH密钥：

```bash
cd /home/user/projects/SDF/sdf_processor

# 添加远程仓库（将 YOUR_USERNAME 和 REPO_NAME 替换为你的实际值）
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# 将分支重命名为 main
git branch -M main

# 推送代码到远程仓库
git push -u origin main
```

## 步骤4：验证推送结果

推送成功后，访问你的GitHub仓库页面，应该能看到所有文件。

## 后续更新代码

以后修改代码后，使用以下命令推送：

```bash
cd /home/user/projects/SDF/sdf_processor

# 查看修改的文件
git status

# 添加修改的文件
git add .

# 提交修改
git commit -m "描述你的修改内容"

# 推送到远程
git push
```

## 常见问题

### 1. 如果推送时要求输入用户名和密码

HTTPS方式可能需要输入GitHub用户名和密码（或Personal Access Token）。

**解决方案**：
- 使用Personal Access Token代替密码：
  1. 访问：https://github.com/settings/tokens
  2. 点击 "Generate new token (classic)"
  3. 选择权限：至少勾选 `repo`
  4. 生成token并复制
  5. 推送时，密码处输入这个token

### 2. 如果远程仓库已存在文件

如果远程仓库已经有README等文件，需要先拉取：

```bash
git pull origin main --allow-unrelated-histories
# 解决可能的冲突后
git push -u origin main
```

### 3. 查看远程仓库配置

```bash
git remote -v
```

### 4. 修改远程仓库URL

如果URL设置错误，可以修改：

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

## 快速命令模板

将以下命令中的 `YOUR_USERNAME` 和 `REPO_NAME` 替换为你的实际值：

```bash
cd /home/user/projects/SDF/sdf_processor
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

