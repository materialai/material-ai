# 移动端 APK 构建指南

## 方案：Flet（基于Flutter引擎的Python移动框架）

将现有 Python 诊断引擎、审计引擎、SQLite 知识库完整打包为 APK，
可在华为鸿蒙（APK兼容）和安卓手机上运行。

## 文件结构

```
material-ai-diagnosis/
├── mobile_app.py          ← Flet 移动端主UI (入口)
├── mobile_init_db.py      ← 移动端知识库初始化
├── build_apk.sh           ← 本地构建脚本
├── .github/workflows/
│   └── build-apk.yml      ← GitHub Actions 自动构建
├── .gitignore             ← 排除敏感文件
├── .env.example           ← 配置模板（不含真实密钥）
└── flet_build.toml        ← Flet 项目配置
```

## ⚠️ 安全警告

**永远不要将 API Key 等敏感信息提交到 Git！**

- `.env` 文件包含真实密钥，已被 `.gitignore` 排除
- `.env.example` 仅为模板，使用占位符
- GitHub Actions 从 **Secrets** 环境变量获取密钥
- 如果发现密钥泄露，立即在火山引擎控制台重新生成

## 构建方式

### 方式一：GitHub Actions 自动构建（推荐）

#### 第一步：在 GitHub 仓库配置 Secrets

1. 进入 GitHub 仓库页面
2. 点击 **Settings** > **Secrets and variables** > **Actions**
3. 点击 **New repository secret**，添加以下 Secrets：

| Secret 名称 | 说明 | 示例值 |
|-------------|------|--------|
| `DOUBAO_API_KEY` | 豆包企业 API Key | `ark-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx-xxxxxxxxxx` |
| `DOUBAO_MODEL_ID` | 端点模型 ID | `ep-20241015190627-xxxxx` |

4. Secrets 页面在仓库 `Settings > Secrets and variables > Actions > New repository secret`

#### 第二步：推送代码，自动构建

1. 将代码推送到 `main` 分支
2. 进入 GitHub 仓库 > **Actions** 标签页
3. 选择 "Build APK" workflow，查看构建日志
4. 构建完成后，在 Artifacts 区域下载 APK

> 国内网络原因可能导致 Actions 失败，点击 "Re-run" 重新运行即可

#### Actions 构建失败排查

复制完整构建日志，发回分析即可定位。

### 方式二：本地 Linux/Docker 构建

```bash
# 1. 配置 API Key（手动创建 .env）
cp .env.example .env
# 编辑 .env 填入真实密钥

# 2. 安装依赖
pip install flet flet-cli

# 3. 初始化知识库
python mobile_init_db.py

# 4. 构建 APK
flet build apk \
  --output build/app.apk \
  --android-permission android.permission.INTERNET \
  --org com.materialai.diagnosis \
  --project material-ai-diagnosis
```

### 方式三：WSL (Windows Subsystem for Linux)

```powershell
# PowerShell
wsl --install Ubuntu
# 在 WSL Ubuntu 中执行方式二的步骤
```

### 方式四：Docker

```bash
docker run --rm -v $(pwd):/app -w /app python:3.12 bash -c "
  pip install flet flet-cli
  python mobile_init_db.py
  flet build apk --output build/app.apk
"
```

## API Key 安全机制

本仓库采用多层防护：

1. **文件隔离**：`.env` 在 `.gitignore` 中，不会被提交
2. **运行时读取**：所有代码通过 `os.getenv()` 读环境变量，从不硬编码
3. **Actions Secrets**：GitHub Actions 从 Secrets 读取密钥，写入文件后构建
4. **日志脱敏**：workflow 中只打印密钥长度，不打印内容
5. **校验占位符**：如果 API Key 为 `YOUR_API_KEY_HERE`，构建会失败，避免空 Key 打包进 APK

## 构建产物安全

生成的 APK 中会包含 `.env` 文件（因为运行时需要读取）。这是标准做法：
- APK 可以被反编译获取密钥（任何客户端软件都如此）
- 建议为每台设备或每个用户分配独立 API Key
- 在火山引擎控制台设置 RPM 限额，防止超额使用
- 如有更高安全需求，可配合后端代理服务

## Flet 打包原理

- Flet 基于 Flutter 引擎：Python 代码由 Flet 运行时解释执行
- SQLite 数据库随 APK 打包，首次启动复制到 app 私有目录
- 豆包 API 调用走 HTTPS，无需额外配置
- 审计 Agent、诊断引擎均运行在手机本地

## APK 系统要求

| 项目 | 要求 |
|------|------|
| Android 版本 | Android 5.0 (API 21) 及以上 |
| 鸿蒙 | HarmonyOS 2.0+（APK兼容层） |
| 网络 | 需要联网（豆包 API 调用） |
| 存储 | ~50MB 安装包 |

## 知识库更新

如需更新知识库内容：

```bash
# 1. 重新导入知识
python import_pcm_sqlite.py
python import_bulk_knowledge.py

# 2. 重新打包
python mobile_init_db.py
flet build apk --output build/app-updated.apk
```
