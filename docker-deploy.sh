#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# docker-deploy.sh — Mac 本地构建 Docker 镜像 + 推送到服务器
# ═══════════════════════════════════════════════════════════════════════════
#
# 前置条件：
#   - Docker Desktop 已安装并启动（菜单栏绿色 Running）
#   - Docker Desktop → Settings → General → ✅ Use Rosetta for x86/amd64
#   - Docker Desktop → Settings → Docker Engine → registry-mirrors 已配置
#
# 用法：
#   bash docker-deploy.sh build      # 仅构建镜像
#   bash docker-deploy.sh test-build # 干跑测试（快速验证构建流程）
#   bash docker-deploy.sh export     # 导出 tar.gz
#   bash docker-deploy.sh upload     # 上传到服务器
#   bash docker-deploy.sh all        # 一键全流程
#
# 环境变量：
#   IMAGE_TAG=v1.0                  # 镜像标签
#   SERVER_USER=neu                 # SSH 用户
#   SERVER_HOST=219.216.65.31       # SSH 主机
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cuda-multimodal-opt}"
IMAGE_TAG="${IMAGE_TAG:-v1.0}"
FULL_TAG="${IMAGE_NAME}:${IMAGE_TAG}"
EXPORT_FILE="${IMAGE_NAME}-${IMAGE_TAG}.tar.gz"
SERVER_USER="${SERVER_USER:-neu}"
SERVER_HOST="${SERVER_HOST:-219.216.65.31}"
SERVER_PATH="${SERVER_PATH:-/home/${SERVER_USER}/cuda-multimodal-optimization}"
MODELS_PATH="${MODELS_PATH:-/home/${SERVER_USER}/cuda-multimodal-optimization/models}"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

check_prereqs() {
    info "检查前置条件..."
    if ! command -v docker &>/dev/null; then
        err "Docker 未安装。brew install --cask docker"
        exit 1
    fi
    if ! docker info &>/dev/null; then
        err "Docker daemon 未运行。请启动 Docker Desktop。"
        exit 1
    fi
    ok "Docker: $(docker --version | cut -d',' -f1)"
    if docker buildx version &>/dev/null; then
        ok "buildx: $(docker buildx version | head -1 | awk '{print $2}')"
    else
        warn "buildx 不可用，将用传统 docker build（可能架构不匹配）"
    fi
    cd "$PROJECT_DIR"
    ok "工作目录: $PROJECT_DIR"
}

# ═══════════════════════════════════════════════════════════════════════════
# 构建镜像
# ═══════════════════════════════════════════════════════════════════════════
step_build() {
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Step 1：构建 Docker 镜像（amd64 → 服务器）        ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
    info "镜像: $FULL_TAG"
    info "目标平台: linux/amd64（交叉编译，Rosetta 模拟）"
    info "预计耗时: 首次 15-25 分钟，增量构建 2-5 分钟"
    echo ""

    cd "$PROJECT_DIR"

    # 构建参数说明：
    #   --platform linux/amd64  → 为目标 x86_64 服务器构建
    #   --load                  → 加载到本地镜像仓库
    #   --progress=plain        → 显示完整构建日志（便于排查）
    docker buildx build \
        --platform linux/amd64 \
        --tag "$FULL_TAG" \
        --progress=plain \
        --load \
        . 2>&1 | tee /tmp/docker-build.log

    local exit_code=${PIPESTATUS[0]}
    if [[ "$exit_code" -ne 0 ]]; then
        err "构建失败！查看日志: /tmp/docker-build.log"
        exit "$exit_code"
    fi

    ok "镜像构建完成: $FULL_TAG"
    echo ""
    docker images "$IMAGE_NAME" --no-trunc
}

# ═══════════════════════════════════════════════════════════════════════════
# 干跑测试（快速验证 .dockerignore + Dockerfile 语法）
# ═══════════════════════════════════════════════════════════════════════════
step_test_build() {
    echo ""
    info "干跑测试：只执行到 FROM 阶段结束，验证 Dockerfile 语法 + 基础镜像可达性..."
    echo ""
    # 仅拉取最外层镜像并验证 Dockerfile 语法
    docker buildx build \
        --platform linux/amd64 \
        --target runtime \
        --progress=plain \
        --no-cache \
        . 2>&1 | head -80

    ok "语法检查通过（完整构建用 bash docker-deploy.sh build）"
}

# ═══════════════════════════════════════════════════════════════════════════
# 导出 tar.gz
# ═══════════════════════════════════════════════════════════════════════════
step_export() {
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Step 2：导出镜像为压缩包                          ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""

    if ! docker image inspect "$FULL_TAG" &>/dev/null; then
        err "镜像 $FULL_TAG 不存在，请先 build"
        exit 1
    fi

    info "导出: $FULL_TAG → $EXPORT_FILE"
    info "预计文件大小: 2.5-5 GB（运行时镜像 + Python 包）"
    echo ""

    docker save "$FULL_TAG" | gzip > "$EXPORT_FILE"

    FILE_SIZE=$(ls -lh "$EXPORT_FILE" | awk '{print $5}')
    ok "导出完成: $EXPORT_FILE ($FILE_SIZE)"
}

# ═══════════════════════════════════════════════════════════════════════════
# 上传到服务器
# ═══════════════════════════════════════════════════════════════════════════
step_upload() {
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Step 3：上传镜像到服务器                          ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""

    if [[ ! -f "$EXPORT_FILE" ]]; then
        err "文件不存在: $EXPORT_FILE"
        exit 1
    fi

    info "目标: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/"
    info "文件大小: $(ls -lh "$EXPORT_FILE" | awk '{print $5}')"
    echo ""

    scp "$EXPORT_FILE" "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/"
    ok "镜像已上传"

    scp docker-deploy.sh "${SERVER_USER}@${SERVER_HOST}:${SERVER_PATH}/"
    ok "部署脚本已上传"
}

# ═══════════════════════════════════════════════════════════════════════════
# 一键全流程
# ═══════════════════════════════════════════════════════════════════════════
step_all() {
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  一键部署：构建 → 导出 → 上传 → 服务器加载         ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""

    check_prereqs
    step_build
    step_export
    step_upload

    echo ""
    info "在服务器上加载镜像并运行..."
    echo ""

    ssh "${SERVER_USER}@${SERVER_HOST}" "bash -s" << REMOTE_SCRIPT
set -e
cd ${SERVER_PATH}
echo ''
echo '=== [服务器] 加载镜像 ==='
docker load < ${IMAGE_NAME}-${IMAGE_TAG}.tar.gz
echo ''
echo '=== [服务器] 验证镜像 ==='
docker images ${FULL_TAG}
echo ''
echo '=== [服务器] GPU 冒烟测试 ==='
docker run --rm --gpus all ${FULL_TAG} python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA: {torch.version.cuda}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB')
print('冒烟测试通过!')
"
echo ''
echo '=== 部署成功 ==='
echo ''
echo '运行压测:'
echo '  docker run --rm --gpus all \'
echo '    -v ${MODELS_PATH}:/app/models \'
echo '    -v ${SERVER_PATH}/outputs:/app/outputs \'
echo '    -e FP16_MODEL_PATH=/app/models/llama-fp16-text \'
echo '    -e INT4_MODEL_PATH=/app/models/llava-llama-int4 \'
echo "    ${FULL_TAG} python scripts/benchmark_fp16_vs_int4.py"
echo ''
REMOTE_SCRIPT

    ok "一键部署全部完成！"
}

# ═══════════════════════════════════════════════════════════════════════════
# 帮助
# ═══════════════════════════════════════════════════════════════════════════
show_help() {
    cat << EOF
用法: bash docker-deploy.sh <command>

命令:
  build        构建 Docker 镜像（Mac 本地，amd64 交叉编译）
  test-build   仅验证 Dockerfile 语法和基础镜像可达性
  export       导出镜像为 tar.gz
  upload       上传镜像到服务器（SCP）
  all          一键全流程（构建 + 导出 + 上传 + 远程加载 + 测试）

环境变量（可选）:
  IMAGE_TAG    镜像标签（默认 v1.0）
  SERVER_USER  SSH 用户（默认 neu）
  SERVER_HOST  SSH 主机（默认 219.216.65.31）

示例:
  bash docker-deploy.sh build
  IMAGE_TAG=v2.0 bash docker-deploy.sh all
EOF
}

# ═══════════════════════════════════════════════════════════════════════════
cd "$PROJECT_DIR"

case "${1:-help}" in
    build)      check_prereqs; step_build ;;
    test-build) check_prereqs; step_test_build ;;
    export)     check_prereqs; step_export ;;
    upload)     check_prereqs; step_upload ;;
    all)        step_all ;;
    help|*)     show_help ;;
esac
