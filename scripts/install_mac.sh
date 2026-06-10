#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# scripts/install_mac.sh — Mac Apple Silicon 一键安装依赖
# ═══════════════════════════════════════════════════════════════════════════
# 适用环境：
#   - MacBook Air/Pro M1 / M2 / M3 / M4 / M5
#   - macOS 14.0+
#   - Xcode Command Line Tools 已安装 (xcode-select --install)
#
# 安装内容：
#   1. Python 3.10/3.11 虚拟环境（推荐）
#   2. PyTorch (MPS 后端，无 CUDA)
#   3. MLX / MLX-LM（Apple Metal GPU）
#   4. Transformers / FastAPI / 量化工具 / 测试框架
#   5. ✅ 跳过 vLLM / bitsandbytes（Mac 无法运行）
#
# 用法：
#   chmod +x scripts/install_mac.sh
#   bash scripts/install_mac.sh            # 装到当前 Python 环境
#   bash scripts/install_mac.sh --venv     # 创建 .venv 并安装（推荐）
#   bash scripts/install_mac.sh --dev      # 连开发工具一起装
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

USE_VENV=false
WITH_DEV=false

for arg in "$@"; do
    case "$arg" in
        --venv) USE_VENV=true ;;
        --dev)  WITH_DEV=true ;;
    esac
done

# ── 颜色 ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# ═══════════════════════════════════════════════════════════════════════════
# Step 0 — 环境检测
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo " Mac Apple Silicon 依赖安装"
echo "============================================"
echo ""

# 确认是 Apple Silicon
ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    warn "检测到架构=$ARCH（非 arm64）。此脚本只适用于 Apple Silicon Mac。"
    warn "如果是 Intel Mac 也继续，但没有 GPU 加速。"
fi
ok "架构: $ARCH"

# Python 版本
PYTHON="$(which python3 || echo '')"
if [[ -z "$PYTHON" ]]; then
    echo "❌ 未找到 python3，请先安装 Python 3.10+"
    echo "   brew install python@3.11"
    exit 1
fi
PY_VER="$($PYTHON --version 2>&1)"
ok "$PY_VER ($PYTHON)"

# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — 创建虚拟环境（可选）
# ═══════════════════════════════════════════════════════════════════════════
if $USE_VENV; then
    VENV_DIR="$PROJECT_DIR/.venv"
    if [[ ! -d "$VENV_DIR" ]]; then
        info "创建虚拟环境: $VENV_DIR"
        $PYTHON -m venv "$VENV_DIR"
        ok "虚拟环境已创建"
    else
        ok "虚拟环境已存在: $VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"
    PYTHON="$VENV_DIR/bin/python"
    ok "已激活虚拟环境"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — 升级 pip
# ═══════════════════════════════════════════════════════════════════════════
info "升级 pip..."
$PYTHON -m pip install --upgrade pip -q
ok "pip $($PYTHON -m pip --version | awk '{print $2}')"

# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — 安装 PyTorch（Mac MPS 版，无 CUDA）
# ═══════════════════════════════════════════════════════════════════════════
info "安装 PyTorch (MPS backend, cpu-only)..."
$PYTHON -m pip install \
    torch>=2.3.0 \
    torchvision>=0.18.0 \
    torchaudio>=2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    -q

# 验证 MPS 是否可用
$PYTHON -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  MPS available: {torch.backends.mps.is_available()}')
print(f'  MPS built:     {torch.backends.mps.is_built()}')
" 2>&1 | while read line; do ok "$line"; done

# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — 安装 MLX（Apple Metal GPU 加速）
# ═══════════════════════════════════════════════════════════════════════════
info "安装 MLX (Apple Metal 后端)..."
$PYTHON -m pip install "mlx>=0.29.4" "mlx-lm>=0.29.0" -q

$PYTHON -c "
import mlx.core as mx
print(f'  MLX:      {mx.__version__}')
print(f'  Metal:    {mx.metal.is_available()}')
print(f'  GPU:      {mx.metal.get_device_name()}')
" 2>&1 | while read line; do ok "$line"; done

# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — 安装公共依赖
# ═══════════════════════════════════════════════════════════════════════════
info "安装公共依赖..."
$PYTHON -m pip install \
    -r "$PROJECT_DIR/requirements.txt" \
    -r "$PROJECT_DIR/requirements-mac.txt" \
    -q

# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — 安装开发工具（可选）
# ═══════════════════════════════════════════════════════════════════════════
if $WITH_DEV; then
    info "安装开发工具..."
    $PYTHON -m pip install -r "$PROJECT_DIR/requirements-dev.txt" -q
    ok "black + ruff + mypy + pre-commit"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 7 — 验证
# ═══════════════════════════════════════════════════════════════════════════
echo ""
info "验证已安装的包..."

$PYTHON -c "
import torch;       print(f'  torch         {torch.__version__}')
import mlx.core;    print(f'  mlx           {mlx.core.__version__}')
import transformers; print(f'  transformers  {transformers.__version__}')
import fastapi;     print(f'  fastapi       {fastapi.__version__}')
import PIL;         print(f'  pillow        {PIL.__version__}')
import numpy;       print(f'  numpy         {numpy.__version__}')
" 2>&1 | while read line; do
    if echo "$line" | grep -q "Error\|ModuleNotFoundError"; then
        warn "$line"
    else
        ok "$line"
    fi
done

# ── 验证我们自己的 MLX 算子 ──────────────────────────────────────────────
echo ""
info "验证项目 MLX 算子..."
cd "$PROJECT_DIR"
$PYTHON -c "
import sys; sys.path.insert(0, '.')
import mlx.core as mx
import numpy as np

# Linear
from csrc.linear.linear_mlx import tiled_gemm, tiled_gemm_bt, linear_bias_gelu
C = tiled_gemm(mx.random.normal((4,256)), mx.random.normal((256,512)))
assert C.shape == (4,512), f'GEMM shape error: {C.shape}'
print('  ✓ tiled_gemm')

C = tiled_gemm_bt(mx.random.normal((4,256)), mx.random.normal((512,256)))
assert C.shape == (4,512)
print('  ✓ tiled_gemm_bt')

C = linear_bias_gelu(mx.random.normal((4,256)), mx.random.normal((256,1024)), mx.random.normal((1024,)))
assert C.shape == (4,1024)
print('  ✓ linear_bias_gelu')

# Softmax
from csrc.softmax.softmax_mlx import online_safe_softmax, causal_softmax, softmax_backward
p = online_safe_softmax(mx.random.normal((16,512)))
assert mx.allclose(p.sum(axis=-1), mx.ones(16), rtol=1e-4)
print('  ✓ online_safe_softmax')

probs = causal_softmax(mx.random.normal((2,4,32,32)))
upper = np.array(probs[0,0])
assert np.all(np.triu(upper, k=1) == 0)
print('  ✓ causal_softmax')

dx = softmax_backward(mx.softmax(mx.random.normal((8,256)), axis=-1), mx.random.normal((8,256)))
assert mx.allclose(dx.sum(axis=-1), mx.zeros(8), atol=1e-5)
print('  ✓ softmax_backward')
" 2>&1 | while read line; do
    if echo "$line" | grep -q "✓"; then
        ok "$line"
    elif echo "$line" | grep -q "Error\|Traceback\|assert\|FAIL"; then
        warn "$line"
    else
        echo "     $line"
    fi
done

# ═══════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo -e " ${GREEN}Mac 环境安装完成！${NC}"
echo "============================================"
echo ""
echo "  运行测试:"
echo "    python -m pytest tests/test_linear.py -v"
echo "    python -m pytest tests/test_softmax.py -v"
echo "    python -m pytest tests/test_quantization.py -v"
echo ""
echo "  本地调试 MLX 算子:"
echo "    python csrc/linear/linear_mlx.py"
echo "    python csrc/softmax/softmax_mlx.py"
echo ""
if $USE_VENV; then
    echo "  下次激活虚拟环境:"
    echo "    source .venv/bin/activate"
    echo ""
fi
echo "  ❌ vLLM / bitsandbytes 未安装（仅限 Linux + NVIDIA）"
echo "  ❌ CUDA 扩展未编译（需 Linux 服务器）"
echo ""
