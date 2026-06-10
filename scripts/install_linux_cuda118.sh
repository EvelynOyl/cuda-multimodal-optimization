#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# scripts/install_linux_cuda118.sh — Linux + CUDA 11.8 一键安装依赖
# ═══════════════════════════════════════════════════════════════════════════
# 适用环境：
#   - Ubuntu 22.04 / 20.04
#   - NVIDIA GPU (A100, A6000, RTX 3090, RTX 4090, H100, ...)
#   - CUDA 11.8 已安装 (nvcc --version 确认)
#   - cuDNN 8.x 已安装
#
# 安装内容：
#   1. PyTorch 2.1.2 + CUDA 11.8（torch 2.2+ 不支持 CUDA 11.8）
#   2. vLLM 0.3.3（最后一个 CUDA 11.8 预编译版）
#   3. bitsandbytes 0.41.3（INT4/INT8 量化）
#   4. Transformers / FastAPI / 量化工具 / 测试
#   5. （可选）Flash Attention 2 → 源码编译
#   6. （可选）CUDA 扩展编译 → csrc/setup.py
#
# 用法：
#   chmod +x scripts/install_linux_cuda118.sh
#   bash scripts/install_linux_cuda118.sh              # 装到当前 Python
#   bash scripts/install_linux_cuda118.sh --venv       # venv 环境（推荐）
#   bash scripts/install_linux_cuda118.sh --dev        # 连开发工具
#   bash scripts/install_linux_cuda118.sh --flash-attn # 编译 Flash Attention
#   bash scripts/install_linux_cuda118.sh --all        # venv + dev + flash-attn
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 命令行参数 ────────────────────────────────────────────────────────────
USE_VENV=false
WITH_DEV=false
WITH_FLASH_ATTN=false

for arg in "$@"; do
    case "$arg" in
        --venv)       USE_VENV=true ;;
        --dev)        WITH_DEV=true ;;
        --flash-attn) WITH_FLASH_ATTN=true ;;
        --all)
            USE_VENV=true
            WITH_DEV=true
            WITH_FLASH_ATTN=true
            ;;
    esac
done

# ── 颜色 ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ═══════════════════════════════════════════════════════════════════════════
# Step 0 — 环境检测
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "============================================"
echo " Linux + CUDA 11.8 依赖安装"
echo "============================================"
echo ""

# ── Python ────────────────────────────────────────────────────────────────
PYTHON="$(which python3 || which python || echo '')"
if [[ -z "$PYTHON" ]]; then
    err "未找到 Python。请安装 Python 3.10"
    echo "   sudo apt install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi
PY_VER="$($PYTHON --version 2>&1)"
ok "$PY_VER ($PYTHON)"

PY_MAJOR_MINOR="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ ! "$PY_MAJOR_MINOR" =~ ^3\.(10|11)$ ]]; then
    warn "推荐 Python 3.10 或 3.11（当前: $PY_MAJOR_MINOR）"
    warn "vLLM / PyTorch 2.1 对 Python 3.12 支持有限"
fi

# ── NVIDIA 驱动 ───────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi 未找到。请先安装 NVIDIA 驱动。"
    exit 1
fi
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
ok "GPU: $GPU_INFO"

# ── CUDA 版本 ─────────────────────────────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
    warn "nvcc 未找到，如果 CUDA 11.8 安装在非标准路径，设置 CUDA_HOME"
    warn "  export CUDA_HOME=/usr/local/cuda-11.8"
fi

if command -v nvcc &>/dev/null; then
    NVCC_VER="$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | tr -d ',')"
    ok "NVCC: $NVCC_VER"
    if [[ ! "$NVCC_VER" =~ ^11\.8 ]]; then
        warn "NVCC 版本 = $NVCC_VER，不是 11.8。安装可能不完全兼容。"
    fi
fi

# ── GCC ────────────────────────────────────────────────────────────────────
GCC_VER="$(gcc --version 2>/dev/null | head -1 || echo 'not found')"
ok "GCC: $GCC_VER"

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
# Step 2 — 升级 pip / setuptools / wheel
# ═══════════════════════════════════════════════════════════════════════════
info "升级 pip + setuptools + wheel..."
$PYTHON -m pip install --upgrade pip setuptools wheel -q
ok "pip $($PYTHON -m pip --version | awk '{print $2}')"

# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — 安装 PyTorch 2.1.2 + CUDA 11.8
# ═══════════════════════════════════════════════════════════════════════════
# torch 2.2+ 已移除 CUDA 11.8 预编译包。只能用 2.1.x。
# 如果未来升级到 CUDA 12.1+，可以装 torch>=2.5.0。
info "安装 PyTorch 2.1.2 (CUDA 11.8)..."
$PYTHON -m pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118 \
    -q

$PYTHON -c "
import torch
print(f'  PyTorch:     {torch.__version__}')
print(f'  CUDA:        {torch.version.cuda}')
print(f'  cuDNN:       {torch.backends.cudnn.version()}')
print(f'  GPU count:   {torch.cuda.device_count()}')
print(f'  GPU 0:       {torch.cuda.get_device_name(0)}')
print(f'  GPU memory:  {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
" 2>&1 | while read line; do ok "$line"; done

# ── 快速功能验证 ──────────────────────────────────────────────────────────
$PYTHON -c "
import torch
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')
c = a @ b
torch.cuda.synchronize()
print('  ✓ GPU matmul 1000x1000 正常')
" 2>&1 | while read line; do
    if echo "$line" | grep -q "✓"; then ok "$line"; else warn "$line"; fi
done

# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — 安装 vLLM 0.3.3
# ═══════════════════════════════════════════════════════════════════════════
info "安装 vLLM 0.3.3 (CUDA 11.8 预编译)..."
$PYTHON -m pip install vllm==0.3.3 -q

$PYTHON -c "
import vllm
print(f'  vLLM: {vllm.__version__}')
" 2>&1 | while read line; do ok "$line"; done

# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — 安装 bitsandbytes + xformers
# ═══════════════════════════════════════════════════════════════════════════
info "安装 bitsandbytes 0.41.3 (CUDA 11.8)..."
$PYTHON -m pip install bitsandbytes==0.41.3 -q

info "安装 xformers 0.0.23 (CUDA 11.8)..."
$PYTHON -m pip install xformers==0.0.23.post1 -q 2>&1 | tail -1 || warn "xformers 安装失败（非必需）"

# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — 安装 Flash Attention 2（可选，源码编译）
# ═══════════════════════════════════════════════════════════════════════════
if $WITH_FLASH_ATTN; then
    echo ""
    info "编译 Flash Attention 2（需要 ~10-15 分钟）..."
    info "如果卡住，Ctrl+C 跳过（Flash Attention 非必需）"

    # 确认有足够内存编译（需要 16GB+）
    MEM_GB="$(free -g 2>/dev/null | awk '/Mem:/{print $2}' || echo 0)"
    if [[ "$MEM_GB" -lt 16 ]]; then
        warn "系统内存 < 16 GB，编译可能失败。跳过。"
    else
        $PYTHON -m pip install ninja packaging -q
        $PYTHON -m pip install flash-attn==2.3.6 --no-build-isolation -q 2>&1 | tail -5 || {
            warn "Flash Attention 编译失败（非必需，vLLM 有内置 xformers 后备）"
        }
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 7 — 安装公共 + Linux 专属依赖
# ═══════════════════════════════════════════════════════════════════════════
info "安装公共依赖..."
$PYTHON -m pip install \
    -r "$PROJECT_DIR/requirements.txt" \
    -r "$PROJECT_DIR/requirements-linux-cuda118.txt" \
    -q

# ═══════════════════════════════════════════════════════════════════════════
# Step 8 — 开发工具（可选）
# ═══════════════════════════════════════════════════════════════════════════
if $WITH_DEV; then
    info "安装开发工具..."
    $PYTHON -m pip install -r "$PROJECT_DIR/requirements-dev.txt" -q
    ok "black + ruff + mypy + pre-commit"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 9 — 验证
# ═══════════════════════════════════════════════════════════════════════════
echo ""
info "────────── 依赖检查 ──────────"

# 核心包版本检查
$PYTHON -c "
packages = {
    'torch':        None,
    'torchvision':  None,
    'transformers': None,
    'accelerate':   None,
    'vllm':         None,
    'fastapi':      None,
    'PIL':          'pillow',
    'numpy':        None,
    'bitsandbytes': None,
    'datasets':     None,
    'einops':       None,
    'tqdm':         None,
    'yaml':         'pyyaml',
    'aiohttp':      None,
    'pydantic':     None,
}
for mod, name in packages.items():
    try:
        pkg = __import__(mod)
        ver = getattr(pkg, '__version__', '?')
        print(f'  ✓ {name or mod:20s} {ver}')
    except ImportError:
        print(f'  ✗ {name or mod:20s} NOT FOUND')
" 2>&1 | while read line; do
    if echo "$line" | grep -q "✓"; then ok "$line"; else warn "$line"; fi
done

# ── vLLM GPU 测试 ─────────────────────────────────────────────────────────
echo ""
info "vLLM GPU 检测..."
$PYTHON -c "
from vllm.utils import get_gpu_memory
try:
    mem = get_gpu_memory()
    for gpu_id, (total, free) in enumerate(mem.items()):
        print(f'  GPU {gpu_id}: {total:.1f} GB total, {free:.1f} GB free')
except Exception as e:
    print(f'  (跳过) {e}')
" 2>&1 | while read line; do ok "$line"; done

# ═══════════════════════════════════════════════════════════════════════════
# Step 10 — 编译项目 CUDA 扩展（可选提示）
# ═══════════════════════════════════════════════════════════════════════════
echo ""
info "下一步 — 编译项目 CUDA 扩展:"
echo ""
echo "    bash scripts/build_cuda.sh"
echo ""
info "或者直接启动 vLLM 服务:"
echo ""
echo "    bash scripts/run_server.sh"
echo "    bash scripts/run_server.sh --int4    # INT4 量化版"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════
echo "============================================"
echo -e " ${GREEN}Linux + CUDA 11.8 环境安装完成！${NC}"
echo "============================================"
echo ""
echo "  包版本汇总:"
echo "    PyTorch      2.1.2+cu118  （CUDA 11.8 最后一个支持版）"
echo "    vLLM         0.3.3         （CUDA 11.8 最后一个预编译版）"
echo "    bitsandbytes 0.41.3        （CUDA 11.8 最后一个支持版）"
echo "    xformers     0.0.23        （可选注意力加速）"
echo ""
echo "  💡 如果想获得最新特性（vLLM 0.6+, torch 2.5+），联系管理员升级到 CUDA 12.1+"
echo ""
