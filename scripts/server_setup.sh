#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# scripts/server_setup.sh — 服务器一键环境配置 + vLLM 升级
# ═══════════════════════════════════════════════════════════════════════════
# 运行方式（在服务器上）：
#   source .venv/bin/activate
#   bash scripts/server_setup.sh
#
# 做什么：
#   1. 检测 CUDA 版本 & GPU 信息
#   2. 升级 vLLM 到 >=0.5.0（自动判断是否需要源码编译）
#   3. 安装缺失依赖
#   4. 打印最终环境摘要
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  服务器环境检测 & vLLM 升级                      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Step 1：硬件 & CUDA ──────────────────────────────────────────────────
info "【Step 1】硬件 & CUDA 检测"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || err "nvidia-smi 未找到"
echo ""

# CUDA 版本
CUDA_VER="unknown"
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version 2>&1 | grep 'release' | awk '{print $6}' | tr -d ',' || echo "unknown")
fi
ok "CUDA 版本: $CUDA_VER"

# Python
PY_VER=$(python --version 2>&1)
ok "Python: $PY_VER"

# ═══════════════════════════════════════════════════════════════════════════
# Step 2：升级 pip
# ═══════════════════════════════════════════════════════════════════════════
info "【Step 2】升级 pip + setuptools + wheel"
pip install --upgrade pip setuptools wheel -q 2>&1 | tail -1
ok "pip $(pip --version | awk '{print $2}')"

# ═══════════════════════════════════════════════════════════════════════════
# Step 3：升级 vLLM
# ═══════════════════════════════════════════════════════════════════════════
info "【Step 3】升级 vLLM"

CURRENT_VLLM=$(pip show vllm 2>/dev/null | grep Version | awk '{print $2}' || echo "not-installed")
info "  当前版本: $CURRENT_VLLM"

# 判断 CUDA 版本，决定安装策略
CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

if [[ "$CUDA_VER" == "unknown" ]]; then
    warn "  无法确定 CUDA 版本，尝试直接安装 vLLM 0.6.0+"
    pip install "vllm>=0.5.4" 2>&1 | tail -5
elif [[ "$CUDA_MAJOR" -ge 12 ]]; then
    ok "  CUDA >= 12.x，直接安装预编译 vLLM"
    pip install "vllm>=0.5.4" 2>&1 | tail -5
else
    warn "  CUDA $CUDA_VER < 12.0，vLLM >=0.4.0 无预编译包"
    warn "  尝试两种方案..."
    echo ""
    info "  方案 A：安装 vLLM 0.3.3（最后一个 CUDA 11.8 预编译版）"
    if pip install vllm==0.3.3 2>&1 | tail -3; then
        ok "  vLLM 0.3.3 安装成功（GPTQ 支持有限但可用）"
    else
        warn "  方案 A 失败"
        info "  方案 B：从源码编译 vLLM（需要 15-30 分钟 + 16GB 内存）"
        pip install "vllm>=0.5.4" --no-build-isolation 2>&1 | tail -10 || {
            warn "  源码编译也失败了。使用 vLLM 0.3.3 或升级 CUDA到12.1+"
        }
    fi
fi

# 验证
VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "FAILED")
if [[ "$VLLM_VER" != "FAILED" ]]; then
    ok "vLLM 最终版本: $VLLM_VER"
else
    err "vLLM 安装失败！请手动检查"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Step 4：补充依赖
# ═══════════════════════════════════════════════════════════════════════════
info "【Step 4】安装 / 升级其他依赖"
pip install -q \
    "transformers>=4.40.0" \
    "accelerate>=0.28.0" \
    "safetensors>=0.4.0" \
    "einops>=0.7.0" \
    "tqdm>=4.66.0" \
    "numpy>=1.24.0" \
    2>&1 | tail -1

# bitsandbytes（量化推理时需要）
pip install -q "bitsandbytes>=0.41.0" 2>&1 | tail -1 || warn "bitsandbytes 安装失败（非致命）"

ok "依赖安装完成"

# ═══════════════════════════════════════════════════════════════════════════
# Step 5：最终环境摘要
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  环境摘要                                         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
python -c "
import torch;          print(f'  PyTorch:       {torch.__version__}  (CUDA {torch.version.cuda})')
print(f'  GPU count:     {torch.cuda.device_count()}')
print(f'  GPU 0:         {torch.cuda.get_device_name(0)}')
print(f'  GPU memory:    {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
" 2>&1

python -c "
import vllm;          print(f'  vLLM:          {vllm.__version__}')
import transformers;  print(f'  Transformers:  {transformers.__version__}')
import accelerate;    print(f'  Accelerate:    {accelerate.__version__}')
import einops;        print(f'  Einops:        {einops.__version__}')
" 2>&1

echo ""
echo "  下一步："
echo "    python scripts/int4_infer.py"
echo "    python scripts/benchmark_fp16_vs_int4.py"
echo ""
