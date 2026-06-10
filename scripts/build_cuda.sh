#!/usr/bin/env bash
#
# scripts/build_cuda.sh — Build CUDA extensions for Linear and Softmax operators.
#
# Usage:
#   bash scripts/build_cuda.sh            # Build with default CUDA archs
#   bash scripts/build_cuda.sh clean      # Clean then build
#   TORCH_CUDA_ARCH_LIST="8.0;9.0" bash scripts/build_cuda.sh  # Custom archs
#
# Requirements:
#   - Linux with NVIDIA GPU
#   - CUDA Toolkit 12.1+
#   - PyTorch 2.1+ with CUDA support
#   - GCC 9+ / Clang 12+
#
# Environment variables:
#   CUDA_HOME            — Path to CUDA (default: /usr/local/cuda)
#   TORCH_CUDA_ARCH_LIST — Semicolon-separated CUDA archs (default: 7.0;7.5;8.0;8.6;8.9;9.0)
#   MAX_JOBS             — Parallel compilation jobs
#   DEBUG                — Set to 1 for debug build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CSRC_DIR="$PROJECT_DIR/csrc"

# ── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Clean ──────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "clean" ]]; then
    info "Cleaning build artifacts..."
    rm -rf "$CSRC_DIR/build"
    rm -f "$CSRC_DIR"/*.so
    rm -f "$CSRC_DIR"/linear/*.so
    rm -f "$CSRC_DIR"/softmax/*.so
    ok "Clean complete."
    [[ "${2:-}" != "build" ]] && exit 0
fi

# ── Environment Check ──────────────────────────────────────────────────────
info "Checking build environment..."

# Python
python_exe="${PYTHON:-python}"
if ! command -v "$python_exe" &>/dev/null; then
    error "Python not found. Set PYTHON env var."
fi
ok "Python: $($python_exe --version 2>&1)"

# PyTorch
if ! $python_exe -c "import torch; print(torch.__version__)" &>/dev/null; then
    error "PyTorch not installed."
fi
torch_ver=$($python_exe -c "import torch; print(torch.__version__)")
ok "PyTorch: $torch_ver"

if ! $python_exe -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" &>/dev/null; then
    warn "PyTorch CUDA not available. Extensions will be defined but may not compile."
else
    cuda_ver=$($python_exe -c "import torch; print(torch.version.cuda)")
    gpu_name=$($python_exe -c "import torch; print(torch.cuda.get_device_name(0))")
    ok "CUDA: $cuda_ver, GPU: $gpu_name"
fi

# CUDA Toolkit
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
    nvcc_ver=$("$CUDA_HOME/bin/nvcc" --version 2>&1 | grep "release" | awk '{print $6}' | tr -d ',')
    ok "NVCC: $nvcc_ver ($CUDA_HOME)"
else
    warn "NVCC not found at $CUDA_HOME. Set CUDA_HOME or install CUDA Toolkit."
fi

# CUDA architectures
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
    info "Using default CUDA archs: $TORCH_CUDA_ARCH_LIST"
else
    info "CUDA archs: $TORCH_CUDA_ARCH_LIST"
fi

# ── Build ──────────────────────────────────────────────────────────────────
echo ""
info "Building CUDA extensions..."
info "Source directory: $CSRC_DIR"

cd "$PROJECT_DIR"

# Install csrc dependencies needed for build
$python_exe -m pip install -q ninja 2>/dev/null || true

# Build extensions using csrc/setup.py
$python_exe "$CSRC_DIR/setup.py" build_ext --inplace 2>&1 | tee /tmp/cuda_build.log

# Check for .so files
so_files=$(find "$PROJECT_DIR" -maxdepth 2 -name "*.so" -type f 2>/dev/null || true)

if [[ -z "$so_files" ]]; then
    echo ""
    warn "No .so files found after build."
    warn "This is expected on Mac (use MLX backend)."
    warn "On Linux: check /tmp/cuda_build.log for errors."
else
    echo ""
    ok "Built extensions:"
    for so in $so_files; do
        echo "    $so"
    done
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Build Complete"
echo "=========================================="
echo ""
echo "Test with:"
echo "  python tests/test_linear.py --backend cuda"
echo "  python tests/test_softmax.py --backend cuda"
echo ""
echo "On Mac (MLX backend):"
echo "  OPS_BACKEND=mlx python tests/test_linear.py"
echo ""
