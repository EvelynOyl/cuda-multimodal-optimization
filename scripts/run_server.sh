#!/usr/bin/env bash
#
# scripts/run_server.sh — Start the LLaVA-1.5 + vLLM inference server.
#
# Usage:
#   bash scripts/run_server.sh                     # Default config
#   bash scripts/run_server.sh --port 8080         # Custom port
#   bash scripts/run_server.sh --int4              # Load INT4 quantized model
#   bash scripts/run_server.sh --help              # Show all options
#
# Requirements:
#   - NVIDIA GPU with sufficient VRAM (7B model: ~14GB FP16, ~5GB INT4)
#   - vLLM installed (pip install vllm)
#   - CUDA 12.1+
#
# Environment variables:
#   MODEL_NAME               — HuggingFace model ID or path
#   VLLM_PORT                — Server port (default: 8000)
#   GPU_MEMORY_UTILIZATION   — Fraction of GPU memory to use (default: 0.90)
#   QUANTIZATION             — Quantization method (none/int4/awq)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ───────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-liuhaotian/llava-v1.5-7b}"
VLLM_PORT="${VLLM_PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
QUANTIZATION="${QUANTIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# ── Parse Args ─────────────────────────────────────────────────────────────
usage() {
    cat << EOF
Usage: bash scripts/run_server.sh [OPTIONS]

Options:
  --model MODEL             Model name or path (default: liuhaotian/llava-v1.5-7b)
  --port PORT               Server port (default: 8000)
  --host HOST               Bind address (default: 0.0.0.0)
  --int4                    Load INT4 quantized model
  --awq                     Load AWQ quantized model
  --gpu-mem FRACTION        GPU memory fraction (default: 0.90)
  --max-len LEN             Max model length (default: 4096)
  --max-seqs N              Max concurrent sequences (default: 256)
  --no-continuous-batching  Disable continuous batching
  --no-prefix-caching       Disable prefix caching
  --help                    Show this help

Examples:
  bash scripts/run_server.sh                              # Default FP16 server
  bash scripts/run_server.sh --int4                       # INT4 quantized
  bash scripts/run_server.sh --model llava-v1.5-13b --port 9090
  bash scripts/run_server.sh --gpu-mem 0.80               # Conservative memory

EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)               MODEL_NAME="$2"; shift 2 ;;
        --port)                VLLM_PORT="$2"; shift 2 ;;
        --host)                HOST="$2"; shift 2 ;;
        --int4)                QUANTIZATION="int4"; shift ;;
        --awq)                 QUANTIZATION="awq"; shift ;;
        --gpu-mem)             GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max-len)             MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-seqs)            MAX_NUM_SEQS="$2"; shift 2 ;;
        --no-continuous-batching) CONTINUOUS_FLAG="--disable-continuous-batching"; shift ;;
        --no-prefix-caching)   PREFIX_FLAG="--disable-prefix-caching"; shift ;;
        --help)                usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Colors ─────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Check Environment ──────────────────────────────────────────────────────
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  LLaVA-1.5 + vLLM Inference Server${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "  Model:           ${GREEN}$MODEL_NAME${NC}"
echo -e "  Quantization:    ${GREEN}${QUANTIZATION:-none}${NC}"
echo -e "  Port:            ${GREEN}$VLLM_PORT${NC}"
echo -e "  GPU Memory:      ${GREEN}$GPU_MEMORY_UTILIZATION${NC}"
echo -e "  Max Seq Len:     ${GREEN}$MAX_MODEL_LEN${NC}"
echo -e "  Max Seqs:        ${GREEN}$MAX_NUM_SEQS${NC}"
echo ""

# Check for GPU
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true
    echo ""
fi

# Check if vLLM is installed
python -c "import vllm; print(f'  vLLM version: {vllm.__version__}')" 2>/dev/null || {
    echo -e "${YELLOW}[WARN] vLLM not found. Install with: pip install vllm${NC}"
}

# ── Quantized Model Path ───────────────────────────────────────────────────
if [[ "$QUANTIZATION" == "int4" ]]; then
    INT4_MODEL_PATH="${PROJECT_DIR}/models/llava-v1.5-7b-int4-gptq"
    if [[ -d "$INT4_MODEL_PATH" ]]; then
        MODEL_NAME="$INT4_MODEL_PATH"
        echo -e "  Using INT4 model: ${GREEN}$MODEL_NAME${NC}"
    else
        echo -e "  ${YELLOW}[WARN] INT4 model not found at $INT4_MODEL_PATH${NC}"
        echo -e "  Run quantization first: python -m quantization.gptq_quantizer"
        echo -e "  Falling back to FP16 model."
        QUANTIZATION=""
    fi
fi

# ── Launch ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Starting server...${NC}"
echo ""

cd "$PROJECT_DIR"

# Option 1: Use our custom FastAPI server (with full control)
# python -m vllm_deploy.api_server

# Option 2: Use vLLM's built-in server (production-grade)
VLLM_CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL_NAME"
    --host "$HOST"
    --port "$VLLM_PORT"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --trust-remote-code
    --dtype auto
)

# Add optional flags
if [[ -n "$QUANTIZATION" ]]; then
    VLLM_CMD+=(--quantization "$QUANTIZATION")
fi

if [[ "${CONTINUOUS_FLAG:-}" != "" ]]; then
    VLLM_CMD+=("$CONTINUOUS_FLAG")
fi

echo "  Command: ${VLLM_CMD[*]}"
echo ""

exec "${VLLM_CMD[@]}"
