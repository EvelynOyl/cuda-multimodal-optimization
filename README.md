# CUDA Multimodal Operator Optimization + LLaVA-1.5 + vLLM

Optimized CUDA/C++ kernels for multimodal LLM inference, featuring custom Linear & Softmax operators with PyTorch C++ bindings, Apple MLX fallback for local development, vLLM-powered LLaVA-1.5 deployment with Continuous Batching, and INT4 weight-only quantization.

## Project Structure

```
cuda-multimodal-optimization/
├── csrc/                        # CUDA C++ source kernels
│   ├── linear/
│   │   ├── linear_cuda_kernel.cu    # Optimized Linear CUDA kernel
│   │   ├── linear_cuda.cpp          # PyTorch C++ binding
│   │   └── linear_mlx.py            # Apple MLX fallback (Mac M-series)
│   ├── softmax/
│   │   ├── softmax_cuda_kernel.cu   # Online Safe Softmax CUDA kernel
│   │   ├── softmax_cuda.cpp         # PyTorch C++ binding
│   │   └── softmax_mlx.py           # Apple MLX fallback
│   └── setup.py                     # torch.utils.cpp_extension build script
├── ops/                         # Python operator wrappers
│   ├── linear.py
│   ├── softmax.py
│   └── utils.py
├── vllm_deploy/                 # vLLM LLaVA-1.5 serving
│   ├── llava_worker.py              # LLaVA multimodal worker
│   ├── continuous_batching.py       # Continuous batching scheduler
│   ├── api_server.py                # FastAPI inference server
│   └── config.py                    # Deployment configuration
├── quantization/                # INT4 quantization
│   ├── int4_quantizer.py            # Weight-only INT4 quantizer
│   ├── gptq_quantizer.py            # GPTQ-style calibration
│   └── calibration.py               # Calibration dataset utils
├── configs/config.yaml
├── tests/
│   ├── test_linear.py
│   ├── test_softmax.py
│   └── test_quantization.py
├── scripts/
│   ├── build_cuda.sh
│   └── run_server.sh
├── setup.py
└── requirements.txt
```

## Quick Start

### Local Development (Apple Silicon Mac)

```bash
# Install dependencies
pip install -r requirements.txt

# MLX operators work natively on Apple Silicon
python tests/test_linear.py --backend mlx
python tests/test_softmax.py --backend mlx
```

### NVIDIA GPU Server (Linux)

```bash
# Build CUDA extensions
bash scripts/build_cuda.sh

# Run tests
python tests/test_linear.py --backend cuda
python tests/test_softmax.py --backend cuda

# INT4 quantization
python -m quantization.int4_quantizer \
    --model liuhaotian/llava-v1.5-7b \
    --output ./llava-int4

# Start vLLM server
bash scripts/run_server.sh
```

## Key Features

- **Tiled GEMM Linear Kernel** — Shared-memory tiled matrix multiply with vectorized loads
- **Online Safe Softmax** — Fused reduce-max + exp + normalize in a single kernel pass
- **MLX Parity** — Apple Silicon native operators for local debugging with identical API
- **Continuous Batching** — Dynamic request scheduling for maximal GPU utilization
- **INT4 GPTQ** — Group-wise asymmetric quantization with Hessian-based calibration
- **LLaVA-1.5 Serving** — Vision-language model serving with image preprocessing pipeline

## Requirements

| Environment | Requirements |
|-------------|-------------|
| Apple Silicon Mac | Python 3.10+, MLX, PyTorch |
| NVIDIA GPU Server | Python 3.10+, CUDA 12.1+, PyTorch 2.1+, vLLM |
