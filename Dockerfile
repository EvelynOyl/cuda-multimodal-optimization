# ═══════════════════════════════════════════════════════════════════════════
# Dockerfile — CUDA 11.8 多模态优化推理镜像
# ═══════════════════════════════════════════════════════════════════════════
#
# 前置：Docker Desktop Settings → Docker Engine 中配置 registry-mirrors
#   { "registry-mirrors": ["https://docker.1ms.run"] }
#
# 构建：
#   docker buildx build --platform linux/amd64 -t cuda-multimodal-opt:v1.0 --load .
#
# 运行（服务器）：
#   docker run --gpus all -v /path/to/models:/app/models \
#       -e FP16_MODEL_PATH=/app/models/llama-fp16-text \
#       -e INT4_MODEL_PATH=/app/models/llava-llama-int4 \
#       cuda-multimodal-opt:v1.0 \
#       python scripts/benchmark_fp16_vs_int4.py
# ═══════════════════════════════════════════════════════════════════════════

# ── Stage 1: 依赖安装 ─────────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# ── ① 切换 Ubuntu APT 源到阿里云镜像（解决 archive.ubuntu.com 被墙）─────────
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list \
    && sed -i 's|http://ports.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list \
    && apt-get update

# ── ② 系统依赖 ────────────────────────────────────────────────────────────
RUN apt-get install -y --no-install-recommends --allow-change-held-packages \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    python3-setuptools \
    build-essential \
    gcc-11 g++-11 \
    git \
    curl \
    ca-certificates \
    libcudnn8-dev \
    libssl-dev \
    libffi-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 1 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 1 \
    && rm -rf /var/lib/apt/lists/*

# ── ③ 配置 pip 国内镜像 ───────────────────────────────────────────────────
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com

# ── ④ 升级 pip ────────────────────────────────────────────────────────────
RUN python -m pip install --no-cache-dir --upgrade \
    pip==24.0 \
    setuptools==69.5.1 \
    wheel==0.43.0

# ── ⑤ PyTorch 2.1.2 + CUDA 11.8 ──────────────────────────────────────────
# PyTorch wheel 必须从官方源下载（阿里云镜像没有 cu118 构建版）
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ── ⑥ vLLM + bitsandbytes + xformers ─────────────────────────────────────
RUN pip install --no-cache-dir \
    vllm==0.3.3 \
    ray>=2.9.0 \
    nvidia-ml-py>=12.535 \
    bitsandbytes==0.41.3 \
    xformers==0.0.23.post1

# ── ⑦ Transformers 生态 ──────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    tokenizers>=0.19.0 \
    accelerate==0.28.0 \
    safetensors>=0.4.0

# ── ⑧ 图像 / 推理服务 / 量化 ─────────────────────────────────────────────
RUN pip install --no-cache-dir \
    pillow>=10.0.0,<11.0.0 \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.27.0" \
    pydantic>=2.5.0 \
    aiohttp>=3.9.0 \
    httpx>=0.27.0 \
    datasets>=2.18.0 \
    einops>=0.7.0 \
    tqdm>=4.66.0

# ── ⑨ GPTQ 量化 ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir auto-gptq==0.7.1

# ── ⑩ 配置 / 监控 / 基础 ─────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    pyyaml>=6.0 \
    "numpy>=1.24.0,<2.0.0" \
    psutil>=5.9.0 \
    prometheus-client>=0.19.0

# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: 精简运行时镜像
# ═══════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PYTHONUNBUFFERED=1

# ── ① 切换 APT 源 + 安装运行时系统依赖 ────────────────────────────────────
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list \
    && sed -i 's|http://ports.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends --allow-change-held-packages \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3-setuptools \
        libcudnn8 \
        libssl-dev \
        libffi-dev \
        curl \
        ca-certificates \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && rm -rf /var/lib/apt/lists/*

# ── ② 从 builder 拷贝所有已安装的 Python 包 ──────────────────────────────
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ── ③ 拷贝项目源码 ────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app/

# ── ④ 运行时目录 ──────────────────────────────────────────────────────────
RUN mkdir -p /app/models /app/outputs

# ── ⑤ 包导入验证 ──────────────────────────────────────────────────────────
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA_API: {torch.version.cuda}')" \
    && python -c "import vllm; print(f'vLLM: {vllm.__version__}')" \
    && python -c "import bitsandbytes; print('bitsandbytes: OK')" \
    && python -c "import auto_gptq; print('auto-gptq: OK')" \
    && python -c "import transformers; print(f'transformers: {transformers.__version__}')" \
    && python -c "import fastapi; print(f'fastapi: {fastapi.__version__}')" \
    && echo "=== All packages verified ==="

# ── ⑥ 默认入口 ────────────────────────────────────────────────────────────
ENTRYPOINT ["python"]
CMD ["-c", "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU detected. 请使用 --gpus all 运行容器'); print('用法: docker run --gpus all -v /path/to/models:/app/models cuda-multimodal-opt:v1.0 python scripts/benchmark_fp16_vs_int4.py')"]
