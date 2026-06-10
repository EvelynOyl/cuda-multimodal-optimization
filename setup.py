"""
Project-level setup.py — installs the `ops` Python package.
CUDA extensions are built separately via `csrc/setup.py`.
"""

from setuptools import setup, find_packages

setup(
    name="cuda-multimodal-optimization",
    version="0.1.0",
    description="CUDA/C++ multimodal operator optimization + LLaVA-1.5 + vLLM inference",
    packages=find_packages(include=["ops", "ops.*", "vllm_deploy", "vllm_deploy.*", "quantization", "quantization.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "mlx": ["mlx>=0.16.0"],
        "vllm": ["vllm>=0.5.0", "fastapi", "uvicorn", "aiohttp"],
        "quantize": ["accelerate", "datasets", "einops"],
        "dev": ["pytest", "pytest-benchmark", "black", "ruff"],
    },
)
