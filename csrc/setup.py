"""
csrc/setup.py — PyTorch C++/CUDA extension build script.

Builds the `linear_cuda` and `softmax_cuda` extension modules using
torch.utils.cpp_extension.

Usage:
    # Build both extensions
    python csrc/setup.py build_ext --inplace

    # Build with specific CUDA architecture
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" python csrc/setup.py build_ext --inplace

    # Debug build
    DEBUG=1 python csrc/setup.py build_ext --inplace

    # Clean build
    python csrc/setup.py clean && python csrc/setup.py build_ext --inplace

Environment variables:
    TORCH_CUDA_ARCH_LIST — Semicolon-separated list of CUDA architectures
                           (e.g., "7.5;8.0;8.6;9.0")
    CUDA_HOME             — Path to CUDA installation
    DEBUG                 — Set to 1 for debug symbols and line info
    MAX_JOBS              — Number of parallel compilation jobs
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CppExtension,
)

# ── Configuration ──────────────────────────────────────────────────────────
BUILD_DIR = Path(__file__).parent
DEBUG = os.environ.get("DEBUG", "0") == "1"

# Detect CUDA availability
try:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    HAS_CUDA = os.path.exists(nvcc_path)
except Exception:
    HAS_CUDA = False

# Detect PyTorch CUDA support
try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False


def get_cuda_arch_flags():
    """
    Generate NVCC architecture flags.

    If TORCH_CUDA_ARCH_LIST is set, use it. Otherwise default to
    a reasonable set for modern GPUs (A100/H100 era).
    """
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if arch_list:
        arches = arch_list.split(";")
        flags = []
        for arch in arches:
            arch = arch.strip()
            # Convert "8.0" → "80" for -gencode
            arch_code = arch.replace(".", "")
            flags.extend([f"-gencode=arch=compute_{arch_code},code=sm_{arch_code}"])
        return flags

    # Default: support Volta → Hopper
    default_arches = ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0"]
    flags = []
    for arch in default_arches:
        arch_code = arch.replace(".", "")
        flags.extend([
            f"-gencode=arch=compute_{arch_code},code=sm_{arch_code}"
        ])
    return flags


def get_compile_args():
    """Get compiler flags for CUDA and C++ compilation."""
    cuda_args = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-Xcompiler", "-fPIC",
        "-Xcompiler", "-Wall",
        "-lineinfo" if DEBUG else "",
        "-G" if DEBUG else "",
    ]
    cuda_args = [a for a in cuda_args if a]  # remove empty strings

    cpp_args = [
        "-O3" if not DEBUG else "-O0 -g",
        "-std=c++17",
        "-fPIC",
        "-Wall",
        "-Wno-unused-function",
        "-Wno-sign-compare",
    ]

    return cuda_args, cpp_args


def check_environment():
    """Print diagnostic information about the build environment."""
    print("=" * 60)
    print("CUDA Extension Build — Environment Check")
    print("=" * 60)

    print(f"  CUDA available (nvcc found):  {HAS_CUDA}")
    print(f"  Torch CUDA available:          {HAS_TORCH_CUDA}")
    print(f"  CUDA_HOME:                     {os.environ.get('CUDA_HOME', 'not set')}")
    print(f"  CUDA_ARCH_LIST:                {os.environ.get('TORCH_CUDA_ARCH_LIST', 'default')}")
    print(f"  Python:                        {sys.version}")
    print(f"  Debug build:                   {DEBUG}")

    if HAS_CUDA:
        result = subprocess.run(
            [os.path.join(cuda_home, "bin", "nvcc"), "--version"],
            capture_output=True, text=True
        )
        print(f"  NVCC version:                  {result.stdout.splitlines()[-1].strip()}")

    if HAS_TORCH_CUDA:
        import torch
        print(f"  PyTorch version:               {torch.__version__}")
        print(f"  PyTorch CUDA version:          {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"  GPU count:                     {torch.cuda.device_count()}")
            print(f"  GPU 0:                         {torch.cuda.get_device_name(0)}")

    print("=" * 60)


# ── Extension definitions ──────────────────────────────────────────────────

cuda_args, cpp_args = get_compile_args()
cuda_arch_flags = get_cuda_arch_flags()
all_cuda_args = cuda_args + cuda_arch_flags

extensions = []

if HAS_CUDA or True:  # Always define; build will skip CUDA if not available
    extensions.append(
        CUDAExtension(
            name="linear_cuda",
            sources=[
                str(BUILD_DIR / "linear" / "linear_cuda.cpp"),
                str(BUILD_DIR / "linear" / "linear_cuda_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": cpp_args,
                "nvcc": all_cuda_args,
            },
            extra_link_args=["-lcuda", "-lcudart"],
            libraries=["cuda", "cudart"],
        )
    )

    extensions.append(
        CUDAExtension(
            name="softmax_cuda",
            sources=[
                str(BUILD_DIR / "softmax" / "softmax_cuda.cpp"),
                str(BUILD_DIR / "softmax" / "softmax_cuda_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": cpp_args,
                "nvcc": all_cuda_args,
            },
            extra_link_args=["-lcuda", "-lcudart"],
            libraries=["cuda", "cudart"],
        )
    )

# ── Build ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_environment()

    if not HAS_CUDA:
        print("\n⚠️  WARNING: CUDA not detected!")
        print("   CUDA extensions will NOT be compiled.")
        print("   Use the MLX backend for local development on Mac:")
        print("     from csrc.linear.linear_mlx import tiled_gemm")
        print("     from csrc.softmax.softmax_mlx import online_safe_softmax")
        print()
        print("   To build CUDA extensions, run on a Linux server with:")
        print("     python csrc/setup.py build_ext --inplace")
        print()

    setup(
        name="cuda_multimodal_ops",
        version="0.1.0",
        description="Optimized CUDA Linear and Softmax operators for multimodal transformers",
        author="Resume Project",
        ext_modules=extensions,
        cmdclass={
            "build_ext": BuildExtension.with_options(
                use_ninja=True,    # Faster parallel builds
                verbose=True,
            )
        },
    )
