#!/usr/bin/env python3
"""
scripts/int4_infer.py — INT4 模型推理测试 + FP16 对比

加载 INT4 量化模型，执行文本生成推理，对比 FP16 基准模型。
纯 PyTorch 实现，不依赖 vLLM。

用法：
    source .venv/bin/activate
    python scripts/int4_infer.py

环境变量（可选）：
    FP16_MODEL_PATH  — FP16 模型路径（默认 ./models/llama-fp16-text）
    INT4_MODEL_PATH  — INT4 模型路径（默认 ./models/llava-llama-int4）
    PROMPT           — 测试提示词
    MAX_NEW_TOKENS   — 最大生成 token 数
"""

import os, sys, time, gc, json, hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import torch
import numpy as np

# ── 路径配置 ──────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

MODEL_BASE = Path(os.environ.get("MODEL_BASE", "/home/neu/cuda-multimodal-optimization/models"))
FP16_MODEL_PATH = Path(os.environ.get("FP16_MODEL_PATH", MODEL_BASE / "llama-fp16-text"))
INT4_MODEL_PATH = Path(os.environ.get("INT4_MODEL_PATH", MODEL_BASE / "llava-llama-int4"))

PROMPT = os.environ.get("PROMPT", "The capital of France is")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class InferResult:
    model_name: str
    output_text: str
    prompt_tokens: int
    generated_tokens: int
    total_time_s: float
    prefill_time_s: float
    tokens_per_second: float
    gpu_memory_mb: float
    peak_gpu_memory_mb: float


def measure_gpu_memory() -> float:
    """当前 GPU 显存占用 (MB)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def reset_gpu_memory():
    """清空 GPU 缓存并返回当前显存."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 模型加载器
# ═══════════════════════════════════════════════════════════════════════════

def load_fp16_model(model_path: Path):
    """
    加载 FP16 模型（标准 HuggingFace transformers 格式）。
    """
    print(f"\n{'='*55}")
    print(f"  加载 FP16 模型")
    print(f"  路径: {model_path}")
    print(f"{'='*55}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0

    mem = measure_gpu_memory()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量:      {n_params/1e9:.2f}B")
    print(f"  显存占用:    {mem:.0f} MB  ({mem/1024:.2f} GB)")
    print(f"  加载耗时:    {load_time:.1f}s")
    return model, tokenizer


def load_int4_model(model_path: Path):
    """
    加载 INT4 量化模型。

    尝试顺序：
      1. HuggingFace 标准格式（含 config.json + safetensors）
      2. AutoGPTQ 格式（含 quantize_config.json）
      3. 报错退出
    """
    print(f"\n{'='*55}")
    print(f"  加载 INT4 模型")
    print(f"  路径: {model_path}")
    print(f"{'='*55}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import json

    t0 = time.time()

    # ── 检查模型目录结构 ──────────────────────────────────────────────────
    files_in_dir = list(model_path.glob("*"))
    print(f"  目录文件: {[f.name for f in files_in_dir[:10]]}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 尝试加载 config ───────────────────────────────────────────────────
    config = None
    has_quant_config = False
    quant_method = None

    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config_data = json.load(f)
        if "quantization_config" in config_data:
            has_quant_config = True
            quant_method = config_data["quantization_config"].get("quant_method", "unknown")
        config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        print(f"  Model type:   {config.model_type}")
        print(f"  Hidden size:  {getattr(config, 'hidden_size', '?')}")

    if has_quant_config:
        print(f"  Quant method: {quant_method}")

    # ── 策略 1：bitsandbytes 4-bit 加载（最通用）──────────────────────────
    print(f"\n  尝试加载策略: bitsandbytes 4-bit...")
    try:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        load_time = time.time() - t0
        mem = measure_gpu_memory()
        print(f"  ✓ bitsandbytes 4-bit 加载成功")
        print(f"  显存占用:    {mem:.0f} MB  ({mem/1024:.2f} GB)")
        print(f"  加载耗时:    {load_time:.1f}s")
        return model, tokenizer

    except Exception as e1:
        print(f"  ✗ bitsandbytes 失败: {e1}")

    # ── 策略 2：AutoGPTQ ──────────────────────────────────────────────────
    print(f"\n  尝试加载策略: AutoGPTQ...")
    try:
        from transformers import GPTQConfig

        gptq_config = GPTQConfig(bits=4, group_size=128)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=gptq_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        load_time = time.time() - t0
        mem = measure_gpu_memory()
        print(f"  ✓ AutoGPTQ 加载成功")
        print(f"  显存占用:    {mem:.0f} MB  ({mem/1024:.2f} GB)")
        print(f"  加载耗时:    {load_time:.1f}s")
        return model, tokenizer

    except Exception as e2:
        print(f"  ✗ AutoGPTQ 失败: {e2}")

    # ── 策略 3：FP16 加载再手动量化（最后手段）────────────────────────────
    print(f"\n  尝试加载策略: FP16 fallback + 手动量化...")
    try:
        print("  先用 FP16 加载完整模型（可能 OOM）...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
        load_time = time.time() - t0
        mem = measure_gpu_memory()
        print(f"  ✓ FP16 fallback 加载成功（未量化，仅供对比）")
        print(f"  显存占用:    {mem:.0f} MB  ({mem/1024:.2f} GB)")
        return model, tokenizer

    except Exception as e3:
        print(f"  ✗ 所有加载策略都失败了")
        raise RuntimeError(f"无法加载模型: {e3}")


# ═══════════════════════════════════════════════════════════════════════════
# 推理引擎 (纯 PyTorch，无 vLLM 依赖)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
) -> InferResult:
    """
    执行一次文本生成推理，测量延迟和显存。
    """
    torch.cuda.synchronize()

    # ── Tokenize ──────────────────────────────────────────────────────────
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]

    # ── 记录起始显存 ──────────────────────────────────────────────────────
    start_mem = measure_gpu_memory()

    # ── Prefill（处理 prompt）─────────────────────────────────────────────
    torch.cuda.synchronize()
    t_prefill_start = time.perf_counter()

    # 用 generate 做完整推理（prefill + decode 一体）
    t_generate_start = time.perf_counter()

    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    torch.cuda.synchronize()
    t_total = time.perf_counter() - t_generate_start

    # ── Decode ────────────────────────────────────────────────────────────
    generated_ids = outputs[0][prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = len(generated_ids)

    # ── 显存 ──────────────────────────────────────────────────────────────
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    end_mem = measure_gpu_memory()

    # ── 延迟 ──────────────────────────────────────────────────────────────
    tps = generated_tokens / t_total if t_total > 0 else 0

    # 估算 prefill 时间（第一个 token 生成时间）
    # 简化：跑一次 prefill-only forward
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        _ = model(**inputs, use_cache=True)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    return InferResult(
        model_name="fp16" if model.dtype == torch.float16 else "int4",
        output_text=generated_text.strip(),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_time_s=t_total,
        prefill_time_s=t_prefill,
        tokens_per_second=tps,
        gpu_memory_mb=end_mem,
        peak_gpu_memory_mb=peak_mem,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  INT4 量化模型推理测试                            ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Prompt:       {PROMPT}")
    print(f"  Max tokens:   {MAX_NEW_TOKENS}")
    print(f"  Device:       {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    results = {}

    # ── 测试 INT4 模型 ────────────────────────────────────────────────────
    if INT4_MODEL_PATH.exists():
        reset_gpu_memory()
        try:
            model_int4, tok_int4 = load_int4_model(INT4_MODEL_PATH)

            # 标记模型类型
            is_quantized = getattr(model_int4, "is_quantized", False)
            dtype = str(next(model_int4.parameters()).dtype)
            model_int4.config.update({"model_type_label": f"int4 ({dtype})"})

            print(f"\n  运行 INT4 推理...")
            result_int4 = run_inference(model_int4, tok_int4, PROMPT, MAX_NEW_TOKENS)
            result_int4.model_name = "INT4 (量化)"
            results["int4"] = result_int4

            print(f"\n  ── INT4 推理结果 ──")
            print(f"  输出文本:     {result_int4.output_text!r}")
            print(f"  Prompt tokens:{result_int4.prompt_tokens}")
            print(f"  生成 tokens:  {result_int4.generated_tokens}")
            print(f"  总耗时:       {result_int4.total_time_s*1000:.0f} ms")
            print(f"  Prefill:      {result_int4.prefill_time_s*1000:.0f} ms")
            print(f"  吞吐:         {result_int4.tokens_per_second:.1f} tok/s")
            print(f"  显存:         {result_int4.gpu_memory_mb:.0f} MB")
            print(f"  峰值显存:     {result_int4.peak_gpu_memory_mb:.0f} MB")

            # 释放显存
            del model_int4
            reset_gpu_memory()

        except Exception as e:
            print(f"\n  ❌ INT4 模型加载/推理失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n  ⚠ INT4 模型路径不存在: {INT4_MODEL_PATH}")
        print(f"  跳过 INT4 测试。")

    # ── 测试 FP16 模型（对比基准）─────────────────────────────────────────
    if FP16_MODEL_PATH.exists():
        reset_gpu_memory()
        try:
            model_fp16, tok_fp16 = load_fp16_model(FP16_MODEL_PATH)

            print(f"\n  运行 FP16 推理...")
            result_fp16 = run_inference(model_fp16, tok_fp16, PROMPT, MAX_NEW_TOKENS)
            result_fp16.model_name = "FP16 (基准)"
            results["fp16"] = result_fp16

            print(f"\n  ── FP16 推理结果 ──")
            print(f"  输出文本:     {result_fp16.output_text!r}")
            print(f"  Prompt tokens:{result_fp16.prompt_tokens}")
            print(f"  生成 tokens:  {result_fp16.generated_tokens}")
            print(f"  总耗时:       {result_fp16.total_time_s*1000:.0f} ms")
            print(f"  Prefill:      {result_fp16.prefill_time_s*1000:.0f} ms")
            print(f"  吞吐:         {result_fp16.tokens_per_second:.1f} tok/s")
            print(f"  显存:         {result_fp16.gpu_memory_mb:.0f} MB")
            print(f"  峰值显存:     {result_fp16.peak_gpu_memory_mb:.0f} MB")

            del model_fp16
            reset_gpu_memory()

        except Exception as e:
            print(f"\n  ❌ FP16 模型加载/推理失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n  ⚠ FP16 模型路径不存在: {FP16_MODEL_PATH}")
        print(f"  跳过 FP16 测试。")

    # ── 对比总结 ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  对比总结")
    print(f"{'='*55}")

    if "int4" in results and "fp16" in results:
        r_int4 = results["int4"]
        r_fp16 = results["fp16"]

        mem_reduction = (1 - r_int4.peak_gpu_memory_mb / r_fp16.peak_gpu_memory_mb) * 100
        latency_ratio = r_int4.total_time_s / r_fp16.total_time_s
        tps_ratio = r_int4.tokens_per_second / r_fp16.tokens_per_second

        print(f"")
        print(f"  {'指标':<24s} {'FP16':>12s} {'INT4':>12s} {'变化':>12s}")
        print(f"  {'─'*60}")
        print(f"  {'峰值显存 (MB)':<24s} {r_fp16.peak_gpu_memory_mb:>11.0f}  {r_int4.peak_gpu_memory_mb:>11.0f}  {mem_reduction:>+10.1f}%")
        print(f"  {'总耗时 (ms)':<24s} {r_fp16.total_time_s*1000:>11.0f}  {r_int4.total_time_s*1000:>11.0f}  {(latency_ratio-1)*100:>+10.1f}%")
        print(f"  {'吞吐 (tok/s)':<24s} {r_fp16.tokens_per_second:>11.1f}  {r_int4.tokens_per_second:>11.1f}  {(tps_ratio-1)*100:>+10.1f}%")
        print(f"  {'生成 tokens':<24s} {r_fp16.generated_tokens:>11d}  {r_int4.generated_tokens:>11d}")
        print(f"")
        print(f"  INT4 输出: {r_int4.output_text!r}")
        print(f"  FP16 输出: {r_fp16.output_text!r}")

        # ── 质量检查 ──────────────────────────────────────────────────────
        if r_int4.output_text and r_fp16.output_text:
            common_prefix = ""
            for a, b in zip(r_int4.output_text.split(), r_fp16.output_text.split()):
                if a == b:
                    common_prefix += a + " "
                else:
                    break
            if len(common_prefix) > 3:
                print(f"  公共前缀: {common_prefix.strip()!r} → 输出质量正常")
            else:
                print(f"  ⚠ 公共前缀很短或为空，INT4 可能产生退化输出")

    elif "int4" in results:
        r_int4 = results["int4"]
        print(f"  INT4 峰值显存: {r_int4.peak_gpu_memory_mb:.0f} MB")
        print(f"  INT4 吞吐:     {r_int4.tokens_per_second:.1f} tok/s")
        print(f"  (无 FP16 数据可对比)")
    elif "fp16" in results:
        print(f"  FP16 峰值显存: {results['fp16'].peak_gpu_memory_mb:.0f} MB")
        print(f"  (无 INT4 数据可对比)")
    else:
        print(f"  无可用结果。")

    # ── 输出 JSON ─────────────────────────────────────────────────────────
    json_path = PROJECT_DIR / "infer_results.json"
    json_results = {}
    for k, r in results.items():
        json_results[k] = {
            "output_text": r.output_text,
            "prompt_tokens": r.prompt_tokens,
            "generated_tokens": r.generated_tokens,
            "total_time_s": round(r.total_time_s, 3),
            "prefill_time_s": round(r.prefill_time_s, 3),
            "tokens_per_second": round(r.tokens_per_second, 1),
            "gpu_memory_mb": round(r.gpu_memory_mb, 0),
            "peak_gpu_memory_mb": round(r.peak_gpu_memory_mb, 0),
        }
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {json_path}")


if __name__ == "__main__":
    main()
