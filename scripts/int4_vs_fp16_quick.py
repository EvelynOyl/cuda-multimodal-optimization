#!/usr/bin/env python3
"""INT4 vs FP16 快速推理对比（greedy decoding，避免 NaN 采样）"""

import sys, time, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT = "The capital of France is"
MAX_NEW = 32
MODELS = "/home/neu/cuda-multimodal-optimization/models"

def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ---- INT4 ----
print("=" * 55)
print("  INT4 模型推理")
print("=" * 55)
reset()
t0 = time.time()
tok_i4 = AutoTokenizer.from_pretrained(f"{MODELS}/llava-llama-int4", trust_remote_code=True)
if tok_i4.pad_token is None:
    tok_i4.pad_token = tok_i4.eos_token
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
)
model_i4 = AutoModelForCausalLM.from_pretrained(
    f"{MODELS}/llava-llama-int4", quantization_config=bnb_cfg,
    device_map="auto", trust_remote_code=True,
)
model_i4.eval()
load_i4 = time.time() - t0
mem_i4 = torch.cuda.max_memory_allocated() / 1024**2

inputs_i4 = tok_i4(PROMPT, return_tensors="pt").to("cuda")
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    out_i4 = model_i4.generate(
        **inputs_i4, max_new_tokens=MAX_NEW, do_sample=False,
        pad_token_id=tok_i4.eos_token_id, use_cache=True,
    )
torch.cuda.synchronize()
t_infer_i4 = time.perf_counter() - t0
gen_i4 = len(out_i4[0]) - inputs_i4.input_ids.shape[1]
text_i4 = tok_i4.decode(out_i4[0][inputs_i4.input_ids.shape[1]:], skip_special_tokens=True)
tps_i4 = gen_i4 / t_infer_i4 if t_infer_i4 > 0 else 0

print(f"  Load:  {load_i4:.1f}s")
print(f"  Mem:   {mem_i4:.0f} MB  ({mem_i4/1024:.2f} GB)")
print(f"  Infer: {t_infer_i4*1000:.0f} ms")
print(f"  TPS:   {tps_i4:.1f} tok/s")
print(f"  Gen:   {gen_i4} tokens")
print(f"  Text:  {text_i4!r}")
del model_i4
torch.cuda.empty_cache()

# ---- FP16 ----
print()
print("=" * 55)
print("  FP16 模型推理")
print("=" * 55)
reset()
t0 = time.time()
tok_fp = AutoTokenizer.from_pretrained(f"{MODELS}/llama-fp16-text", trust_remote_code=True)
if tok_fp.pad_token is None:
    tok_fp.pad_token = tok_fp.eos_token
model_fp = AutoModelForCausalLM.from_pretrained(
    f"{MODELS}/llama-fp16-text", torch_dtype=torch.float16,
    device_map="auto", trust_remote_code=True,
)
model_fp.eval()
load_fp = time.time() - t0
mem_fp = torch.cuda.max_memory_allocated() / 1024**2
n_params = sum(p.numel() for p in model_fp.parameters())

inputs_fp = tok_fp(PROMPT, return_tensors="pt").to("cuda")
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    out_fp = model_fp.generate(
        **inputs_fp, max_new_tokens=MAX_NEW, do_sample=False,
        pad_token_id=tok_fp.eos_token_id, use_cache=True,
    )
torch.cuda.synchronize()
t_infer_fp = time.perf_counter() - t0
gen_fp = len(out_fp[0]) - inputs_fp.input_ids.shape[1]
text_fp = tok_fp.decode(out_fp[0][inputs_fp.input_ids.shape[1]:], skip_special_tokens=True)
tps_fp = gen_fp / t_infer_fp if t_infer_fp > 0 else 0

print(f"  Load:  {load_fp:.1f}s")
print(f"  Mem:   {mem_fp:.0f} MB  ({mem_fp/1024:.2f} GB)")
print(f"  Params:{n_params/1e9:.2f} B")
print(f"  Infer: {t_infer_fp*1000:.0f} ms")
print(f"  TPS:   {tps_fp:.1f} tok/s")
print(f"  Gen:   {gen_fp} tokens")
print(f"  Text:  {text_fp!r}")
del model_fp
torch.cuda.empty_cache()

# ---- Summary ----
mem_save_pct = (1 - mem_i4 / mem_fp) * 100
latency_ratio = t_infer_fp / t_infer_i4 if t_infer_i4 > 0 else 0
tps_ratio = tps_i4 / tps_fp if tps_fp > 0 else 0

print()
print("=" * 60)
print("  INT4 vs FP16 对比总结")
print("=" * 60)
print(f"  {'指标':<28s} {'FP16':>14s} {'INT4':>14s}")
print(f"  {'-'*56}")
print(f"  {'显存占用':<28s} {mem_fp:>13.0f}MB {mem_i4:>13.0f}MB")
print(f"  {'显存降低':<28s} -- {mem_save_pct:>13.1f}%")
print(f"  {'推理延迟':<28s} {t_infer_fp*1000:>13.0f}ms {t_infer_i4*1000:>13.0f}ms")
print(f"  {'吞吐量':<28s} {tps_fp:>13.1f}t/s {tps_i4:>13.1f}t/s")
print(f"  {'参数量':<28s} {n_params/1e9:>13.2f}B {'4-bit compressed':>13s}")
print(f"  {'FP16输出':<28s} {text_fp!r}")
print(f"  {'INT4输出':<28s} {text_i4!r}")
