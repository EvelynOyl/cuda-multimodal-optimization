# 服务器执行步骤

## Step 0：上传脚本到服务器

```bash
# 在你的 Mac 上执行：
scp scripts/server_setup.sh \
    scripts/int4_infer.py \
    scripts/benchmark_fp16_vs_int4.py \
    neu@219.216.65.31:/home/neu/cuda-multimodal-optimization/scripts/
```

## Step 1：环境配置 + vLLM 升级

```bash
ssh neu@219.216.65.31
cd /home/neu/cuda-multimodal-optimization
source .venv/bin/activate
bash scripts/server_setup.sh
```

## Step 2：INT4 推理测试（单个 case）

```bash
source .venv/bin/activate
python scripts/int4_infer.py
```

预期输出：INT4 和 FP16 各生成一段文本，底部有对比表格。

## Step 3：FP16 vs INT4 全量压测

```bash
source .venv/bin/activate

# 基础压测
python scripts/benchmark_fp16_vs_int4.py

# 或自定义参数
WARMUP_ITERS=5 BENCH_ITERS=20 MAX_NEW_TOKENS=128 \
python scripts/benchmark_fp16_vs_int4.py
```

输出文件：
- `bench_results.json` — 完整压测数据
- `resume_metrics.txt` — 可直接写进简历的数据

## 如果模型路径不同

```bash
FP16_MODEL_PATH=/path/to/fp16/model \
INT4_MODEL_PATH=/path/to/int4/model \
python scripts/int4_infer.py
```
