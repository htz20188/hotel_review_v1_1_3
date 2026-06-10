# Task 9: Complex Query Understanding and Query Rewriting

本目录对应方向 9：复杂 Query 理解，即在 RAG 检索前将用户的复杂、模糊、多意图问题改写为更清晰的检索 Query。

## 目录结构

```text
before/intent.py
after/intent.py
complex_queries.csv
evaluate_query_rewrite.ipynb
query_rewrite_results.csv
query_rewrite_summary.csv
generate_query_rewrite_dataset.py
supervised_data/
finetune_query_rewriter.ipynb
finetune_outputs/
```

## 1. Prompt 优化实验

`before/intent.py` 是原始 Query 改写方案，来自项目原有 `IntentExpander`。

`after/intent.py` 是优化后的 Query 改写方案，主要改动包括：

- 强化复杂 Query 改写 prompt；
- 明确处理多意图、比较型、约束型、模糊体验型问题；
- 增加 JSON 解析容错；
- 增加空 Query 过滤、重复 Query 去重；
- 增加权重规范化，保证权重总和为 1.0；
- 异常时回退原始 Query。

`complex_queries.csv` 是 25 条复杂 Query 测试集，包含：

```text
qid,query,type,expected_intents
```

`evaluate_query_rewrite.ipynb` 用于比较 before 和 after 两个版本的改写效果。

输出结果：

```text
query_rewrite_results.csv
query_rewrite_summary.csv
```

核心结果：

| version | num_queries | parse_success_rate | mean_intent_coverage | mean_num_rewritten_queries | mean_latency_seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 25 | 1.0000 | 0.6807 | 2.0000 | 1.0678 |
| after | 25 | 1.0000 | 0.7153 | 2.0400 | 1.0249 |

## 2. 专用小模型探索

`generate_query_rewrite_dataset.py` 用于调用大模型 API 生成监督数据，将复杂 Query 转换为可用于 SFT 的 JSONL 格式。

输出目录：

```text
supervised_data/
```

其中包含：

```text
query_rewrite_train.jsonl
query_rewrite_val.jsonl
query_rewrite_test.jsonl
query_rewrite_all.jsonl
generation_errors.csv
```

`finetune_query_rewriter.ipynb` 用于在服务器上微调 Qwen3-4B-Instruct。实验使用 LoRA 微调，得到一个本地 Query Rewriter。

输出目录：

```text
finetune_outputs/
```

其中：

```text
eval_summary.csv
eval_predictions.csv
eval_complex25_summary.csv
eval_complex25_predictions.csv
```

`eval_complex25_*` 是在同一批 25 条 `complex_queries.csv` 上评估微调后小模型的结果，可与 Prompt + API 方案直接比较。

核心结果：

| version | num_examples | parse_success_rate | mean_intent_coverage | mean_latency_seconds |
| --- | ---: | ---: | ---: | ---: |
| before prompt + API | 25 | 1.0000 | 0.6807 | 1.0678 |
| after prompt + API | 25 | 1.0000 | 0.7153 | 1.0249 |
| Qwen3-4B + LoRA | 25 | 1.0000 | 0.8053 | 3.5087 |

## 3. 复现顺序

### 3.1 运行 Prompt 改写对比

先准备 DashScope API Key。

运行：

```text
evaluate_query_rewrite.ipynb
```

该 notebook 会读取 `complex_queries.csv`，分别调用 `before/intent.py` 和 `after/intent.py`，输出改写结果和汇总指标。

### 3.2 生成监督数据

如需重新生成监督数据，运行：

```bash
python generate_query_rewrite_dataset.py --variants 2 --model qwen-plus
```

小样本测试：

```bash
python generate_query_rewrite_dataset.py --max-rows 3 --variants 1
```

### 3.3 微调小模型

在 GPU 服务器上运行：

```text
finetune_query_rewriter.ipynb
```

默认模型：

```text
Qwen/Qwen3-4B-Instruct-2507
```

如服务器无法联网下载模型，可将 notebook 中的 `MODEL_NAME_OR_PATH` 改为本地模型路径。

## 4. 说明

本目录不包含：

- LoRA adapter 权重；
- 大模型完整权重。

微调小模型实验显示，本地 Qwen3-4B + LoRA 在同一 25 条测试集上的意图覆盖率高于 Prompt + API 方案，但推理延迟更高。当前结论应理解为课程项目范围内的探索性结果，后续若要用于生产，需要扩充监督数据、人工校验标注，并进一步优化推理速度。
