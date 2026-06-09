# RAG 自动化评估 & HyDE 优化实验模块

本模块对应课程项目的两个算法优化方向：

- **方向 17：RAG 自动化评估** —— 在无人工标注数据的情况下，用 LLM-as-Judge 构建自动化评估体系。
- **方向 11：HyDE 优化** —— 扩展 HyDE 为 `full / light / conditional` 三种模式，分析效果与延迟的权衡。

所有命令均在 `D:\llm_bus\rag-core`（即 `rag-core/`）目录下运行。

---

## 1. 环境准备

```bat
cd D:\llm_bus\rag-core
pip install -r requirements.txt
copy .env.example .env
```

然后用文本编辑器打开 `.env`，**填写 API Key**：

```
DASHSCOPE_API_KEY=你的_DashScope_Key      # 必填（用于嵌入、生成、评分）
DASHSCOPE_INTL_API_KEY=                    # 可选，填了则切换新加坡端点
```

> 代码不会打印或泄露 `.env` 中的任何 Key。

### 关于向量库（重要架构说明）

原系统的评论向量存放在**云端 DashVector**。由于该云端集群已失效，且本机
chromadb / onnxruntime 原生组件不稳定，本项目已改为**纯本地 numpy 向量库**：

- 评论向量由 `build_local_index.py` 用 DashScope 嵌入后存为 `data/comment_vectors.npz`；
- vector / HyDE 检索路改走本地余弦检索（`local_vector.py`），零原生依赖；
- 反向 Query 路、摘要路无本地数据，已停用（在各评估配置中一致缺省，**不影响**
  HyDE / 质量的横向对比）；
- 因此 `DASHVECTOR_*` 不再需要。若仍想用云端 DashVector，设环境变量
  `USE_DASHVECTOR=1` 并填好 `DASHVECTOR_*` 即可切回。

### 1.1 构建本地向量索引（一次性，约 2-3 分钟）

```bat
python build_local_index.py
```

产出 `data/comment_vectors.npz`（2171 条评论向量）。如需重建加 `--rebuild`。

---

## 2. baseline 测试（确认主流程跑通）

```bat
python main.py "酒店的早餐怎么样？"
python main.py "套房空间大吗？" --verbose --hyde
```

### HyDE 三种模式（方向 11）

```bat
python main.py "酒店整体体验怎么样？" --hyde                  # 兼容旧用法，等价于 full
python main.py "酒店整体体验怎么样？" --hyde-mode full        # 3 条假设评论
python main.py "酒店整体体验怎么样？" --hyde-mode light       # 1 条综合假设评论（低延迟）
python main.py "酒店整体体验怎么样？" --hyde-mode conditional # 按问题类型决定是否启用
```

- `full`：原始逻辑，生成 3 条假设评论（2 正 1 负），召回更全但延迟更高。
- `light`：仅生成 1 条同时含正负面信息的综合假设评论，延迟更低。
- `conditional`：由 `should_use_hyde(query)` 决定是否启用（明确属性问题不开、宽泛体验问题开）；
  启用时使用 `full` 生成。规则见 [`intent.py`](../intent.py) 中的 `should_use_hyde`。

---

## 3. 小样本评估（先跑通流程）

```bat
python eval/run_eval.py --max-questions 5 --configs baseline full_hyde
python eval/judge.py --max-rows 10
python eval/summarize_results.py
```

## 4. 完整评估

```bat
python eval/run_eval.py
python eval/judge.py
python eval/summarize_results.py
```

---

## 5. 各脚本说明与产出文件

| 脚本 | 作用 | 产出 |
| --- | --- | --- |
| `run_eval.py` | 读取 `questions.csv`，对每个问题在多种配置下运行 RAG，保存原始结果 | `eval_results_raw.csv` |
| `judge.py` | 对每条回答用 LLM-as-Judge 打分（relevance/completeness/groundedness/hallucination） | `eval_results_judged.csv` |
| `summarize_results.py` | 按 config 聚合指标，生成汇总表 | `eval_summary.csv`、`eval_summary.md` |

### 评估配置（`--configs`）

| 配置 | 含义 |
| --- | --- |
| `baseline` | 不开 HyDE |
| `full_hyde` | 完整 HyDE（3 条假设评论） |
| `light_hyde` | 轻量 HyDE（1 条综合假设评论） |
| `conditional_hyde` | 条件 HyDE（按问题类型决定是否启用） |

### `run_eval.py` 常用参数

- `--max-questions N`：只评估前 N 个问题（默认全部）。
- `--configs A B ...`：只运行指定配置（默认全部 4 种）。

### `judge.py` 常用参数

- `--max-rows N`：只评分前 N 行（默认全部），方便小样本测试。
- `--model`：评分模型（默认 `qwen-plus`）。

---

## 6. 数据与字段

- 测试问题集：`questions.csv`，字段 `qid,query,category,gold_keywords`，
  覆盖早餐、房间、套房、卫生、服务、交通、商务、亲子、性价比、设施、噪音、整体体验等类别。
- 原始结果 `eval_results_raw.csv` 关键字段：`response`、`top_comments`、`top_comment_ids`、
  `total_latency`、`retrieval_latency`、`generation_latency`、`hyde_latency`、`used_hyde`、`error`。
- 评分结果 `eval_results_judged.csv`：在原始字段基础上追加
  `relevance`、`completeness`、`groundedness`、`hallucination`、`judge_reason`、`judge_error`。

> 健壮性：单个问题/单条评分报错都不会中断整体流程，错误会记录到 `error` / `judge_error` 列后继续。
> 所有 CSV 使用 UTF-8（带 BOM）保存，Excel 打开中文不乱码。

---

## 7. 最终产出

```
eval/eval_results_raw.csv      原始运行结果
eval/eval_results_judged.csv   LLM 评分结果
eval/eval_summary.csv          按配置聚合的指标（CSV）
eval/eval_summary.md           按配置聚合的指标（Markdown，可直接贴进报告）
```
