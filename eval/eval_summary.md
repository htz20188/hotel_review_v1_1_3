# RAG 自动化评估汇总

按配置聚合的评估指标（由 eval/summarize_results.py 自动生成）。

| config | num_success | num_error | mean_relevance | mean_completeness | mean_groundedness | hallucination_rate | avg_total_latency(s) | avg_hyde_latency(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 41 | 0 | 4.976 | 3.927 | 5.0 | 0.024 | 9.075 |  |
| full_hyde | 41 | 0 | 4.976 | 3.854 | 5.0 | 0.0 | 12.434 | 3.46 |
| light_hyde | 41 | 0 | 4.976 | 3.951 | 5.0 | 0.0 | 10.579 | 1.363 |
| conditional_hyde | 41 | 0 | 4.976 | 3.927 | 5.0 | 0.0 | 10.892 | 3.024 |

> 说明：mean_* 为 1-5 分制均值（越高越好）；hallucination_rate 越低越好；
> avg_hyde_latency 仅统计实际启用 HyDE 的样本。