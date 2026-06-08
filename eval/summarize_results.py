#!/usr/bin/env python3
"""评估结果汇总（方向 17 + 方向 11）

读取 eval/eval_results_judged.csv，按 config 聚合各项指标，
输出 eval/eval_summary.csv 与 eval/eval_summary.md（Markdown 表格，
可直接粘贴进课程技术报告）。

聚合指标：
    mean_relevance / mean_completeness / mean_groundedness  各维度均值（仅统计有效评分）
    hallucination_rate                                      幻觉率（true 占比）
    avg_total_latency                                       平均端到端延迟
    avg_hyde_latency                                        平均 HyDE 延迟
    num_success / num_error                                 成功/失败次数

用法：
    python eval/summarize_results.py
"""

import sys
import csv
import argparse
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

EVAL_DIR = Path(__file__).resolve().parent
JUDGED_CSV = EVAL_DIR / "eval_results_judged.csv"
SUMMARY_CSV = EVAL_DIR / "eval_summary.csv"
SUMMARY_MD = EVAL_DIR / "eval_summary.md"

SUMMARY_FIELDS = [
    "config", "num_success", "num_error",
    "mean_relevance", "mean_completeness", "mean_groundedness",
    "hallucination_rate", "avg_total_latency", "avg_hyde_latency",
]


def _to_float(v):
    try:
        if v in (None, "", "None"):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def aggregate(rows: list[dict]) -> list[dict]:
    """按 config 分组聚合，返回每个配置一行的汇总。"""
    by_config: dict[str, list[dict]] = {}
    for r in rows:
        by_config.setdefault(r.get("config", "unknown"), []).append(r)

    summary = []
    for config, group in by_config.items():
        # 成功 = 运行阶段无 error；失败 = 运行阶段有 error
        errors = [r for r in group if (r.get("error") or "").strip()]
        success = [r for r in group if not (r.get("error") or "").strip()]

        relevance = [_to_float(r.get("relevance")) for r in success]
        completeness = [_to_float(r.get("completeness")) for r in success]
        groundedness = [_to_float(r.get("groundedness")) for r in success]

        # 幻觉率：在有有效 hallucination 标记的样本中，true 的占比
        hall_flags = [(r.get("hallucination") or "").strip().lower() for r in success]
        hall_valid = [h for h in hall_flags if h in ("true", "false")]
        hall_rate = (round(hall_flags.count("true") / len(hall_valid), 3)
                     if hall_valid else None)

        total_lat = [_to_float(r.get("total_latency")) for r in success]
        # HyDE 延迟只统计实际启用了 HyDE 的样本，避免被 0 拉低
        hyde_lat = [_to_float(r.get("hyde_latency")) for r in success
                    if (r.get("used_hyde") or "").strip().lower() == "true"]

        summary.append({
            "config": config,
            "num_success": len(success),
            "num_error": len(errors),
            "mean_relevance": _mean(relevance),
            "mean_completeness": _mean(completeness),
            "mean_groundedness": _mean(groundedness),
            "hallucination_rate": hall_rate,
            "avg_total_latency": _mean(total_lat),
            "avg_hyde_latency": _mean(hyde_lat),
        })

    # 固定配置顺序，便于报告对照
    order = {"baseline": 0, "full_hyde": 1, "light_hyde": 2, "conditional_hyde": 3}
    summary.sort(key=lambda s: order.get(s["config"], 99))
    return summary


def fmt(v) -> str:
    return "" if v is None else str(v)


def write_csv(summary: list[dict]):
    with open(SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def write_md(summary: list[dict]):
    headers = [
        "config", "num_success", "num_error",
        "mean_relevance", "mean_completeness", "mean_groundedness",
        "hallucination_rate", "avg_total_latency(s)", "avg_hyde_latency(s)",
    ]
    lines = ["# RAG 自动化评估汇总", "",
             "按配置聚合的评估指标（由 eval/summarize_results.py 自动生成）。", "",
             "| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for s in summary:
        cells = [
            s["config"], fmt(s["num_success"]), fmt(s["num_error"]),
            fmt(s["mean_relevance"]), fmt(s["mean_completeness"]),
            fmt(s["mean_groundedness"]), fmt(s["hallucination_rate"]),
            fmt(s["avg_total_latency"]), fmt(s["avg_hyde_latency"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "> 说明：mean_* 为 1-5 分制均值（越高越好）；hallucination_rate 越低越好；",
              "> avg_hyde_latency 仅统计实际启用 HyDE 的样本。"]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    argparse.ArgumentParser(description="评估结果汇总").parse_args()

    if not JUDGED_CSV.exists():
        print(f"错误: 未找到 {JUDGED_CSV}，请先运行 judge.py。")
        sys.exit(1)

    with open(JUDGED_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    summary = aggregate(rows)
    write_csv(summary)
    write_md(summary)

    # 终端预览
    print("汇总结果：\n")
    print(SUMMARY_MD.read_text(encoding="utf-8"))
    print(f"\n已保存: {SUMMARY_CSV}")
    print(f"已保存: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
