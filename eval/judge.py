#!/usr/bin/env python3
"""LLM-as-Judge 自动评分（方向 17）

读取 eval/eval_results_raw.csv，对每条系统回答用 LLM 基于
「用户问题 + 系统回答 + 召回评论」进行多维度打分，
结果保存到 eval/eval_results_judged.csv。

评分维度：
    relevance     回答是否直接回应用户问题       1-5
    completeness  回答是否覆盖问题主要方面       1-5
    groundedness  回答是否被召回评论支持         1-5
    hallucination 是否存在评论中没有依据的编造   true/false
    judge_reason  简短理由

用法：
    python eval/judge.py
    python eval/judge.py --max-rows 10
    python eval/judge.py --model qwen-plus
"""

import os
import re
import sys
import csv
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv()

from clients import LLMClient, DASHSCOPE_INTL_API_BASE

EVAL_DIR = Path(__file__).resolve().parent
RAW_CSV = EVAL_DIR / "eval_results_raw.csv"
JUDGED_CSV = EVAL_DIR / "eval_results_judged.csv"

# 原始字段 + 评分字段
JUDGE_FIELDS = ["relevance", "completeness", "groundedness",
                "hallucination", "judge_reason", "judge_error"]

JUDGE_PROMPT_TEMPLATE = """你是一个严格的 RAG 系统评估专家。请根据"用户问题"、"系统回答"和"召回评论"，对系统回答打分。

【用户问题】
{query}

【系统回答】
{response}

【召回评论（系统检索到的证据，回答应基于这些评论）】
{comments}

【评分维度】
1. relevance（相关性，1-5）：回答是否直接回应用户问题。5=完全切题，1=答非所问。
2. completeness（完整性，1-5）：回答是否覆盖问题的主要方面。5=全面，1=严重遗漏。
3. groundedness（可溯源性，1-5）：回答内容是否被召回评论支持。5=全部有据，1=基本无据。
4. hallucination（幻觉，true/false）：回答中是否存在召回评论里完全没有依据的编造信息。有编造为 true，否则 false。
5. judge_reason：用一句话简要说明打分理由。

【输出要求】
- 只输出一个 JSON 对象，不要输出任何解释或 Markdown 代码块标记。
- 严格使用如下结构（分数为整数 1-5，hallucination 为布尔值）：
{{"relevance": 4, "completeness": 3, "groundedness": 5, "hallucination": false, "judge_reason": "理由"}}
"""


def extract_json(text: str) -> dict:
    """从模型输出中稳健解析 JSON：先直接解析，失败则抽取第一个 {...} 子串。"""
    text = (text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 抽取第一个平衡的大括号子串
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("无法从输出中解析 JSON")


def normalize_score(val, lo=1, hi=5):
    """把分数规整为 [lo, hi] 的整数；无法解析返回 None。"""
    try:
        n = int(round(float(val)))
        return max(lo, min(hi, n))
    except Exception:
        return None


def normalize_bool(val):
    """把各种形式的真假规整为 'true'/'false' 字符串；无法解析返回 ''。"""
    if isinstance(val, bool):
        return "true" if val else "false"
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "是", "有"):
        return "true"
    if s in ("false", "0", "no", "否", "没有", "无"):
        return "false"
    return ""


def judge_one(client: LLMClient, query: str, response: str, comments: str) -> dict:
    """对单条回答评分，返回评分字段；失败时填 judge_error 不抛出。"""
    # 回答为空（如运行阶段就报错）直接跳过评分
    if not response.strip():
        return {f: "" for f in JUDGE_FIELDS} | {"judge_error": "empty_response"}

    comments_text = comments if comments.strip() else "（无召回评论）"
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query, response=response, comments=comments_text
    )

    last_err = ""
    for attempt in range(2):
        try:
            raw = client.generate(prompt, temperature=0.0)
            data = extract_json(raw)
            return {
                "relevance": normalize_score(data.get("relevance")),
                "completeness": normalize_score(data.get("completeness")),
                "groundedness": normalize_score(data.get("groundedness")),
                "hallucination": normalize_bool(data.get("hallucination")),
                "judge_reason": str(data.get("judge_reason", "")).replace("\n", " ").strip(),
                "judge_error": "",
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < 1:
                time.sleep(0.2)

    return {f: "" for f in ["relevance", "completeness", "groundedness",
                            "hallucination", "judge_reason"]} | {"judge_error": last_err}


def build_client(model: str) -> LLMClient:
    api_key = os.getenv("DASHSCOPE_INTL_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 缺少 DASHSCOPE_API_KEY（或 DASHSCOPE_INTL_API_KEY）。")
        sys.exit(1)
    if os.getenv("DASHSCOPE_INTL_API_KEY"):
        import dashscope
        dashscope.base_http_api_url = DASHSCOPE_INTL_API_BASE
    # json=True：利用 DashScope 的 JSON 模式提高结构化输出稳定性
    return LLMClient(api_key, model=model, json=True)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge 自动评分")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="只评分前 N 行（默认全部），方便小样本测试")
    parser.add_argument("--model", type=str, default="qwen-plus",
                        help="评分模型（默认 qwen-plus）")
    args = parser.parse_args()

    if not RAW_CSV.exists():
        print(f"错误: 未找到 {RAW_CSV}，请先运行 run_eval.py。")
        sys.exit(1)

    with open(RAW_CSV, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.max_rows is not None:
        rows = rows[:args.max_rows]

    client = build_client(args.model)
    print(f"待评分行数: {len(rows)} | 评分模型: {args.model}\n")

    in_fields = list(rows[0].keys()) if rows else []
    out_fields = in_fields + [f for f in JUDGE_FIELDS if f not in in_fields]

    with open(JUDGED_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            # 运行阶段就报错的行，直接透传，不再调用 judge
            if row.get("error"):
                scores = {ff: "" for ff in JUDGE_FIELDS}
                scores["judge_error"] = "skipped_run_error"
            else:
                scores = judge_one(
                    client,
                    row.get("query", ""),
                    row.get("response", ""),
                    row.get("top_comments", ""),
                )
            print(f"[{i}/{len(rows)}] qid={row.get('qid')} | {row.get('config')} | "
                  f"rel={scores.get('relevance')} comp={scores.get('completeness')} "
                  f"grd={scores.get('groundedness')} hall={scores.get('hallucination')}"
                  + (f" | judge_error={scores['judge_error']}" if scores.get("judge_error") else ""))
            writer.writerow({**row, **scores})
            f.flush()

    print(f"\n完成，评分结果已保存到: {JUDGED_CSV}")


if __name__ == "__main__":
    main()
