#!/usr/bin/env python3
"""Generate supervised data for direction 9 query rewriting.

This script reads complex hotel-review queries and asks a teacher LLM to produce
standardized query-rewrite labels. The output JSONL can be used for later SFT or
LoRA fine-tuning of a small query rewriter model.

Example:
    python generate_query_rewrite_dataset.py --variants 2 --model qwen-plus
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import dashscope
from dashscope import Generation


BASE_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = """你是酒店评论 RAG 系统的复杂 Query 改写标注员。

任务：把用户的复杂问题改写为 1-3 个适合检索酒店评论的中文 Query，并为每个 Query 分配权重。

要求：
- 每个改写 Query 只聚焦一个清晰关注点。
- 必须保留原问题中的房型、人群、时间、比较对象等关键约束。
- 多意图问题应拆分为多个 Query；明确单意图问题不要强行拆分。
- 不要加入用户没有表达、且无法从评论中检索验证的实时信息，如今日房态、当前价格、实时活动。
- weight 只能是 0.2、0.4、0.6、0.8、1.0，且同一 user_query 下权重总和必须为 1.0。
- 只输出 JSON，不要输出解释或 Markdown。
"""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key() -> tuple[str, bool]:
    load_env_file(BASE_DIR / ".env")
    intl_key = os.getenv("DASHSCOPE_INTL_API_KEY")
    api_key = intl_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DASHSCOPE_API_KEY or DASHSCOPE_INTL_API_KEY. "
            "Set it in task9_scripts_admit/.env or environment variables."
        )
    if intl_key:
        dashscope.base_http_api_url = DASHSCOPE_INTL_API_BASE
    return api_key, bool(intl_key)


def call_llm(api_key: str, model: str, prompt: str, temperature: float) -> dict[str, Any]:
    response = Generation.call(
        api_key=api_key,
        model=model,
        prompt=prompt,
        temperature=temperature,
        result_format="message",
        response_format={"type": "json_object"},
    )
    if response.status_code != 200:
        raise RuntimeError(f"LLM call failed: {response.message}")
    text = response.output.choices[0].message.content.strip()
    return parse_json(text)


def parse_json(text: str) -> dict[str, Any]:
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def build_prompt(row: dict[str, str], variants: int) -> str:
    return f"""{SYSTEM_PROMPT}

请基于下面的种子问题生成监督训练样本。

种子问题：
{row["query"]}

问题类型：
{row["type"]}

期望覆盖意图：
{row["expected_intents"]}

请输出一个 JSON 对象，格式如下：
{{
  "examples": [
    {{
      "user_query": "原始问题或语义等价的复杂问题",
      "query_type": "{row["type"]}",
      "expected_intents": "{row["expected_intents"]}",
      "rewritten_queries": [
        {{"query": "改写 Query 1", "weight": 0.6}},
        {{"query": "改写 Query 2", "weight": 0.4}}
      ]
    }}
  ]
}}

生成要求：
- examples 数组长度必须为 {variants + 1}。
- 第 1 条 example 必须使用原始种子问题本身。
- 后续 {variants} 条 example 是种子问题的自然改写版本，语义保持一致，但表达方式可以变化。
- 每条 example 的 rewritten_queries 必须覆盖期望意图，并保留关键约束。
- 不要生成与酒店评论问答无关的问题。
"""


def normalize_rewrites(rewrites: Any) -> list[dict[str, Any]]:
    if not isinstance(rewrites, list):
        raise ValueError("rewritten_queries must be a list")

    cleaned: list[dict[str, Any]] = []
    seen = set()
    for item in rewrites:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query or query in seen:
            continue
        weight = float(item.get("weight", 0))
        cleaned.append({"query": query, "weight": weight})
        seen.add(query)
        if len(cleaned) == 3:
            break

    if not cleaned:
        raise ValueError("empty rewritten_queries")

    weights = [max(float(item["weight"]), 0.0) for item in cleaned]
    snapped = snap_weights(weights)
    for item, weight in zip(cleaned, snapped):
        item["weight"] = weight
    return cleaned


def snap_weights(weights: list[float]) -> list[float]:
    n = min(max(len(weights), 1), 3)
    if n == 1:
        return [1.0]

    total = sum(weights[:n])
    if total <= 0:
        normalized = [1 / n] * n
    else:
        normalized = [w / total for w in weights[:n]]

    units = [max(1, round(w * 5)) for w in normalized]
    while sum(units) > 5:
        idx = max(range(n), key=lambda i: units[i])
        if units[idx] > 1:
            units[idx] -= 1
        else:
            break
    while sum(units) < 5:
        idx = max(range(n), key=lambda i: normalized[i] - units[i] / 5)
        units[idx] += 1
    return [round(unit / 5, 1) for unit in units]


def to_sft_record(example: dict[str, Any], source_qid: str) -> dict[str, Any]:
    output = {
        "rewritten_queries": normalize_rewrites(example.get("rewritten_queries")),
    }
    user_query = str(example["user_query"]).strip()
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
        ],
        "metadata": {
            "source_qid": source_qid,
            "query_type": example.get("query_type", ""),
            "expected_intents": example.get("expected_intents", ""),
        },
    }


def split_records(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
        "all": records,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT data for task9 query rewriting.")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "complex_queries.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "supervised_data")
    parser.add_argument("--model", default="qwen-plus", help="Teacher model name.")
    parser.add_argument("--variants", type=int, default=2, help="Paraphrase variants per seed query.")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for smoke tests.")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between API calls.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key, use_intl = get_api_key()
    print(f"Teacher model: {args.model}")
    print(f"Using intl endpoint: {use_intl}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    error_path = args.output_dir / "generation_errors.csv"

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.max_rows:
        rows = rows[:args.max_rows]

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for idx, row in enumerate(rows, 1):
        qid = str(row["qid"])
        print(f"[{idx}/{len(rows)}] qid={qid}: {row['query']}")
        try:
            data = call_llm(
                api_key=api_key,
                model=args.model,
                prompt=build_prompt(row, args.variants),
                temperature=args.temperature,
            )
            examples = data.get("examples", [])
            if not isinstance(examples, list):
                raise ValueError("examples must be a list")
            for example in examples:
                records.append(to_sft_record(example, source_qid=qid))
        except Exception as exc:
            errors.append({"qid": qid, "query": row["query"], "error": repr(exc)})
            print(f"  ERROR: {exc}")
        time.sleep(args.sleep)

    splits = split_records(records, seed=args.seed)
    for name, split_records_ in splits.items():
        write_jsonl(args.output_dir / f"query_rewrite_{name}.jsonl", split_records_)

    with error_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "query", "error"])
        writer.writeheader()
        writer.writerows(errors)

    print("\nDone.")
    print(f"Generated records: {len(records)}")
    print(f"Errors: {len(errors)}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
