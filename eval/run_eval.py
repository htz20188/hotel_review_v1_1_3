#!/usr/bin/env python3
"""RAG 自动化批量评估脚本（方向 17 + 方向 11）

读取 eval/questions.csv，对每个问题在多种 RAG 配置下运行，
将原始结果（回答、召回评论、各项延迟、是否使用 HyDE 等）保存到
eval/eval_results_raw.csv，供后续 judge.py 评分与 summarize_results.py 汇总。

配置（--configs 可选子集）：
    baseline          不开 HyDE
    full_hyde         完整 HyDE（3 条假设评论）
    light_hyde        轻量 HyDE（1 条综合假设评论）
    conditional_hyde  条件 HyDE（按问题类型决定是否启用）

用法：
    python eval/run_eval.py
    python eval/run_eval.py --max-questions 5
    python eval/run_eval.py --configs baseline full_hyde
    python eval/run_eval.py --max-questions 5 --configs baseline light_hyde
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

# 让脚本无论从哪个目录运行都能找到 rag-core 的模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 终端 UTF-8，避免中文乱码
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv()  # 兜底：当前工作目录的 .env

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_CSV = EVAL_DIR / "questions.csv"
RAW_CSV = EVAL_DIR / "eval_results_raw.csv"

# 各评估配置 → rag.query() 的参数
CONFIGS = {
    "baseline":         {"enable_hyde": False, "hyde_mode": "full"},
    "full_hyde":        {"enable_hyde": True,  "hyde_mode": "full"},
    "light_hyde":       {"enable_hyde": True,  "hyde_mode": "light"},
    "conditional_hyde": {"enable_hyde": True,  "hyde_mode": "conditional"},
}

# 写入 CSV 的字段顺序
FIELDNAMES = [
    "qid", "query", "category", "config", "response",
    "top_comments", "top_comment_ids",
    "total_latency", "retrieval_latency", "generation_latency", "hyde_latency",
    "used_hyde", "error",
]


def check_env() -> dict:
    """检查必填环境变量，缺失则报错退出（不打印 Key 本身）。

    本地向量模式（默认）只需 DASHSCOPE_API_KEY。
    """
    required = {"DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY")}
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"错误: 缺少环境变量: {', '.join(missing)}")
        print("请在 rag-core/.env 中设置后重试。")
        sys.exit(1)
    return required


def build_rag():
    """初始化 RAG 系统（复用项目已有组件，与 main.py 逻辑一致）。"""
    env = check_env()

    intl_api_key = os.getenv("DASHSCOPE_INTL_API_KEY")
    if intl_api_key:
        import dashscope
        from clients import DASHSCOPE_INTL_API_BASE
        dashscope.base_http_api_url = DASHSCOPE_INTL_API_BASE

    data_dir = ROOT / "data"
    for f in ["inverted_index.pkl", "filtered_comments.csv"]:
        if not (data_dir / f).exists():
            print(f"错误: 数据文件不存在: {data_dir / f}")
            sys.exit(1)

    from rag_system import HotelReviewRAG
    print("正在初始化 RAG 系统...")
    use_local_vectors = os.getenv("USE_DASHVECTOR", "").strip() != "1"
    rag = HotelReviewRAG(
        api_key=env["DASHSCOPE_API_KEY"],
        dashvector_api_key=os.getenv("DASHVECTOR_API_KEY"),
        dashvector_endpoint=os.getenv("DASHVECTOR_HOTEL_ENDPOINT"),
        data_dir=data_dir,
        intl_api_key=intl_api_key,
        use_local_vectors=use_local_vectors,
    )
    print("RAG 系统初始化完成\n")
    return rag


def load_questions(max_questions: int | None) -> list[dict]:
    """读取问题集，返回 [{qid, query, category, gold_keywords}, ...]。"""
    rows = []
    with open(QUESTIONS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if max_questions is not None:
        rows = rows[:max_questions]
    return rows


def _safe(d, *keys, default=None):
    """从嵌套 dict 中安全取值，任意一层缺失都返回 default。"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_record(result: dict) -> dict:
    """从 rag.query() 返回结构中稳健地抽取评估所需字段。

    返回格式可能因系统迭代而变化，这里全部用 .get / 容错提取，
    并把不定结构（评论列表）统一转换为字符串，保证 CSV 可读。
    """
    comments = _safe(result, "references", "comments", default=[]) or []

    # 统一把评论转成字符串（用 ||| 分隔，单条内换行替换为空格）
    comment_texts = []
    comment_ids = []
    for c in comments:
        if isinstance(c, dict):
            text = str(c.get("comment", "")).replace("\n", " ").replace("\r", " ").strip()
            cid = c.get("comment_id", "")
        else:
            text = str(c).replace("\n", " ").strip()
            cid = ""
        if text:
            comment_texts.append(text)
        if cid:
            comment_ids.append(str(cid))

    timing = result.get("timing", {}) if isinstance(result, dict) else {}
    hyde_timing = _safe(timing, "retrieval", "routes", "hyde", default=None)
    if isinstance(hyde_timing, dict):
        hyde_latency = hyde_timing.get("total")
    else:
        hyde_latency = hyde_timing  # 可能是数字或 None

    return {
        "response": str(result.get("response", "")).strip() if isinstance(result, dict) else "",
        "top_comments": " ||| ".join(comment_texts),
        "top_comment_ids": ";".join(comment_ids),
        "total_latency": timing.get("total"),
        "retrieval_latency": _safe(timing, "retrieval", "total"),
        "generation_latency": timing.get("generation"),
        "hyde_latency": hyde_latency,
        "used_hyde": _safe(result, "hyde", "used"),
    }


def run_one(rag, query: str, config_name: str) -> dict:
    """运行单个 (query, config)，返回结果字段；出错时记录 error 不中断。"""
    params = CONFIGS[config_name]
    start = time.time()
    try:
        result = rag.query(
            query,
            enable_hyde=params["enable_hyde"],
            hyde_mode=params["hyde_mode"],
            print_response=False,
        )
        rec = extract_record(result)
        rec["error"] = ""
        # 兜底：若系统未返回 total_latency，用外层计时
        if rec.get("total_latency") in (None, ""):
            rec["total_latency"] = round(time.time() - start, 4)
        return rec
    except Exception as e:
        return {
            "response": "", "top_comments": "", "top_comment_ids": "",
            "total_latency": round(time.time() - start, 4),
            "retrieval_latency": None, "generation_latency": None,
            "hyde_latency": None, "used_hyde": None,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="RAG 自动化批量评估")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="只评估前 N 个问题（默认全部）")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        choices=list(CONFIGS.keys()),
                        help="要运行的配置（默认全部）")
    args = parser.parse_args()

    questions = load_questions(args.max_questions)
    configs = args.configs
    print(f"问题数: {len(questions)} | 配置: {', '.join(configs)} | "
          f"共 {len(questions) * len(configs)} 次运行\n")

    rag = build_rag()

    with open(RAW_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        idx = 0
        total = len(questions) * len(configs)
        for q in questions:
            for config_name in configs:
                idx += 1
                qid, query, category = q["qid"], q["query"], q.get("category", "")
                print(f"[{idx}/{total}] qid={qid} | {config_name} | {query}")
                rec = run_one(rag, query, config_name)
                row = {
                    "qid": qid, "query": query, "category": category,
                    "config": config_name, **rec,
                }
                writer.writerow(row)
                f.flush()  # 实时落盘，长时间评估中途中断也能保留已完成结果
                if rec["error"]:
                    print(f"    ! 出错: {rec['error']}")

    print(f"\n完成，原始结果已保存到: {RAW_CSV}")


if __name__ == "__main__":
    main()
