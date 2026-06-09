#!/usr/bin/env python3
"""构建本地评论向量索引（替代失效的云端 DashVector）

读取 data/filtered_comments.csv，用 DashScope text-embedding-v4 对每条评论计算向量，
存为 data/comment_vectors.npz（纯 numpy，零原生依赖），供 retriever 的 vector / hyde
路做本地余弦检索。向量维度与查询端一致（1024），保证检索语义可比。

用法：
    python build_local_index.py                 # 构建（已存在则跳过）
    python build_local_index.py --rebuild       # 强制重建
    python build_local_index.py --limit 50      # 仅构建前 50 条（快速联调）
"""

import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv()

from clients import EmbeddingClient, DASHSCOPE_INTL_API_BASE
from local_vector import VECTORS_FILE

DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "filtered_comments.csv"
NPZ_PATH = DATA_DIR / VECTORS_FILE

BATCH_SIZE = 10  # DashScope text-embedding 单次批量上限


def get_embedding_client() -> EmbeddingClient:
    api_key = os.getenv("DASHSCOPE_INTL_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 缺少 DASHSCOPE_API_KEY。")
        sys.exit(1)
    if os.getenv("DASHSCOPE_INTL_API_KEY"):
        import dashscope
        dashscope.base_http_api_url = DASHSCOPE_INTL_API_BASE
    return EmbeddingClient(api_key)


def embed_with_retry(client: EmbeddingClient, texts: list[str], retries: int = 3):
    """带简单重试的批量嵌入。"""
    for attempt in range(retries):
        try:
            return client.embed_batch(texts)
        except Exception as e:
            if attempt < retries - 1:
                wait = 1.5 * (attempt + 1)
                print(f"    嵌入失败({e})，{wait:.1f}s 后重试...")
                time.sleep(wait)
            else:
                raise


def _meta_val(v) -> str:
    """把房型等字段规整为字符串（NaN/None → 空串），供过滤匹配。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def main():
    parser = argparse.ArgumentParser(description="构建本地评论向量索引")
    parser.add_argument("--rebuild", action="store_true", help="强制重建（覆盖已有文件）")
    parser.add_argument("--limit", type=int, default=None, help="仅构建前 N 条（联调用）")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"嵌入批大小（默认{BATCH_SIZE}）")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"错误: 未找到评论数据 {CSV_PATH}")
        sys.exit(1)

    if NPZ_PATH.exists() and not args.rebuild and not args.limit:
        print(f"本地向量文件已存在: {NPZ_PATH}（如需重建请加 --rebuild）")
        return

    df = pd.read_csv(CSV_PATH, dtype={"_id": str})
    if args.limit:
        df = df.head(args.limit)
    total = len(df)
    print(f"待索引评论数: {total}")

    ids_all = df["_id"].astype(str).tolist()
    comments_all = df["comment"].astype(str).tolist()
    room_all = [_meta_val(v) for v in df.get("room_type", pd.Series([None] * total))]
    fuzzy_all = [_meta_val(v) for v in df.get("fuzzy_room_type", pd.Series([None] * total))]

    embed_client = get_embedding_client()

    bs = args.batch_size
    all_emb = []
    t0 = time.time()
    for start in range(0, total, bs):
        end = min(start + bs, total)
        embeddings = embed_with_retry(embed_client, comments_all[start:end])
        all_emb.extend(embeddings)
        done = end
        if done % 200 < bs or done == total:
            print(f"  进度 {done}/{total}  ({time.time() - t0:.1f}s)")

    emb = np.asarray(all_emb, dtype=np.float32)
    print(f"嵌入完成: shape={emb.shape}")

    np.savez(
        NPZ_PATH,
        ids=np.asarray(ids_all, dtype=object),
        emb=emb,
        room=np.asarray(room_all, dtype=object),
        fuzzy=np.asarray(fuzzy_all, dtype=object),
    )
    print(f"\n完成：已保存 {total} 条向量 → {NPZ_PATH}，耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
