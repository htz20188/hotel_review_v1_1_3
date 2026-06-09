#!/usr/bin/env python3
"""测试层次化摘要生成器"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hierarchical_summarizer import HierarchicalSummarizer, evaluate_summarizer
from clients import LLMClient, EmbeddingClient
from rag_system import HotelReviewRAG
import pandas as pd


def main():
    # 初始化
    from dotenv import load_dotenv
    load_dotenv()
    import os
    
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    # 先初始化 RAG 系统（复用检索能力）
    rag = HotelReviewRAG(
        api_key=api_key,
        dashvector_api_key=os.getenv("DASHVECTOR_API_KEY"),
        dashvector_endpoint=os.getenv("DASHVECTOR_HOTEL_ENDPOINT"),
        data_dir=Path(__file__).parent / "data"
    )
    
    # 初始化摘要器
    llm_client = LLMClient(api_key, model="qwen-plus")
    summarizer = HierarchicalSummarizer(llm_client, rag.retriever.embedding_client)
    
    # 测试查询
    test_queries = [
        "酒店的早餐怎么样？",
        "房间空间大吗？",
        "服务态度好不好？",
        "酒店位置方便吗？",
        "整体体验如何？",
    ]
    
    # 获取检索结果
    retrieval_results = []
    for query in test_queries:
        print(f"\n检索: {query}")
        # 使用 LTR 模式获取候选
        result = rag.query(
            query,
            use_ltr=True,
            enable_ranking=True,
            print_response=False
        )
        comments = result['references']['comments'][:50]  # 取前50条
        retrieval_results.append(comments)
        print(f"  召回了 {len(comments)} 条评论")
    
    # 评估
    print("\n" + "="*60)
    print("开始评估摘要生成器")
    print("="*60)
    
    eval_result = evaluate_summarizer(summarizer, test_queries, retrieval_results)
    
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(eval_result['summary'])
    print("\n详细指标:")
    for k, v in eval_result['average'].items():
        print(f"  {k}: {v:.4f}")
    
    # 输出示例摘要
    print("\n" + "="*60)
    print("示例摘要")
    print("="*60)
    for r in eval_result['per_query'][:2]:
        print(f"\n查询: {r['query']}")
        print(f"摘要: {r['summary']}...")
        print(f"压缩比: {r['metrics']['compression_ratio']:.2%}")
        print(f"覆盖率: {r['metrics']['coverage']:.1%}")


if __name__ == "__main__":
    main()
