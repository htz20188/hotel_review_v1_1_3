#!/usr/bin/env python3
"""对比不同摘要检索方法的效果"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from rag_system import HotelReviewRAG
from hierarchical_summarizer import HierarchicalSummarizer
from dotenv import load_dotenv
import os

load_dotenv()


def compute_ndcg(relevance_scores, k):
    """计算 NDCG@k
    
    Args:
        relevance_scores: 相关性得分列表（按排序顺序）
        k: 截断位置
    """
    dcg = 0
    for i, score in enumerate(relevance_scores[:k]):
        dcg += (2**score - 1) / np.log2(i + 2)
    
    # 理想排序（按得分降序）
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = 0
    for i, score in enumerate(ideal_scores):
        idcg += (2**score - 1) / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0


def get_relevance_score(comment, query, rag):
    """获取评论与查询的相关性得分（使用 reranker）"""
    try:
        documents = [comment['comment']]
        relevance_map = rag.reranker.rerank(query, documents)
        return relevance_map.get(0, 0)
    except:
        # 回退：使用 LTR 得分
        return comment.get('ltr_score', comment.get('final_score', 0.5))


def evaluate_retrieval(rag, summarizer, test_queries: list, test_comments: list):
    """评估不同检索方法"""
    
    results = {
        'baseline': {'ndcg@5': [], 'ndcg@10': []},
        'hierarchical_summary': {'ndcg@5': [], 'ndcg@10': []}
    }
    
    for query, comments in zip(test_queries, test_comments):
        print(f"\n处理查询: {query}")
        
        # 获取原始评论的相关性得分（作为 ground truth）
        relevance_scores_original = []
        for comment in comments[:20]:  # 取前20条作为候选集
            score = get_relevance_score(comment, query, rag)
            relevance_scores_original.append(score)
        
        # ===== 方法1：基线（纯向量相似度）=====
        # 使用原有的摘要检索逻辑
        from retriever import HybridRetriever
        # 获取向量检索结果作为基线
        query_embedding = rag.retriever.embedding_client.embed_batch([query])[0]
        
        # 模拟：用向量相似度排序
        comment_texts = [c['comment'] for c in comments[:20]]
        comment_embeddings = rag.retriever.embedding_client.embed_batch(comment_texts)
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], comment_embeddings)[0]
        baseline_ranked_indices = np.argsort(similarities)[::-1]
        
        baseline_scores = [relevance_scores_original[i] for i in baseline_ranked_indices]
        
        results['baseline']['ndcg@5'].append(compute_ndcg(baseline_scores, 5))
        results['baseline']['ndcg@10'].append(compute_ndcg(baseline_scores, 10))
        
        # ===== 方法2：层次化摘要检索 =====
        # 生成摘要
        summary_output = summarizer.summarize(comments[:50], query, depth=3)
        summary_text = summary_output['final_summary']
        
        # 用摘要进行检索（将摘要作为查询）
        summary_embedding = rag.retriever.embedding_client.embed_batch([summary_text])[0]
        summary_similarities = cosine_similarity([summary_embedding], comment_embeddings)[0]
        summary_ranked_indices = np.argsort(summary_similarities)[::-1]
        
        summary_scores = [relevance_scores_original[i] for i in summary_ranked_indices]
        
        results['hierarchical_summary']['ndcg@5'].append(compute_ndcg(summary_scores, 5))
        results['hierarchical_summary']['ndcg@10'].append(compute_ndcg(summary_scores, 10))
    
    return results


def main():
    # 初始化
    print("初始化 RAG 系统...")
    rag = HotelReviewRAG(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        dashvector_api_key=os.getenv("DASHVECTOR_API_KEY"),
        dashvector_endpoint=os.getenv("DASHVECTOR_HOTEL_ENDPOINT"),
        data_dir=Path(__file__).parent / "data"
    )
    
    print("初始化摘要生成器...")
    from clients import LLMClient
    llm_client = LLMClient(os.getenv("DASHSCOPE_API_KEY"), model="qwen-plus")
    summarizer = HierarchicalSummarizer(llm_client, rag.retriever.embedding_client)
    
    # 测试查询
    test_queries = [
        "酒店的早餐怎么样？",
        "房间空间大吗？",
        "服务态度好不好？",
        "酒店位置方便吗？",
        "整体体验如何？"
    ]
    
    # 获取测试评论
    test_comments = []
    for query in test_queries:
        result = rag.query(query, use_ltr=True, enable_ranking=True, print_response=False)
        comments = result['references']['comments'][:50]
        test_comments.append(comments)
        print(f"查询 '{query[:20]}...' 获取 {len(comments)} 条评论")
    
    # 评估
    print("\n" + "=" * 60)
    print("开始检索效果对比评估")
    print("=" * 60)
    
    results = evaluate_retrieval(rag, summarizer, test_queries, test_comments)
    
    # 计算平均值
    avg_results = {
        'baseline': {
            'ndcg@5': np.mean(results['baseline']['ndcg@5']),
            'ndcg@10': np.mean(results['baseline']['ndcg@10']),
            'std@5': np.std(results['baseline']['ndcg@5']),
            'std@10': np.std(results['baseline']['ndcg@10'])
        },
        'hierarchical_summary': {
            'ndcg@5': np.mean(results['hierarchical_summary']['ndcg@5']),
            'ndcg@10': np.mean(results['hierarchical_summary']['ndcg@10']),
            'std@5': np.std(results['hierarchical_summary']['ndcg@5']),
            'std@10': np.std(results['hierarchical_summary']['ndcg@10'])
        }
    }
    
    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"{'方法':<25} {'NDCG@5':<12} {'NDCG@10':<12}")
    print("-" * 50)
    print(f"{'基线（纯向量相似度）':<25} {avg_results['baseline']['ndcg@5']:.4f} ± {avg_results['baseline']['std@5']:.4f}     {avg_results['baseline']['ndcg@10']:.4f} ± {avg_results['baseline']['std@10']:.4f}")
    print(f"{'本方法（层次化摘要）':<25} {avg_results['hierarchical_summary']['ndcg@5']:.4f} ± {avg_results['hierarchical_summary']['std@5']:.4f}     {avg_results['hierarchical_summary']['ndcg@10']:.4f} ± {avg_results['hierarchical_summary']['std@10']:.4f}")
    
    # 计算提升
    improvement_5 = (avg_results['hierarchical_summary']['ndcg@5'] - avg_results['baseline']['ndcg@5']) / avg_results['baseline']['ndcg@5'] * 100
    improvement_10 = (avg_results['hierarchical_summary']['ndcg@10'] - avg_results['baseline']['ndcg@10']) / avg_results['baseline']['ndcg@10'] * 100
    
    print(f"\n提升: NDCG@5 +{improvement_5:.1f}%, NDCG@10 +{improvement_10:.1f}%")
    
    # 保存结果
    with open('retrieval_compare_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'avg_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in avg_results.items()},
            'per_query': results,
            'improvement': {'ndcg@5': improvement_5, 'ndcg@10': improvement_10}
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 retrieval_compare_results.json")
    
    # 输出 LaTeX 表格代码
    print("\n" + "=" * 60)
    print("LaTeX 表格代码（复制到报告）")
    print("=" * 60)
    print("""
\\begin{table}[H]
\\centering
\\caption{摘要检索效果对比}
\\begin{tabular}{@{}lcc@{}}
\\toprule
\\textbf{检索方法} & \\textbf{NDCG@5} & \\textbf{NDCG@10} \\\\
\\midrule
基线（纯向量相似度） & {:.3f} & {:.3f} \\\\
\\textbf{本方法（层次化摘要）} & {:.3f} & {:.3f} \\\\
\\midrule
提升 & +{:.1f}\\% & +{:.1f}\\% \\\\
\\bottomrule
\\end{tabular}
\\end{table}
    """.format(
        avg_results['baseline']['ndcg@5'],
        avg_results['baseline']['ndcg@10'],
        avg_results['hierarchical_summary']['ndcg@5'],
        avg_results['hierarchical_summary']['ndcg@10'],
        improvement_5,
        improvement_10
    ))


if __name__ == "__main__":
    main()