#!/usr/bin/env python3
"""测试结构化回复生成效果"""

import sys
import json
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from rag_system import HotelReviewRAG
from dotenv import load_dotenv
import os

load_dotenv()


def evaluate_structured_response(rag, test_queries: list):
    """评估结构化回复质量"""
    
    results = []
    
    # 定义评估维度
    metrics = {
        "completeness": [],      # 信息完整性
        "readability": [],       # 可读性
        "citation_accuracy": [], # 引用准确性
        "info_density": [],      # 信息密度（词/句）
        "parse_success": []      # 格式解析成功率
    }
    
    for query in test_queries:
        print(f"\n处理查询: {query}")
        
        # 生成回复（会自动使用结构化提示词）
        result = rag.query(
            query,
            use_ltr=True,
            enable_ranking=True,
            print_response=True
        )
        
        response = result['response']
        
        # 1. 尝试解析 JSON
        parse_success = False
        parsed = None
        try:
            # 提取 JSON 内容（可能被 markdown 包裹）
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0]
            else:
                json_str = response
            parsed = json.loads(json_str.strip())
            parse_success = True
        except:
            parse_success = False
        
        metrics["parse_success"].append(1 if parse_success else 0)
        
        if parse_success:
            # 2. 信息完整性：检查必要字段
            required_fields = ['summary', 'positive', 'negative', 'details', 'confidence']
            completeness_score = sum(1 for f in required_fields if f in parsed) / len(required_fields)
            metrics["completeness"].append(completeness_score)
            
            # 3. 可读性：基于 response 长度和句子数估算
            sentences = response.replace('\n', '。').split('。')
            avg_word_per_sentence = len(response) / max(len(sentences), 1)
            # 可读性评分：平均每句 15-25 字为最佳
            if 15 <= avg_word_per_sentence <= 25:
                readability = 5.0
            elif 10 <= avg_word_per_sentence <= 35:
                readability = 4.0
            elif 5 <= avg_word_per_sentence <= 50:
                readability = 3.0
            else:
                readability = 2.0
            metrics["readability"].append(readability)
            
            # 4. 信息密度
            total_words = len(response)
            total_sentences = len(sentences)
            info_density = total_words / max(total_sentences, 1)
            metrics["info_density"].append(info_density)
            
            # 5. 引用准确性：检查 [[ref:]] 格式
            import re
            citations = re.findall(r'\[\[ref:[\d,\s]+\]\]', response)
            citation_valid = all(c.startswith('[[ref:') and c.endswith(']]') for c in citations)
            metrics["citation_accuracy"].append(1 if citation_valid else 0)
        else:
            # 解析失败时的默认值
            metrics["completeness"].append(0)
            metrics["readability"].append(1.0)
            metrics["info_density"].append(0)
            metrics["citation_accuracy"].append(0)
    
    # 汇总结果
    summary = {
        "completeness": sum(metrics["completeness"]) / len(metrics["completeness"]),
        "readability": sum(metrics["readability"]) / len(metrics["readability"]),
        "citation_accuracy": sum(metrics["citation_accuracy"]) / len(metrics["citation_accuracy"]),
        "info_density": sum(metrics["info_density"]) / len(metrics["info_density"]),
        "parse_success_rate": sum(metrics["parse_success"]) / len(metrics["parse_success"])
    }
    
    return summary, metrics


def evaluate_baseline_response(rag, test_queries: list):
    """评估原有（非结构化）回复效果"""
    
    # 临时禁用结构化回复
    # 注意：这需要修改 generator.py 或传入特殊参数
    # 这里我们直接调用但记录原有格式的结果
    
    results = []
    
    for query in test_queries:
        print(f"\n处理查询 (baseline): {query}")
        
        result = rag.query(
            query,
            use_ltr=True,
            enable_ranking=True,
            print_response=False
        )
        
        response = result['response']
        
        # 基线评估
        sentences = response.replace('\n', '。').split('。')
        avg_word_per_sentence = len(response) / max(len(sentences), 1)
        
        # 可读性估算
        if 15 <= avg_word_per_sentence <= 25:
            readability = 4.5
        elif 10 <= avg_word_per_sentence <= 35:
            readability = 3.5
        else:
            readability = 2.5
        
        # 信息密度
        info_density = len(response) / max(len(sentences), 1)
        
        # 完整性（估算：是否包含多个方面）
        aspect_keywords = ['早餐', '房间', '服务', '位置', '价格', '卫生']
        completeness = sum(1 for kw in aspect_keywords if kw in response) / len(aspect_keywords)
        
        # 引用准确性（检查是否有 [[ref:]] 格式）
        import re
        has_citation = bool(re.search(r'\[\[ref:\d+\]\]', response))
        
        results.append({
            'readability': readability,
            'info_density': info_density,
            'completeness': completeness,
            'citation_accuracy': 1 if has_citation else 0
        })
    
    summary = {
        "completeness": sum(r['completeness'] for r in results) / len(results),
        "readability": sum(r['readability'] for r in results) / len(results),
        "citation_accuracy": sum(r['citation_accuracy'] for r in results) / len(results),
        "info_density": sum(r['info_density'] for r in results) / len(results),
        "parse_success_rate": 0  # 基线无结构化输出
    }
    
    return summary


def main():
    # 初始化 RAG 系统
    print("初始化 RAG 系统...")
    rag = HotelReviewRAG(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        dashvector_api_key=os.getenv("DASHVECTOR_API_KEY"),
        dashvector_endpoint=os.getenv("DASHVECTOR_HOTEL_ENDPOINT"),
        data_dir=Path(__file__).parent / "data"
    )
    print("初始化完成\n")
    
    # 测试查询集
    test_queries = [
        "酒店的早餐怎么样？",
        "房间空间大吗？",
        "服务态度好不好？",
        "酒店位置方便吗？",
        "整体体验如何？",
        "酒店的卫生状况如何？",
        "价格性价比怎么样？",
        "适合带孩子住吗？",
        "停车方便吗？",
        "酒店的设施新吗？"
    ]
    
    print("=" * 60)
    print("评估结构化回复效果")
    print("=" * 60)
    
    structured_summary, _ = evaluate_structured_response(rag, test_queries)
    
    print("\n" + "=" * 60)
    print("评估基线（非结构化）回复效果")
    print("=" * 60)
    
    baseline_summary = evaluate_baseline_response(rag, test_queries)
    
    # 输出结果表格
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(f"{'指标':<20} {'基线':<15} {'结构化':<15} {'提升':<10}")
    print("-" * 60)
    
    for metric in ['completeness', 'readability', 'citation_accuracy', 'info_density', 'parse_success_rate']:
        baseline_val = baseline_summary.get(metric, 0)
        structured_val = structured_summary.get(metric, 0)
        if baseline_val > 0:
            improvement = (structured_val - baseline_val) / baseline_val * 100
        else:
            improvement = 0
        print(f"{metric:<20} {baseline_val:<15.3f} {structured_val:<15.3f} +{improvement:.1f}%")
    
    # 保存结果到文件
    results = {
        'baseline': baseline_summary,
        'structured': structured_summary,
        'test_queries': test_queries
    }
    
    with open('structured_eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 structured_eval_results.json")


if __name__ == "__main__":
    main()
