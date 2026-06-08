#!/usr/bin/env python3
"""酒店评论 RAG 智能问答 — CLI 入口

用法:
    python main.py "酒店的早餐怎么样？"
    python main.py "套房空间大吗？" --no-hyde
    python main.py "最近入住的体验如何？" --verbose

环境变量（可通过 .env 文件或直接设置）:
    DASHSCOPE_API_KEY      — DashScope API Key（北京，必填）
    DASHSCOPE_INTL_API_KEY — DashScope API Key（新加坡，可选）
    DASHVECTOR_API_KEY     — DashVector API Key（必填）
    DASHVECTOR_HOTEL_ENDPOINT — DashVector 集合端点（必填）
"""

import os
import sys
import argparse
from pathlib import Path

# Windows 终端默认编码可能不是 UTF-8，强制重配置 stdout/stderr 避免中文乱码/报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()
load_dotenv(Path(__file__).parent / ".env")


def print_rag_result(result: dict):
    """格式化打印 RAG 完整结果"""
    refs = result['references']
    qp = result['query_processing']
    timing = result['timing']

    # 1. 时间统计
    print(f"\n{'='*60}")
    print(f"延迟统计")
    print(f"{'='*60}")
    print(f"  查询处理（不含HyDE）: {timing['query_processing_total']:.3f}s")
    print(f"    • 意图识别: {timing['intent_recognition']:.3f}s")
    print(f"    • 意图检测: {timing['intent_detection']:.3f}s")
    print(f"    • 意图扩展: {timing['intent_expansion']:.3f}s")

    if timing.get('retrieval'):
        rt = timing['retrieval']
        print(f"  混合检索: {rt['total']:.3f}s")
        print(f"    • 文本召回: {rt['routes']['bm25']:.3f}s")
        print(f"    • 向量召回: {rt['routes']['vector']:.3f}s")
        print(f"    • 反向召回: {rt['routes']['reverse']:.3f}s")
        hyde_total = rt['routes']['hyde']['total'] if isinstance(rt['routes']['hyde'], dict) else rt['routes']['hyde']
        print(f"    • HyDE召回: {hyde_total:.3f}s")
        print(f"    • 摘要召回: {rt['routes']['summary']:.3f}s")
        # RRF 融合（传统模式）或候选收集（LTR 模式）
        if 'rrf_fusion' in rt['routes']:
            print(f"    • RRF融合: {rt['routes']['rrf_fusion']:.3f}s")
        if 'candidate_collection' in rt['routes']:
            print(f"    • 候选收集(LTR): {rt['routes']['candidate_collection']:.3f}s")
        if 'num_candidates' in rt:
            print(f"    • 候选数: {rt['num_candidates']}")

    if timing.get('ranking'):
        rk = timing['ranking']
        if rk['total'] > 0:
            if 'feature_extraction' in rk:
                print(f"  排序(LTR): {rk['total']:.3f}s (Rerank {rk['rerank']:.3f}s + 特征 {rk['feature_extraction']:.3f}s + 打分 {rk['scoring']:.3f}s)")
            else:
                print(f"  排序: {rk['total']:.3f}s (Rerank {rk['rerank']:.3f}s + 打分 {rk['scoring']:.3f}s)")
            if rk.get('diversity_eval'):
                de = rk['diversity_eval']
                print(f"  多样性重排({de['method']}): {de['total_time']:.3f}s")
                print(f"    APS: {de['aps_original']:.4f} → {de['aps_reranked']:.4f} (多样性提升 {de['diversity_gain']:.1%})")
                print(f"    相关性保留: {de['relevance_retention']:.1%}")
        else:
            print(f"  排序: 0.000s")

    print(f"  模型回复: {timing['generation']:.3f}s")
    print(f"    • 首字延迟: {timing['ttft_model']:.3f}s")
    print(f"    • 后续回复: {timing['subsequent']:.3f}s")
    print(f"  端到端首字: {timing['ttft']:.3f}s")
    print(f"  总延迟: {timing['total']:.3f}s")

    # 2. 查询处理
    if qp['intent_recognition']:
        print(f"\n{'='*60}")
        print(f"查询处理")
        print(f"{'='*60}")
        print(f"  意图检测: {qp['intent_detection']}")
        if qp['intent_expansion']:
            print(f"  意图扩展:")
            for q in qp['intent_expansion']:
                print(f"    - {q['query']} (weight={q['weight']})")
        else:
            print(f"  意图扩展: 未启用")
    else:
        print(f"\n  未触发检索，直接回答")

    # 3. HyDE 假设回复
    if refs['hyde_responses']:
        print(f"\n{'='*60}")
        print(f"HyDE 假设回复")
        print(f"{'='*60}")
        for q_idx, responses in sorted(refs['hyde_responses'].items()):
            for h_idx, response in enumerate(responses):
                print(f"  Q{q_idx}-H{h_idx}: {response}")
    else:
        print(f"\n  HyDE 召回: 未启用")

    # 4. 召回的摘要
    if refs['summaries']:
        print(f"\n{'='*60}")
        print(f"召回摘要类别 ({len(refs['summaries'])}个)")
        print(f"{'='*60}")
        for i, summary in enumerate(refs['summaries'], 1):
            print(f"  [{i}] {summary['metadata']['category']}")
            print(f"      关键词: {summary['metadata']['keywords']}")
            print(f"      评论数: {summary['metadata']['comment_count']}")
            print(f"      摘要: {summary['summary'][:120]}...")
    else:
        print(f"\n  摘要召回: 未启用")

    # 5. 召回的评论（精简版）
    if refs['comments']:
        print(f"\n{'='*60}")
        print(f"Top {len(refs['comments'])} 评论")
        print(f"{'='*60}")
        for comment in refs['comments']:
            parts = []
            if 'final_rank' in comment:
                parts.append(f"综合=#{comment['final_rank']}({comment['final_score']:.4f})")
            if 'ltr_rank' in comment:
                parts.append(f"LTR=#{comment['ltr_rank']}({comment['ltr_score']:.4f})")
            if 'diversity_rank' in comment:
                parts.append(f"Div=#{comment['diversity_rank']}(orig=#{comment.get('original_rank', '?')})")
            if 'rrf_rank' in comment:
                parts.append(f"RRF=#{comment['rrf_rank']}({comment['rrf_score']:.4f})")
            if 'rerank_rank' in comment:
                parts.append(f"Rerank=#{comment['rerank_rank']}({comment['rerank_score']:.4f})")

            if parts:
                print(f"  {' | '.join(parts)}")
            elif 'rrf_score' in comment:
                print(f"  RRF=#{comment['rrf_rank']} | {comment['rrf_score']:.4f}")
            print(f"      房型: {comment['metadata']['room_type']} | "
                  f"评分: {comment['metadata']['score']} | "
                  f"质量: {comment['metadata']['quality_score']} | "
                  f"日期: {comment['metadata']['publish_date']}")
            print(f"      内容: {comment['comment'][:100]}...")
            print()
    else:
        print(f"\n  未召回评论")


def check_env() -> dict:
    """检查并返回环境变量。

    本地向量模式（默认）只需 DASHSCOPE_API_KEY；DASHVECTOR_* 仅在使用云端
    向量库（USE_DASHVECTOR=1）时才需要，因此此处不作强制要求。
    """
    required = {"DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY")}
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(f"错误: 缺少环境变量: {', '.join(missing)}")
        print("请在 .env 文件中设置这些变量，或直接 export 它们。")
        sys.exit(1)
    return required


def main():
    parser = argparse.ArgumentParser(
        description="酒店评论 RAG 智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py "酒店的早餐怎么样？"
  python main.py "房间隔音效果如何？" --verbose
  python main.py "最近的服务质量有提升吗？" --hyde --verbose
  python main.py "花园大床房空间大吗？" --no-ranking
        """
    )
    parser.add_argument("query", type=str, help="用户问题")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细检索信息")
    parser.add_argument("--hyde", action="store_true", help="启用 HyDE 增强召回（默认等价于 --hyde-mode full）")
    parser.add_argument("--hyde-mode", type=str, default=None,
                        choices=["full", "light", "conditional"],
                        help="HyDE 模式: full(3条假设评论) / light(1条综合假设评论) / "
                             "conditional(按问题类型决定是否启用)。指定该参数即视为启用 HyDE")
    parser.add_argument("--no-ranking", action="store_true", help="禁用排序")
    parser.add_argument("--no-expansion", action="store_true", help="禁用意图扩展")
    parser.add_argument("--no-bm25", action="store_true", help="禁用 BM25 召回")
    parser.add_argument("--no-vector", action="store_true", help="禁用向量召回")
    parser.add_argument("--no-reverse", action="store_true", help="禁用反向 Query 召回")
    parser.add_argument("--no-summary", action="store_true", help="禁用摘要召回")
    parser.add_argument("--topk", type=int, default=10, help="最终返回评论数 (默认10)")
    parser.add_argument("--model", type=str, default="qwen-plus", help="生成模型 (默认qwen-plus)")
    parser.add_argument("--ltr", action="store_true", help="使用 LTR 排序替代 RRF + 线性加权")
    parser.add_argument("--ltr-model", type=str, default=None, help="LTR 预训练模型路径")
    parser.add_argument("--diversity", type=str, default=None, choices=['mmr', 'dpp'],
                        help="启用多样性重排 (mmr 或 dpp)")
    parser.add_argument("--diversity-lambda", type=float, default=0.7,
                        help="MMR 相关性权重 (0-1, 默认0.7)")

    args = parser.parse_args()

    # 解析 HyDE 开关与模式：
    #   - 指定 --hyde-mode 即启用 HyDE，并使用对应模式
    #   - 仅指定 --hyde 时启用 HyDE，模式默认为 full（兼容原有逻辑）
    #   - 都不指定则不启用 HyDE（baseline）
    if args.hyde_mode is not None:
        enable_hyde = True
        hyde_mode = args.hyde_mode
    elif args.hyde:
        enable_hyde = True
        hyde_mode = "full"
    else:
        enable_hyde = False
        hyde_mode = "full"

    # 检查环境变量
    env = check_env()

    # 确定 API Key
    intl_api_key = os.getenv("DASHSCOPE_INTL_API_KEY")
    if intl_api_key:
        import dashscope
        from clients import DASHSCOPE_INTL_API_BASE
        dashscope.base_http_api_url = DASHSCOPE_INTL_API_BASE
        api_key = intl_api_key
    else:
        api_key = env["DASHSCOPE_API_KEY"]

    data_dir = Path(__file__).parent / "data"

    # 检查数据文件（本地向量模式需要 inverted_index.pkl / filtered_comments.csv /
    # comment_vectors.npz；最后一个由 build_local_index.py 生成，缺失时在初始化阶段报错）
    for f in ["inverted_index.pkl", "filtered_comments.csv"]:
        if not (data_dir / f).exists():
            print(f"错误: 数据文件不存在: {data_dir / f}")
            sys.exit(1)

    # 初始化 RAG 系统
    print("正在初始化 RAG 系统...")
    from rag_system import HotelReviewRAG

    # 是否使用云端 DashVector（默认走本地 numpy 向量库）
    use_local_vectors = os.getenv("USE_DASHVECTOR", "").strip() != "1"

    rag = HotelReviewRAG(
        api_key=env["DASHSCOPE_API_KEY"],
        dashvector_api_key=os.getenv("DASHVECTOR_API_KEY"),
        dashvector_endpoint=os.getenv("DASHVECTOR_HOTEL_ENDPOINT"),
        data_dir=data_dir,
        intl_api_key=intl_api_key,
        generation_model=args.model,
        use_local_vectors=use_local_vectors,
    )
    print("RAG 系统初始化完成\n")

    # 执行查询
    print(f"用户问题: {args.query}")
    print("-" * 60)

    result = rag.query(
        args.query,
        enable_hyde=enable_hyde,
        hyde_mode=hyde_mode,
        enable_expansion=not args.no_expansion,
        enable_bm25=not args.no_bm25,
        enable_vector=not args.no_vector,
        enable_reverse=not args.no_reverse,
        enable_summary=not args.no_summary,
        enable_ranking=not args.no_ranking,
        ranking_topk=args.topk,
        use_ltr=args.ltr,
        ltr_model_path=args.ltr_model,
        diversity_method=args.diversity,
        diversity_lambda=args.diversity_lambda,
        print_response=True,
    )

    print("-" * 60)

    if args.verbose:
        print_rag_result(result)

    return result


if __name__ == "__main__":
    main()
