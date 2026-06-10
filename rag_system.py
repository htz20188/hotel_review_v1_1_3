"""酒店评论 RAG 系统：完整的检索增强生成工作流"""

import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import dashvector
import chromadb

from config import TODAY, EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES
from clients import LLMClient, EmbeddingClient
from index import InvertedIndex
from intent import IntentRecognizer, IntentDetector, IntentExpander, HyDEGenerator, should_use_hyde
from retriever import HybridRetriever
from ranker import Reranker, MultiFactorRanker
from generator import ResponseGenerator
from ltr import LTRRanker
from diversity import DiversityReranker

from comment_analyzer import CommentAnalyzer


def load_comments_from_csv(csv_path: str) -> pd.DataFrame:
    """从本地 CSV 加载评论数据（替代 Insforge 数据库）"""
    df = pd.read_csv(csv_path, dtype={'_id': str})
    if '_id' in df.columns:
        df.set_index('_id', inplace=True)
    print(f"从 CSV 加载 {len(df)} 条评论数据")
    return df


class HotelReviewRAG:
    """酒店评论 RAG 系统：完整的检索增强生成工作流"""

    def __init__(self, api_key: str, dashvector_api_key: str, dashvector_endpoint: str,
                 data_dir: Path, df_comments: pd.DataFrame = None,
                 intl_api_key: str = None,
                 detection_model: str = "qwen-plus",
                 expansion_hyde_model: str = "qwen-flash",
                 generation_model: str = "qwen-plus",
                 use_local_vectors: bool = True):
        # 评论/反向向量库后端：
        #   use_local_vectors=True（默认）→ 使用本地 numpy 向量库（data/comment_vectors.npz），
        #     不依赖云端 DashVector 与 chromadb；反向 Query / 摘要路无本地数据，用空集合占位。
        #   use_local_vectors=False → 使用原云端 DashVector（需可用的 endpoint）。
        # 注：原云端 DashVector 集群已失效、且本机 chromadb/onnxruntime 原生组件不稳定，
        #     故默认走本地向量库；该后端对 LTR / 多样性等下游模块透明（接口一致）。
        if use_local_vectors:
            from local_vector import (
                LocalCommentCollection, EmptyCollection,
                EmptySummaryCollection, VECTORS_FILE,
            )
            npz_path = data_dir / VECTORS_FILE
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"本地评论向量文件不存在: {npz_path}\n"
                    "请先运行: python build_local_index.py"
                )
            self.comments_collection = LocalCommentCollection.load(npz_path)
            self.reverse_queries_collection = EmptyCollection()
            self.summaries_collection = EmptySummaryCollection()
        else:
            chroma_db_path = data_dir / "chroma_db"
            chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
            self.summaries_collection = chroma_client.get_collection("summary_database")
            dashvector_client = dashvector.Client(
                api_key=dashvector_api_key, endpoint=dashvector_endpoint
            )
            self.comments_collection = dashvector_client.get("comment_database")
            self.reverse_queries_collection = dashvector_client.get("reverse_query_database")

        # 加载倒排索引
        self.inverted_index = InvertedIndex()
        self.inverted_index.load(str(data_dir / "inverted_index.pkl"))

        # 加载评论数据
        if df_comments is not None:
            self.df_comments = df_comments
        else:
            csv_path = data_dir / "filtered_comments.csv"
            if csv_path.exists():
                self.df_comments = load_comments_from_csv(str(csv_path))
            else:
                raise FileNotFoundError(
                    f"评论数据文件不存在: {csv_path}\n"
                    "请将 filtered_comments.csv 放入 data/ 目录"
                )

        # 确定 API Key
        key = intl_api_key if intl_api_key else api_key

        # 初始化各组件
        detection_client = LLMClient(key, model=detection_model, json=True)
        expansion_hyde_client = LLMClient(key, model=expansion_hyde_model, json=True)
        embedding_client = EmbeddingClient(key)

        # 初始化评论分析器
        self.comment_analyzer = CommentAnalyzer()
        # 如果有训练好的模型，加载它
        model_path = data_dir / "models" / "bert_analyzer"
        if model_path.exists():
            self.comment_analyzer.load_pretrained(str(model_path))
            print("评论分析器已加载")
        else:
            print("评论分析器未训练，将使用启发式规则")

        self.intent_recognizer = IntentRecognizer(key)
        self.intent_detector = IntentDetector(
            detection_client, EXACT_ROOM_TYPES, FUZZY_ROOM_TYPES
        )
        self.intent_expander = IntentExpander(expansion_hyde_client)
        self.hyde_generator = HyDEGenerator(expansion_hyde_client)
        self.retriever = HybridRetriever(
            self.inverted_index, self.comments_collection,
            self.reverse_queries_collection, self.summaries_collection,
            embedding_client, self.df_comments, self.hyde_generator
        )
        self.reranker = Reranker(key)
        self.generator = ResponseGenerator(key, model=generation_model)

        # LTR 排序器（可选，需安装 lightgbm）
        self.ltr_ranker = LTRRanker(self.reranker, embedding_client)
        self._diversity_reranker = None  # 延迟初始化

    def query(self, user_query: str,
              route_topk: int = 150,
              retrieval_topk: int = 100,
              ranking_topk: int = 10,
              enable_expansion: bool = True,
              enable_bm25: bool = True,
              enable_vector: bool = True,
              enable_reverse: bool = True,
              enable_hyde: bool = False,
              hyde_mode: str = "full",
              enable_summary: bool = True,
              enable_ranking: bool = True,
              enable_generation: bool = True,
              print_response: bool = True,
              use_ltr: bool = False,
              ltr_model_path: str = None,
              diversity_method: str = None,
              diversity_lambda: float = 0.7,
              w_relevance: float = 0.40,
              w_quality: float = 0.25,
              w_length: float = 0.05,
              w_review: float = 0.05,
              w_useful: float = 0.05,
              w_recency: float = 0.20,
              base_decay: float = 0.5,
              implied_boost: float = 0.5,
              clear_boost: float = 0.5,
              half_life_days: int = 180,
              today: datetime | None = TODAY,
              history: dict | None = None) -> dict:
        total_start = time.time()
        timing = {}
        if not today:
            today = datetime.today()

        # 一、查询处理
        query_processing_start = time.time()

        intent_recognition_start = time.time()
        need_retrieval = self.intent_recognizer.recognize(user_query)
        timing['intent_recognition'] = time.time() - intent_recognition_start

        intent_detection_result = None
        intent_expansion_result = None
        timing['intent_detection'] = 0
        timing['intent_expansion'] = 0

        if need_retrieval:
            if enable_expansion:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_detect = executor.submit(
                        self._timed_call, self.intent_detector.detect, user_query
                    )
                    future_expand = executor.submit(
                        self._timed_call, self.intent_expander.expand, user_query
                    )
                    intent_detection_result, timing['intent_detection'] = future_detect.result()
                    intent_expansion_result, timing['intent_expansion'] = future_expand.result()
            else:
                intent_detection_result, timing['intent_detection'] = self._timed_call(
                    self.intent_detector.detect, user_query
                )
                intent_expansion_result, timing['intent_expansion'] = None, 0

        timing['query_processing_total'] = time.time() - query_processing_start

        # 直接回答
        if not need_retrieval:
            if enable_generation:
                first_token_base = time.time() - total_start
                response, ttft_model, subsequent, generation = self.generator.generate(
                    user_query, need_retrieval=False, print_response=print_response,
                    today=today, history=history
                )
                timing['ttft'] = first_token_base + ttft_model
                timing['ttft_model'] = ttft_model
                timing['subsequent'] = subsequent
                timing['generation'] = generation
            else:
                response = ""
                timing['ttft'] = 0
                timing['ttft_model'] = 0
                timing['subsequent'] = 0
                timing['generation'] = 0

            timing['total'] = time.time() - total_start
            return {
                'response': response,
                'references': {'comments': [], 'summaries': [], 'hyde_responses': {}},
                'query_processing': {
                    'intent_recognition': need_retrieval,
                    'intent_detection': None,
                    'intent_expansion': None
                },
                'hyde': {'requested': enable_hyde, 'mode': hyde_mode, 'used': False},
                'timing': timing
            }

        # 二、混合检索
        if enable_ranking:
            final_topk_for_retrieval = retrieval_topk
        else:
            final_topk_for_retrieval = ranking_topk

        rewritten_queries = (intent_expansion_result
                             if intent_expansion_result
                             else [{'query': user_query, 'weight': 1.0}])

        # HyDE 模式解析（方向 11）：决定本次是否实际启用 HyDE，以及生成模式
        #   full        : 启用，生成 3 条假设评论（原始逻辑）
        #   light       : 启用，仅生成 1 条综合性假设评论（低延迟）
        #   conditional : 由 should_use_hyde() 按问题类型决定是否启用；启用时用 full 生成
        used_hyde = enable_hyde
        if enable_hyde:
            if hyde_mode == "conditional":
                used_hyde = should_use_hyde(user_query)
                self.hyde_generator.mode = "full"
            elif hyde_mode == "light":
                self.hyde_generator.mode = "light"
            else:  # full（默认）
                self.hyde_generator.mode = "full"

        if use_ltr:
            # LTR 模式：收集所有候选文档（不进行 RRF 融合）
            comments, summaries, retrieval_timing, hyde_results = self.retriever.retrieve_for_ltr(
                rewritten_queries,
                room_type=intent_detection_result.get('room_type'),
                fuzzy_room_type=intent_detection_result.get('fuzzy_room_type'),
                topk=route_topk,
                enable_bm25=enable_bm25,
                enable_vector=enable_vector,
                enable_reverse=enable_reverse,
                enable_hyde=used_hyde,
                enable_summary=enable_summary
            )
        else:
            comments, summaries, retrieval_timing, hyde_results = self.retriever.retrieve(
                rewritten_queries,
                room_type=intent_detection_result.get('room_type'),
                fuzzy_room_type=intent_detection_result.get('fuzzy_room_type'),
                topk=route_topk,
                final_topk=final_topk_for_retrieval,
                enable_bm25=enable_bm25,
                enable_vector=enable_vector,
                enable_reverse=enable_reverse,
                enable_hyde=used_hyde,
                enable_summary=enable_summary
            )
        timing['retrieval'] = retrieval_timing

        # 三、排序
        if enable_ranking:
            if use_ltr:
                # LTR 排序：加载指定模型或使用回退特征加权
                if ltr_model_path:
                    self.ltr_ranker.load(ltr_model_path)
                ranked_comments, ranking_timing = self.ltr_ranker.rank(
                    user_query, comments,
                    time_sensitivity=intent_detection_result.get('time_sensitivity'),
                    topk=ranking_topk, today=today
                )
            else:
                # 传统 MultiFactorRanker
                ranker = MultiFactorRanker(
                    self.reranker,
                    w_relevance=w_relevance, w_quality=w_quality,
                    w_length=w_length, w_review=w_review,
                    w_useful=w_useful, w_recency=w_recency,
                    base_decay=base_decay, implied_boost=implied_boost,
                    clear_boost=clear_boost, half_life_days=half_life_days
                )
                ranked_comments, ranking_timing = ranker.rank(
                    user_query, comments,
                    time_sensitivity=intent_detection_result.get('time_sensitivity'),
                    topk=ranking_topk, today=today
                )

            # 多样性重排（MMR 或 DPP）
            diversity_eval = None
            if diversity_method and ranked_comments:
                if self._diversity_reranker is None or \
                   self._diversity_reranker.method != diversity_method or \
                   self._diversity_reranker.lambda_param != diversity_lambda:
                    self._diversity_reranker = DiversityReranker(
                        method=diversity_method,
                        lambda_param=diversity_lambda,
                        embedding_client=self.retriever.embedding_client
                    )
                ranked_comments, diversity_eval = self._diversity_reranker.rerank(
                    ranked_comments, query=user_query, topk=ranking_topk
                )
                # 将多样性评估指标合并到 timing 中
                ranking_timing['diversity_eval'] = diversity_eval

            timing['ranking'] = ranking_timing
        else:
            ranked_comments = comments
            timing['ranking'] = {'total': 0, 'rerank': 0, 'scoring': 0}

        # 四、回复生成
        if enable_generation:
            first_token_base = time.time() - total_start
            response, ttft_model, subsequent, generation = self.generator.generate(
                user_query,
                rewritten_queries=intent_expansion_result,
                ranked_comments=ranked_comments,
                summaries=summaries,
                need_retrieval=True,
                print_response=print_response,
                today=today,
                history=history
            )
            timing['ttft'] = first_token_base + ttft_model
            timing['ttft_model'] = ttft_model
            timing['subsequent'] = subsequent
            timing['generation'] = generation
        else:
            response = ""
            timing['ttft'] = 0
            timing['ttft_model'] = 0
            timing['subsequent'] = 0
            timing['generation'] = 0

        timing['total'] = time.time() - total_start

        # 五、构建返回结果
        processed_comments = []
        for c in ranked_comments:
            comment_data = {
                'comment_id': c['comment_id'],
                'comment': c['comment'],
                'route_ranks': c['route_ranks'],
                'metadata': c['metadata']
            }
            # RRF 相关字段（仅在传统模式下存在）
            if 'rrf_score' in c:
                comment_data['rrf_score'] = c['rrf_score']
                comment_data['rrf_rank'] = c['rrf_rank']
            # LTR 相关字段
            if 'ltr_score' in c:
                comment_data['ltr_score'] = c['ltr_score']
                comment_data['ltr_rank'] = c.get('ltr_rank', 0)
            # 多样性重排字段
            if 'diversity_rank' in c:
                comment_data['diversity_rank'] = c['diversity_rank']
                comment_data['original_rank'] = c.get('original_rank', 0)
            # 传统排序字段（仅在 MultiFactorRanker 路径存在）
            if 'rerank_score' in c:
                comment_data['rerank_score'] = c['rerank_score']
            if 'rerank_rank' in c:
                comment_data['rerank_rank'] = c['rerank_rank']
            if 'final_score' in c:
                comment_data['final_score'] = c['final_score']
            if 'final_rank' in c:
                comment_data['final_rank'] = c['final_rank']
            if 'feature_scores' in c:
                comment_data['feature_scores'] = c['feature_scores']
            processed_comments.append(comment_data)

        result = {
            'response': response,
            'references': {
                'comments': processed_comments,
                'summaries': summaries,
                'hyde_responses': hyde_results
            },
            'query_processing': {
                'intent_recognition': need_retrieval,
                'intent_detection': intent_detection_result,
                'intent_expansion': intent_expansion_result
            },
            'hyde': {
                'requested': enable_hyde,
                'mode': hyde_mode,
                'used': used_hyde
            },
            'timing': timing
        }

        # 附加多样性评估结果
        if enable_ranking and diversity_method and 'diversity_eval' in timing.get('ranking', {}):
            result['diversity_evaluation'] = timing['ranking']['diversity_eval']

        return result

    def _timed_call(self, func, *args) -> tuple:
        start = time.time()
        result = func(*args)
        return result, time.time() - start

    def analyze_comment(self, comment: str) -> tuple:
        """分析单个评论的质量和类别"""
        return self.comment_analyzer.predict(comment)

