"""LTR（Learning to Rank）模块：使用 LightGBM LambdaRank 替代 RRF + 线性加权排序

核心思路：
  传统的 RRF 融合使用固定公式 (1/(k+rank)) 合并多路召回结果，再用人工设定的
  线性权重组合 6 个因子。LTR 模型通过机器学习自动学习最优的特征组合方式，
  能够捕捉特征之间的非线性交互，从而获得更好的排序效果。

架构变化：
  原流程: 多路召回 → RRF 融合 → Cross-encoder Rerank → 线性加权 → 排序
  新流程: 多路召回 → 候选收集 → LTR 特征提取 → LightGBM LambdaRank → 排序

训练策略：
  使用现有 MultiFactorRanker 的评分作为伪标签 (pseudo-labels)，训练 LightGBM
  LambdaRank 模型。虽然伪标签来自启发式方法，但 LTR 模型能够：
  1. 学习特征间的非线性交互（如"高 BM25 排名 + 高时效性"的联合效应）
  2. 通过 LambdaRank 的 listwise 损失函数优化排序指标 (NDCG)
  3. 在新查询上泛化，减少对人工权重的依赖
"""

import time
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# ── 特征归一化常量（从数据集统计得出） ─────────────────────────

NORM_LOG_LENGTH = 7.51       # max(log(comment_len + 1))
NORM_LOG_REVIEW = 6.32       # max(log(review_count + 1))
NORM_LOG_USEFUL = 3.64       # max(log(useful_count + 1))
NORM_QUALITY = 10.0          # quality_score 范围
NORM_SCORE = 5.0             # rating score 范围
HALF_LIFE_DAYS = 180


def _safe_float(val, default=0.0):
    """安全转换浮点数"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0):
    """安全转换整数"""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


class LTRRanker:
    """LTR 排序器：使用 LightGBM LambdaRank 学习最优排序函数

    替换原有的 RRF 融合 + MultiFactorRanker 两阶段流程，
    通过统一的 LTR 模型直接从多路召回候选中预测排序得分。

    Parameters
    ----------
    reranker : Reranker
        DashScope Reranker 实例，用于获取 cross-encoder 相关性得分。
    embedding_client : EmbeddingClient
        嵌入客户端，用于后续多样性重排的向量计算。
    model_path : str, optional
        预训练 LTR 模型路径，提供则直接加载。
    """

    def __init__(self, reranker, embedding_client, model_path: str = None):
        self.reranker = reranker
        self.embedding_client = embedding_client
        self.model = None
        self.trained = False

        if model_path and HAS_LIGHTGBM:
            try:
                self.model = lgb.Booster(model_file=model_path)
                self.trained = True
                print(f"LTR 模型已加载: {model_path}")
            except Exception as e:
                print(f"LTR 模型加载失败: {e}，将使用默认特征权重")

    # ── 特征提取 ──────────────────────────────────────────────

    def _extract_features(self, query: str, candidates: list[dict],
                          today: datetime | None = None) -> np.ndarray:
        """从候选文档中提取 LTR 特征向量

        为每个候选文档提取 17 维特征：
          0. bm25_signal       — BM25 路由信号 (1/rank or 0)
          1. vector_signal     — 向量路由信号
          2. reverse_signal    — 反向 Query 路由信号
          3. hyde_signal       — HyDE 路由信号
          4. num_routes        — 命中的路由数量
          5. best_route_rank   — 所有路由中的最佳排名
          6. avg_route_rank    — 所有路由的平均排名
          7. has_bm25          — 是否被 BM25 召回 (0/1)
          8. has_vector        — 是否被向量召回 (0/1)
          9. has_reverse       — 是否被反向召回 (0/1)
         10. has_hyde          — 是否被 HyDE 召回 (0/1)
         11. norm_quality      — 归一化内容质量得分
         12. norm_length       — 归一化评论长度
         13. norm_review_cnt   — 归一化点评次数
         14. norm_useful_cnt   — 归一化有用数
         15. norm_rating       — 归一化酒店评分
         16. recency_score     — 时效性得分
        """
        if not today:
            today = datetime.today()

        n = len(candidates)
        features = np.zeros((n, 17), dtype=np.float64)

        for i, c in enumerate(candidates):
            route_ranks = c.get('route_ranks', {})
            meta = c.get('metadata', {})

            # 路由信号: 1/rank 作为该路由的贡献，未命中则为 0
            bm25_ranks = [r['rank'] for r in route_ranks.get('bm25', [])]
            vector_ranks = [r['rank'] for r in route_ranks.get('vector', [])]
            reverse_ranks = [r['rank'] for r in route_ranks.get('reverse', [])]
            hyde_ranks = [r['rank'] for r in route_ranks.get('hyde', [])]

            features[i, 0] = 1.0 / min(bm25_ranks) if bm25_ranks else 0.0
            features[i, 1] = 1.0 / min(vector_ranks) if vector_ranks else 0.0
            features[i, 2] = 1.0 / min(reverse_ranks) if reverse_ranks else 0.0
            features[i, 3] = 1.0 / min(hyde_ranks) if hyde_ranks else 0.0

            all_ranks = bm25_ranks + vector_ranks + reverse_ranks + hyde_ranks
            features[i, 4] = len([r for r in [bm25_ranks, vector_ranks,
                                              reverse_ranks, hyde_ranks] if r])
            features[i, 5] = min(all_ranks) if all_ranks else 999.0
            features[i, 6] = np.mean(all_ranks) if all_ranks else 999.0

            features[i, 7] = 1.0 if bm25_ranks else 0.0
            features[i, 8] = 1.0 if vector_ranks else 0.0
            features[i, 9] = 1.0 if reverse_ranks else 0.0
            features[i, 10] = 1.0 if hyde_ranks else 0.0

            # 内容质量特征
            quality = _safe_float(meta.get('quality_score', 0))
            features[i, 11] = quality / NORM_QUALITY

            comment_len = len(c.get('comment', ''))
            features[i, 12] = np.log(comment_len + 1) / NORM_LOG_LENGTH

            review_cnt = _safe_int(meta.get('review_count', 0))
            features[i, 13] = np.log(review_cnt + 1) / NORM_LOG_REVIEW

            useful_cnt = _safe_int(meta.get('useful_count', 0))
            features[i, 14] = np.log(useful_cnt + 1) / NORM_LOG_USEFUL

            rating = _safe_float(meta.get('score', 0))
            features[i, 15] = rating / NORM_SCORE

            # 时效性特征
            try:
                pub_date = pd.to_datetime(meta.get('publish_date', str(today.date())))
                days_ago = max(0, (today - pub_date).days)
            except Exception:
                days_ago = 365
            features[i, 16] = np.exp(-0.5 * days_ago / HALF_LIFE_DAYS)

        return features

    # ── 伪标签生成（用于训练） ──────────────────────────────────

    def generate_pseudo_labels(self, candidates: list[dict],
                               time_sensitivity: str = None,
                               today: datetime | None = None,
                               w_relevance: float = 0.40,
                               w_quality: float = 0.25,
                               w_length: float = 0.05,
                               w_review: float = 0.05,
                               w_useful: float = 0.05,
                               w_recency: float = 0.20,
                               base_decay: float = 0.5,
                               implied_boost: float = 0.5,
                               clear_boost: float = 0.5) -> np.ndarray:
        """使用现有的 MultiFactorRanker 逻辑生成伪标签

        将连续得分离散化为 5 级相关度标签 (0-4)，适配 LambdaRank 的 listwise 训练。
        """
        if not today:
            today = datetime.today()

        n = len(candidates)
        relevance = np.zeros(n)
        quality_arr = np.zeros(n)
        length_arr = np.zeros(n)
        review_arr = np.zeros(n)
        useful_arr = np.zeros(n)
        recency_arr = np.zeros(n)

        for i, c in enumerate(candidates):
            meta = c.get('metadata', {})
            rerank_s = _safe_float(c.get('rerank_score', 0))
            relevance[i] = rerank_s

            quality_arr[i] = _safe_float(meta.get('quality_score', 0)) / NORM_QUALITY

            c_len = len(c.get('comment', ''))
            length_arr[i] = np.log(c_len + 1) / NORM_LOG_LENGTH

            rc = _safe_int(meta.get('review_count', 0))
            review_arr[i] = np.log(rc + 1) / NORM_LOG_REVIEW

            uc = _safe_int(meta.get('useful_count', 0))
            useful_arr[i] = np.log(uc + 1) / NORM_LOG_USEFUL

            pub_date = pd.to_datetime(meta.get('publish_date', str(today.date())))
            days_ago = max(0, (today - pub_date).days)

            decay = base_decay
            if time_sensitivity == "implied":
                decay += implied_boost
            elif time_sensitivity == "clear":
                decay += implied_boost + clear_boost
            recency_arr[i] = np.exp(-decay * days_ago / HALF_LIFE_DAYS)

        final_scores = (
            w_relevance * relevance +
            w_quality * quality_arr +
            w_length * length_arr +
            w_review * review_arr +
            w_useful * useful_arr +
            w_recency * recency_arr
        )

        # 将连续得分映射为 0-4 的相关度等级
        if n >= 5:
            quantiles = np.percentile(final_scores, [20, 40, 60, 80])
            labels = np.zeros(n, dtype=np.int32)
            labels[final_scores >= quantiles[3]] = 4
            labels[(final_scores >= quantiles[2]) & (final_scores < quantiles[3])] = 3
            labels[(final_scores >= quantiles[1]) & (final_scores < quantiles[2])] = 2
            labels[(final_scores >= quantiles[0]) & (final_scores < quantiles[1])] = 1
        else:
            # 候选太少时直接用得分整数部分
            labels = np.clip((final_scores * 5).astype(np.int32), 0, 4)

        return labels

    # ── 模型训练 ──────────────────────────────────────────────

    def train(self, train_queries: list[str],
              train_candidates_list: list[list[dict]],
              train_time_sensitivities: list[str] = None,
              val_queries: list[str] = None,
              val_candidates_list: list[list[dict]] = None,
              val_time_sensitivities: list[str] = None,
              model_save_path: str = None,
              **rank_kwargs) -> dict:
        """训练 LTR 模型

        使用伪标签训练 LightGBM LambdaRank 模型。

        Parameters
        ----------
        train_queries : 训练查询列表
        train_candidates_list : 每个查询对应的候选文档列表
        train_time_sensitivities : 每个查询的时效性标签
        val_queries : 验证查询列表
        val_candidates_list : 验证候选文档列表
        val_time_sensitivities : 验证时效性标签
        model_save_path : 模型保存路径
        **rank_kwargs : 传递给 generate_pseudo_labels 的权重参数

        Returns
        -------
        dict : 训练评估结果
        """
        if not HAS_LIGHTGBM:
            raise ImportError("请安装 lightgbm: pip install lightgbm")

        print("=" * 60)
        print("LTR 模型训练")
        print("=" * 60)

        # 构建训练数据
        print("构建训练特征与标签...")
        X_train_list, y_train_list, q_train_list = [], [], []

        for idx, (query, candidates) in enumerate(zip(
            train_queries, train_candidates_list
        )):
            if len(candidates) < 2:
                continue

            # 1. 先跑 reranker 获取相关性得分（特征之一）
            try:
                documents = [c['comment'] for c in candidates]
                relevance_map = self.reranker.rerank(query, documents)
                for i, c in enumerate(candidates):
                    c['rerank_score'] = relevance_map.get(i, 0)
            except Exception as e:
                print(f"  警告: Reranker 调用失败 (query {idx}): {e}")
                for c in candidates:
                    c.setdefault('rerank_score', 0)

            # 2. 提取特征
            ts = train_time_sensitivities[idx] if train_time_sensitivities else None
            X = self._extract_features(query, candidates)
            y = self.generate_pseudo_labels(candidates, time_sensitivity=ts,
                                            **rank_kwargs)

            X_train_list.append(X)
            y_train_list.append(y)
            q_train_list.append(np.full(len(candidates), idx, dtype=np.int32))

        if not X_train_list:
            raise ValueError("训练数据为空，请检查候选文档是否有效")

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        q_train = np.concatenate(q_train_list)

        # 构建验证数据
        valid_sets = []
        eval_names = []
        if val_queries and val_candidates_list:
            X_val_list, y_val_list, q_val_list = [], [], []
            for idx, (query, candidates) in enumerate(zip(
                val_queries, val_candidates_list
            )):
                if len(candidates) < 2:
                    continue
                try:
                    documents = [c['comment'] for c in candidates]
                    relevance_map = self.reranker.rerank(query, documents)
                    for i, c in enumerate(candidates):
                        c['rerank_score'] = relevance_map.get(i, 0)
                except Exception:
                    for c in candidates:
                        c.setdefault('rerank_score', 0)

                ts = val_time_sensitivities[idx] if val_time_sensitivities else None
                X = self._extract_features(query, candidates)
                y = self.generate_pseudo_labels(candidates, time_sensitivity=ts,
                                                **rank_kwargs)
                X_val_list.append(X)
                y_val_list.append(y)
                q_val_list.append(np.full(len(candidates), idx, dtype=np.int32))

            X_val = np.vstack(X_val_list)
            y_val = np.concatenate(y_val_list)
            q_val = np.concatenate(q_val_list)
            valid_sets = [(X_val, y_val)]
            eval_names = ['valid']
            print(f"验证集: {len(val_candidates_list)} 个查询, {X_val.shape[0]} 个文档")

        print(f"训练集: {len(X_train_list)} 个查询, {X_train.shape[0]} 个文档")
        print(f"标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")

        # 训练 LightGBM LambdaRank
        print("\n训练 LightGBM LambdaRank...")
        train_data = lgb.Dataset(
            X_train, label=y_train, group=[len(x) for x in X_train_list]
        )
        train_data.set_feature_name([
            'bm25_signal', 'vector_signal', 'reverse_signal', 'hyde_signal',
            'num_routes', 'best_route_rank', 'avg_route_rank',
            'has_bm25', 'has_vector', 'has_reverse', 'has_hyde',
            'norm_quality', 'norm_length', 'norm_review_cnt',
            'norm_useful_cnt', 'norm_rating', 'recency_score'
        ])

        if valid_sets:
            valid_data = lgb.Dataset(
                X_val, label=y_val, group=[len(x) for x in X_val_list],
                reference=train_data
            )
            valid_sets = [valid_data]

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'min_sum_hessian_in_leaf': 1e-3,
            'lambda_l1': 0.01,
            'lambda_l2': 0.01,
            'verbose': 1,
            'seed': 42,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=valid_sets,
            valid_names=eval_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=20)
            ]
        )

        self.trained = True
        best_score = self.model.best_score

        # 特征重要性
        importance = dict(zip(
            self.model.feature_name(),
            self.model.feature_importance(importance_type='gain')
        ))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("\n特征重要性 (gain):")
        for name, imp in sorted_imp:
            print(f"  {name:25s}: {imp:.1f}")

        # 保存模型
        if model_save_path:
            self.model.save_file(model_save_path)
            print(f"\n模型已保存: {model_save_path}")

        return {
            'best_score': best_score,
            'feature_importance': importance,
            'num_features': X_train.shape[1],
            'num_train_queries': len(X_train_list),
            'num_train_docs': X_train.shape[0]
        }

    # ── LTR 排序 ──────────────────────────────────────────────

    def rank(self, query: str, candidates: list[dict],
             time_sensitivity: str = None,
             topk: int = 10,
             today: datetime | None = None) -> tuple[list[dict], dict]:
        """使用 LTR 模型对候选文档排序

        Parameters
        ----------
        query : 用户查询
        candidates : 候选文档列表（来自 retriever.retrieve_for_ltr()）
        time_sensitivity : 时效性标签
        topk : 返回结果数
        today : 当前日期

        Returns
        -------
        (ranked_results, timing_info)
        """
        ranking_start = time.time()
        if not today:
            today = datetime.today()

        if not candidates:
            return [], {'total': 0, 'rerank': 0, 'feature_extraction': 0, 'scoring': 0}

        # 1. Reranker 打分（作为 LTR 特征之一，也用于回退）
        rerank_start = time.time()
        documents = [c['comment'] for c in candidates]
        try:
            relevance_map = self.reranker.rerank(query, documents)
        except Exception as e:
            print(f"警告: Reranker 调用失败: {e}, 使用零值填充")
            relevance_map = {}
        rerank_time = time.time() - rerank_start

        for i, c in enumerate(candidates):
            c['rerank_score'] = relevance_map.get(i, 0)

        # 2. 特征提取
        feat_start = time.time()
        X = self._extract_features(query, candidates, today)
        feat_time = time.time() - feat_start

        # 3. LTR 模型预测 / 回退到启发式
        score_start = time.time()

        if self.trained and HAS_LIGHTGBM:
            ltr_scores = self.model.predict(X)
        else:
            # 未训练时的回退：使用特征加权和（模拟原 MultiFactorRanker 的线性组合）
            ltr_scores = (
                0.25 * X[:, 0] +    # bm25_signal
                0.20 * X[:, 1] +    # vector_signal
                0.10 * X[:, 2] +    # reverse_signal
                0.05 * X[:, 3] +    # hyde_signal
                0.05 * X[:, 4] / 4.0 +  # num_routes (normalized)
                0.05 * (1.0 - X[:, 5] / 999.0) +  # best_route_rank
                0.30 * X[:, 11] +   # norm_quality
                0.15 * X[:, 16] +   # recency_score
                0.10 * X[:, 15]     # norm_rating
            )

        score_time = time.time() - score_start

        # 4. 排序并返回 topk
        sorted_idx = np.argsort(ltr_scores)[::-1]

        ranked_results = []
        for rank, idx in enumerate(sorted_idx[:topk], 1):
            c = candidates[idx]
            result = {
                **c,
                'ltr_score': float(ltr_scores[idx]),
                'ltr_rank': rank,
                'rerank_score': float(c.get('rerank_score', 0)),
                'feature_vector': X[idx].tolist(),
            }
            ranked_results.append(result)

        timing_info = {
            'total': time.time() - ranking_start,
            'rerank': rerank_time,
            'feature_extraction': feat_time,
            'scoring': score_time
        }

        return ranked_results, timing_info

    # ── 模型持久化 ────────────────────────────────────────────

    def save(self, filepath: str):
        if self.model and HAS_LIGHTGBM:
            self.model.save_file(filepath)
            print(f"LTR 模型已保存: {filepath}")
        else:
            print("无可用模型保存")

    def load(self, filepath: str):
        if HAS_LIGHTGBM:
            self.model = lgb.Booster(model_file=filepath)
            self.trained = True
            print(f"LTR 模型已加载: {filepath}")
        else:
            print("lightgbm 未安装，无法加载模型")


class LTRTrainer:
    """LTR 训练辅助器：管理训练数据收集与模型训练流程"""

    def __init__(self, ltr_ranker: LTRRanker):
        self.ranker = ltr_ranker
        self.train_queries = []
        self.train_candidates = []
        self.train_ts = []
        self.val_queries = []
        self.val_candidates = []
        self.val_ts = []

    def add_train_sample(self, query: str, candidates: list[dict],
                         time_sensitivity: str = None):
        """添加训练样本"""
        self.train_queries.append(query)
        self.train_candidates.append(candidates)
        self.train_ts.append(time_sensitivity)

    def add_val_sample(self, query: str, candidates: list[dict],
                       time_sensitivity: str = None):
        """添加验证样本"""
        self.val_queries.append(query)
        self.val_candidates.append(candidates)
        self.val_ts.append(time_sensitivity)

    def train(self, model_save_path: str = None, **kwargs) -> dict:
        """执行训练"""
        return self.ranker.train(
            train_queries=self.train_queries,
            train_candidates_list=self.train_candidates,
            train_time_sensitivities=self.train_ts if any(t is not None for t in self.train_ts) else None,
            val_queries=self.val_queries if self.val_queries else None,
            val_candidates_list=self.val_candidates if self.val_candidates else None,
            val_time_sensitivities=self.val_ts if self.val_ts and any(t is not None for t in self.val_ts) else None,
            model_save_path=model_save_path,
            **kwargs
        )
