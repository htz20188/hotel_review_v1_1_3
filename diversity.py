"""多样性重排模块：MMR 和 DPP 算法

在 LTR 排序后对 Top-K 结果进行多样性重排，平衡相关性与多样性。

算法简介：
  MMR (Maximal Marginal Relevance):
    贪心选择策略，每次选择"与查询最相关且与已选结果最不相似"的文档。
    通过参数 λ 控制相关性-多样性权衡。

  DPP (Determinantal Point Processes):
    基于行列式点过程的概率模型，通过核矩阵 L 建模子集选择的联合概率。
    L_ij = q_i * S_ij * q_j，其中 q_i 为质量项，S_ij 为相似度项。
    最大化 det(L_Y) 等价于选择高质量且多样化的子集。

评估指标：
  - Average Pairwise Similarity (APS): 越低越多样
  - Relevance Retention (RR): 重排后保留的相关性得分比例
  - Diversity Gain (DG): 多样性提升幅度
"""

import time
import numpy as np
from typing import Literal


class DiversityReranker:
    """多样性重排器：支持 MMR 和 DPP 两种算法

    Parameters
    ----------
    method : 'mmr' | 'dpp'
        重排算法。
    lambda_param : float
        MMR 的 λ 参数（相关性权重），范围 [0, 1]。
        仅 MMR 使用，默认 0.7（偏重相关性）。
    embedding_client : EmbeddingClient, optional
        嵌入客户端，用于计算文档间相似度。若未提供则使用文本重叠度。
    """

    def __init__(self, method: Literal['mmr', 'dpp'] = 'mmr',
                 lambda_param: float = 0.7,
                 embedding_client=None):
        if method not in ('mmr', 'dpp'):
            raise ValueError(f"不支持的重排方法: {method}，可选 'mmr' 或 'dpp'")
        if not 0 <= lambda_param <= 1:
            raise ValueError(f"lambda_param 需在 [0,1] 范围内，当前: {lambda_param}")

        self.method = method
        self.lambda_param = lambda_param
        self.embedding_client = embedding_client

    def rerank(self, ranked_docs: list[dict],
               query: str = None,
               topk: int = None) -> tuple[list[dict], dict]:
        """对排序后的文档进行多样性重排

        Parameters
        ----------
        ranked_docs : 已排序文档列表（按 score 降序，来自 LTR 或 MultiFactorRanker）
            每个文档需包含:
            - 'comment': 评论文本
            - 'ltr_score' 或 'final_score': 相关性得分
            - 'comment_id': 文档 ID
        query : 用户查询（用于 MMR 的第一项相关性计算）
        topk : 返回结果数，默认全部

        Returns
        -------
        (reranked_docs, eval_metrics)
        """
        if not ranked_docs:
            return [], self._empty_metrics()

        if topk is None:
            topk = len(ranked_docs)
        topk = min(topk, len(ranked_docs))

        start_time = time.time()

        # 提取相关性得分
        scores = np.array([
            doc.get('ltr_score', doc.get('final_score', 0))
            for doc in ranked_docs
        ])

        # 归一化得分到 [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones_like(scores) * 0.5

        # 计算相似度矩阵
        sim_start = time.time()
        sim_matrix = self._compute_similarity_matrix(ranked_docs)
        sim_time = time.time() - sim_start

        # 执行重排
        algo_start = time.time()
        if self.method == 'mmr':
            selected_indices = self._mmr_select(norm_scores, sim_matrix, topk)
        else:
            selected_indices = self._dpp_select(norm_scores, sim_matrix, topk)
        algo_time = time.time() - algo_start

        # 构建结果
        reranked = []
        for new_rank, idx in enumerate(selected_indices, 1):
            doc = dict(ranked_docs[idx])
            doc['diversity_rank'] = new_rank
            doc['original_rank'] = idx + 1
            reranked.append(doc)

        # 计算评估指标
        metrics = self._evaluate(
            norm_scores, sim_matrix, selected_indices,
            len(ranked_docs), topk
        )
        metrics['algorithm_time'] = algo_time
        metrics['similarity_time'] = sim_time
        metrics['total_time'] = time.time() - start_time

        return reranked, metrics

    # ── 相似度计算 ────────────────────────────────────────────

    def _compute_similarity_matrix(self, docs: list[dict]) -> np.ndarray:
        """计算文档间的相似度矩阵"""
        n = len(docs)

        if self.embedding_client and n > 1:
            try:
                comments = [doc['comment'] for doc in docs]
                embeddings = self.embedding_client.embed_batch(comments)
                emb_matrix = np.array(embeddings, dtype=np.float64)
                # 余弦相似度
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                emb_matrix = emb_matrix / norms
                sim = emb_matrix @ emb_matrix.T
                sim = np.clip(sim, -1, 1)
                return sim
            except Exception as e:
                print(f"嵌入相似度计算失败 ({e})，回退到文本重叠度")

        # 回退：Jaccard 相似度
        token_sets = []
        for doc in docs:
            tokens = set(doc['comment'])
            token_sets.append(tokens)

        sim = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    sim[i, j] = 1.0
                else:
                    intersection = len(token_sets[i] & token_sets[j])
                    union = len(token_sets[i] | token_sets[j])
                    val = intersection / union if union > 0 else 0
                    sim[i, j] = val
                    sim[j, i] = val
        return sim

    # ── MMR 算法 ──────────────────────────────────────────────

    def _mmr_select(self, scores: np.ndarray, sim_matrix: np.ndarray,
                    topk: int) -> list[int]:
        """MMR 贪心选择

        MMR(d) = λ * rel(d) - (1-λ) * max_{j in S} sim(d, j)

        其中 S 为已选择文档集合。
        """
        n = len(scores)
        remaining = set(range(n))
        selected = []
        sim_to_selected = np.zeros(n)

        for _ in range(min(topk, n)):
            if not selected:
                # 第一个选择得分最高项
                best = int(np.argmax(scores))
            else:
                mmr_scores = (
                    self.lambda_param * scores -
                    (1 - self.lambda_param) * sim_to_selected
                )
                # 只考虑剩余候选项
                mmr_masked = np.full(n, -np.inf)
                for i in remaining:
                    mmr_masked[i] = mmr_scores[i]
                best = int(np.argmax(mmr_masked))

            selected.append(best)
            remaining.discard(best)

            # 更新与已选集合的最大相似度
            new_sim = sim_matrix[best]
            sim_to_selected = np.maximum(sim_to_selected, new_sim)

        return selected

    # ── DPP 算法 ──────────────────────────────────────────────

    def _dpp_select(self, scores: np.ndarray, sim_matrix: np.ndarray,
                    topk: int) -> list[int]:
        """DPP 贪心 MAP 推理

        构建核矩阵 L_ij = q_i * S_ij * q_j
        贪心选择最大化 log-det 增益的项。

        使用标准 DPP MAP inference 贪心算法:
          j = argmax_{i not in Y} log(L_{Y∪{i}}) - log(L_Y)
            = argmax_{i not in Y} log(q_i^2 * (1 - S_{i,Y} S_{Y,Y}^{-1} S_{Y,i}))
        简化为: j = argmax_{i not in Y} q_i^2 * (1 - S_{i,Y} S_{Y,Y}^{-1} S_{Y,i})
        """
        n = len(scores)
        # 质量项：使用得分（确保为正值）
        quality = np.maximum(scores, 0.01)

        selected = []
        remaining = list(range(n))

        # 第一个选择：质量最高的
        first = int(np.argmax(quality))
        selected.append(first)
        remaining.remove(first)

        # DPP 条件协方差（已选 → 待选的影响）
        # C = I - S[selected, :] @ inv(S[selected, selected]) @ S[:, selected]
        # 实际上我们使用增量更新方式

        for _ in range(1, min(topk, n)):
            best_idx = -1
            best_gain = -np.inf

            # 目前已选集合的相似度子矩阵的逆
            k = len(selected)
            S_YY = sim_matrix[np.ix_(selected, selected)]
            try:
                S_YY_inv = np.linalg.inv(S_YY + np.eye(k) * 1e-8)
            except np.linalg.LinAlgError:
                S_YY_inv = np.linalg.pinv(S_YY)

            for i in remaining:
                s_iY = sim_matrix[i, selected].reshape(-1, 1)
                # 条件方差: 1 - s_{i,Y} @ S_{Y,Y}^{-1} @ s_{Y,i}
                cond_var = 1.0 - (s_iY.T @ S_YY_inv @ s_iY).item()
                cond_var = max(cond_var, 1e-10)

                gain = quality[i] * quality[i] * cond_var
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx < 0:
                break

            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    # ── 评估指标 ──────────────────────────────────────────────

    def _evaluate(self, scores: np.ndarray, sim_matrix: np.ndarray,
                  selected: list[int], original_n: int, topk: int) -> dict:
        """计算多样性重排的评估指标"""
        topk_original = list(range(min(topk, original_n)))

        # 1. Average Pairwise Similarity (APS) — 越低越多样
        aps_original = self._compute_aps(sim_matrix, topk_original)
        aps_reranked = self._compute_aps(sim_matrix, selected)

        # 2. Relevance Retention (RR) — 相关性保留比例
        rr = self._compute_relevance_retention(scores, selected, topk_original)

        # 3. Diversity Gain (DG) — 多样性提升
        dg = (aps_original - aps_reranked) / max(aps_original, 1e-10)

        # 4. Unique Aspect Approximation — 近似独特方面覆盖率
        aspect_coverage = self._compute_aspect_coverage(sim_matrix, selected)

        return {
            'aps_original': float(aps_original),
            'aps_reranked': float(aps_reranked),
            'diversity_gain': float(dg),
            'relevance_retention': float(rr),
            'aspect_coverage': float(aspect_coverage),
            'lambda': self.lambda_param,
            'method': self.method,
            'num_selected': len(selected)
        }

    def _compute_aps(self, sim_matrix: np.ndarray, indices: list[int]) -> float:
        """计算选中集合的平均成对相似度"""
        if len(indices) <= 1:
            return 0.0
        vals = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                vals.append(sim_matrix[indices[i], indices[j]])
        return float(np.mean(vals)) if vals else 0.0

    def _compute_relevance_retention(self, scores: np.ndarray,
                                     selected: list[int],
                                     original_indices: list[int]) -> float:
        """计算重排后保留的相关性总分比例"""
        original_sum = scores[original_indices].sum()
        if original_sum == 0:
            return 1.0
        selected_sum = scores[selected].sum()
        return float(selected_sum / original_sum)

    def _compute_aspect_coverage(self, sim_matrix: np.ndarray,
                                 selected: list[int],
                                 threshold: float = 0.3) -> float:
        """近似的方面覆盖率：相似度低于阈值的文档对视为覆盖不同方面

        计算唯一方面数 / 总文档数，作为覆盖率的近似估计。
        """
        if len(selected) <= 1:
            return 1.0

        # 简单的贪心聚类：将文档划分为不同"方面"
        aspects = []
        for idx in selected:
            found = False
            for aspect in aspects:
                # 如果与方面中所有文档的相似度都高，则归入该方面
                if all(sim_matrix[idx, a] > threshold for a in aspect):
                    aspect.append(idx)
                    found = True
                    break
            if not found:
                aspects.append([idx])

        return len(aspects) / len(selected)

    def _empty_metrics(self) -> dict:
        return {
            'aps_original': 0, 'aps_reranked': 0,
            'diversity_gain': 0, 'relevance_retention': 1.0,
            'aspect_coverage': 1.0, 'lambda': self.lambda_param,
            'method': self.method, 'num_selected': 0,
            'algorithm_time': 0, 'similarity_time': 0, 'total_time': 0
        }


# ── 便捷函数 ──────────────────────────────────────────────────

def mmr_rerank(ranked_docs: list[dict], lambda_param: float = 0.7,
               query: str = None, topk: int = None,
               embedding_client=None) -> tuple[list[dict], dict]:
    """MMR 多样性重排的便捷函数"""
    reranker = DiversityReranker(
        method='mmr', lambda_param=lambda_param,
        embedding_client=embedding_client
    )
    return reranker.rerank(ranked_docs, query, topk)


def dpp_rerank(ranked_docs: list[dict],
               query: str = None, topk: int = None,
               embedding_client=None) -> tuple[list[dict], dict]:
    """DPP 多样性重排的便捷函数"""
    reranker = DiversityReranker(
        method='dpp', embedding_client=embedding_client
    )
    return reranker.rerank(ranked_docs, query, topk)
