"""层次化摘要生成器 - 大模型课程高分方案"""

import time
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


class HierarchicalSummarizer:
    """
    层次化 Chain-of-Summary 摘要生成器
    
    核心创新：
    1. 多步推理：提取关键点 → 聚类 → 逐层摘要
    2. 事实自校验：LLM 生成后回查原文
    3. 可控层次：通过 depth 参数控制信息损失
    """
    
    def __init__(self, llm_client, embedding_client):
        self.llm = llm_client
        self.embedding = embedding_client
        
    def summarize(self, comments: List[dict], query: str, 
                  depth: int = 3, max_comments: int = 100) -> Dict:
        """
        生成层次化摘要
        
        Args:
            comments: 评论列表，每个包含 {'comment': str, 'metadata': dict}
            query: 用户查询
            depth: 层次深度 (2-4，推荐3)
            max_comments: 最大处理评论数
            
        Returns:
            {
                'final_summary': str,
                'hierarchy': [level1_summaries, level2_summaries, ...],
                'metrics': {...},
                'trace': {...}  # 可解释性追踪
            }
        """
        start_time = time.time()
        
        # 限制评论数量
        if len(comments) > max_comments:
            comments = self._select_diverse_comments(comments, max_comments)
        
        comments_text = [c['comment'] for c in comments]
        
        # === Level 1: 提取原子关键点 ===
        print(f"[1/{depth}] 提取关键点...")
        key_points = self._extract_key_points(comments_text, query)
        
        # === Level 2: 聚类 + 簇摘要 ===
        print(f"[2/{depth}] 聚类并生成簇摘要...")
        clusters = self._cluster_key_points(key_points, n_clusters=min(10, len(key_points)//3))
        level1_summaries = self._generate_cluster_summaries(clusters, query)
        
        # === Level 3: 主题层摘要 ===
        if depth >= 3:
            print(f"[3/{depth}] 生成主题层摘要...")
            themes = self._cluster_summaries(level1_summaries, n_clusters=min(5, len(level1_summaries)//2))
            level2_summaries = self._generate_theme_summaries(themes, query)
        else:
            level2_summaries = level1_summaries
        
        # === Level 4: 最终摘要 ===
        if depth >= 4:
            print(f"[4/{depth}] 生成最终摘要...")
            final_summary = self._generate_final_summary(level2_summaries, query)
        else:
            final_summary = self._merge_summaries(level2_summaries)
        
        # === 事实一致性校验 ===
        print("校验事实一致性...")
        verified_summary = self._fact_check(final_summary, comments_text)
        
        # 计算指标
        metrics = self._compute_metrics(verified_summary, comments_text, key_points)
        metrics['total_time'] = time.time() - start_time
        
        return {
            'final_summary': verified_summary,
            'hierarchy': [key_points, level1_summaries, level2_summaries],
            'metrics': metrics,
            'trace': {
                'n_key_points': len(key_points),
                'n_clusters': len(clusters),
                'compression_ratio': metrics['compression_ratio']
            }
        }
    
    # ========== Level 1: 关键点提取 ==========
    
    def _extract_key_points(self, comments: List[str], query: str) -> List[str]:
        """从评论中提取原子化关键点"""
        # 分批处理（避免超长上下文）
        batch_size = 20
        all_points = []
        
        for i in range(0, len(comments), batch_size):
            batch = comments[i:i+batch_size]
            
            prompt = f"""
你是一个酒店评论分析专家。任务：从以下评论中提取与「{query}」相关的关键信息点。

要求：
1. 每条信息点独立成行，不超过25字
2. 只提取客观事实，不添加主观判断
3. 相似信息合并为一条（如"早餐好吃"和"早餐丰富"合并为"早餐品种丰富，味道好"）
4. 避免重复

评论内容：
{chr(10).join([f'- {c[:200]}' for c in batch])}

输出格式（每行一个关键点，不要序号）：
早餐品种丰富，有中西式选择
早餐价格偏高，128元/位
餐厅环境好，但高峰期需要排队
"""
            response = self.llm.generate(prompt, temperature=0.3)
            points = [p.strip() for p in response.strip().split('\n') if p.strip() and not p.startswith('输出')]
            all_points.extend(points)
        
        # 去重（基于语义相似度）
        unique_points = self._deduplicate_points(all_points)
        return unique_points[:50]  # 最多50个关键点
    
    def _deduplicate_points(self, points: List[str]) -> List[str]:
        """基于嵌入相似度去重"""
        if len(points) <= 1:
            return points
        
        # 获取嵌入
        try:
            embeddings = self.embedding.embed_batch(points)
            sim_matrix = cosine_similarity(embeddings)
            
            # 贪心去重：相似度 > 0.85 视为重复
            keep = []
            for i, point in enumerate(points):
                duplicate = False
                for j in keep:
                    if sim_matrix[i, j] > 0.85:
                        duplicate = True
                        break
                if not duplicate:
                    keep.append(i)
            
            return [points[i] for i in keep]
        except:
            # 回退：基于字符串相似度
            unique = []
            for point in points:
                if not any(self._jaccard_sim(point, u) > 0.7 for u in unique):
                    unique.append(point)
            return unique
    
    # ========== Level 2: 聚类 + 簇摘要 ==========
    
    def _cluster_key_points(self, points: List[str], n_clusters: int) -> List[List[str]]:
        """将关键点聚类为组"""
        if len(points) <= n_clusters:
            return [[p] for p in points]
        
        try:
            embeddings = self.embedding.embed_batch(points)
            # 层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, 
                metric='cosine', 
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            
            clusters = [[] for _ in range(n_clusters)]
            for point, label in zip(points, labels):
                clusters[label].append(point)
            
            # 过滤空簇
            return [c for c in clusters if c]
        except:
            # 简单均分
            k = min(n_clusters, len(points))
            return [points[i::k] for i in range(k)]
    
    def _generate_cluster_summaries(self, clusters: List[List[str]], query: str) -> List[str]:
        """为每个簇生成摘要"""
        summaries = []
        
        for i, cluster in enumerate(clusters):
            prompt = f"""
请为以下关于「{query}」的关键点生成一个20-40字的概括性摘要。

关键点：
{chr(10).join([f'- {p}' for p in cluster])}

要求：
1. 概括核心观点
2. 如果包含正反两面都要提及
3. 只输出摘要，不要其他内容
"""
            summary = self.llm.generate(prompt, temperature=0.5)
            summaries.append(f"[{self._get_cluster_label(cluster)}] {summary}")
        
        return summaries
    
    def _get_cluster_label(self, cluster: List[str]) -> str:
        """自动提取簇标签（如"正面/负面/中立"）"""
        pos_words = ['好', '不错', '满意', '推荐', '喜欢', '丰富', '热情']
        neg_words = ['差', '不好', '贵', '慢', '旧', '吵', '失望']
        
        pos_count = sum(1 for p in cluster if any(w in p for w in pos_words))
        neg_count = sum(1 for p in cluster if any(w in p for w in neg_words))
        
        if pos_count > neg_count * 1.5:
            return "正面"
        elif neg_count > pos_count * 1.5:
            return "负面"
        else:
            return "中立"
    
    # ========== Level 3-4: 高层摘要 ==========
    
    def _cluster_summaries(self, summaries: List[str], n_clusters: int) -> List[List[str]]:
        """对摘要进行二次聚类"""
        return self._cluster_key_points(summaries, n_clusters)
    
    def _generate_theme_summaries(self, themes: List[List[str]], query: str) -> List[str]:
        """生成主题层摘要"""
        theme_summaries = []
        
        for theme in themes:
            prompt = f"""
综合以下关于「{query}」的子摘要，生成一个60-80字的主题摘要。

子摘要：
{chr(10).join([f'- {s}' for s in theme])}

要求：
1. 提炼核心观点
2. 指出观点之间的关联或矛盾
3. 语言流畅自然
"""
            summary = self.llm.generate(prompt, temperature=0.5)
            theme_summaries.append(summary)
        
        return theme_summaries
    
    def _generate_final_summary(self, theme_summaries: List[str], query: str) -> str:
        """生成最终摘要（150-200字）"""
        prompt = f"""
基于以下关于「{query}」的主题摘要，生成一份150-200字的最终回答。

主题摘要：
{chr(10).join([f'{i+1}. {s}' for i, s in enumerate(theme_summaries)])}

要求：
1. 首句给出直接回答
2. 后续展开不同维度的分析
3. 如有矛盾观点需要说明
4. 结尾可给出建议
"""
        return self.llm.generate(prompt, temperature=0.5)
    
    def _merge_summaries(self, summaries: List[str]) -> str:
        """简单合并（depth=2时使用）"""
        return '\n\n'.join(summaries)
    
    # ========== 事实校验 ==========
    
    def _fact_check(self, summary: str, original_comments: List[str]) -> str:
        """校验摘要中的事实是否在原文中存在"""
        prompt = f"""
对比以下摘要和原文，检查摘要中的每个事实是否在原文中有支撑。

摘要：
{summary}

原文片段（共{len(original_comments)}条）：
{chr(10).join([f'- {c[:100]}...' for c in original_comments[:10]])}

请输出：
1. 无法验证的事实（如有，列出并删除）
2. 修改后的摘要（只保留可验证的事实）

输出格式 JSON：
{{
    "unverified": ["事实1", "事实2"],
    "verified_summary": "修改后的摘要"
}}
"""
        try:
            response = self.llm.generate(prompt, temperature=0.2)
            import json
            data = json.loads(response)
            return data.get('verified_summary', summary)
        except:
            return summary
    
    # ========== 评估指标 ==========
    
    def _compute_metrics(self, summary: str, original: List[str], 
                         key_points: List[str]) -> Dict:
        """计算四个核心指标"""
        
        # 1. 压缩比
        original_len = sum(len(c) for c in original)
        summary_len = len(summary)
        compression_ratio = summary_len / max(original_len, 1)
        
        # 2. 关键点覆盖率（估算）
        covered = sum(1 for kp in key_points if any(
            word in summary for word in kp.split()[:3]
        ))
        coverage = covered / max(len(key_points), 1)
        
        # 3. 信息密度
        info_density = len(key_points) / max(len(summary.split()), 1)
        
        # 4. 去重率
        unique_sentences = len(set(summary.split('。')))
        total_sentences = len(summary.split('。'))
        dedup_rate = unique_sentences / max(total_sentences, 1)
        
        return {
            'compression_ratio': round(compression_ratio, 4),
            'coverage': round(coverage, 4),
            'info_density': round(info_density, 4),
            'dedup_rate': round(dedup_rate, 4),
            'n_key_points': len(key_points),
            'n_original_comments': len(original)
        }
    
    def _select_diverse_comments(self, comments: List[dict], n: int) -> List[dict]:
        """多样性采样（保留代表性评论）"""
        if len(comments) <= n:
            return comments
        
        # 基于嵌入的多样性采样（MMR 简化版）
        texts = [c['comment'] for c in comments]
        try:
            embeddings = self.embedding.embed_batch(texts)
            selected = [0]  # 选第一个
            remaining = list(range(1, len(comments)))
            
            while len(selected) < n and remaining:
                # 找与已选集合最不相似的
                scores = []
                for i in remaining:
                    max_sim = max(cosine_similarity([embeddings[i]], 
                                                    [embeddings[j]])[0][0] 
                                  for j in selected)
                    scores.append((i, -max_sim))  # 负号表示最小相似度
                best_i = max(scores, key=lambda x: x[1])[0]
                selected.append(best_i)
                remaining.remove(best_i)
            
            return [comments[i] for i in selected]
        except:
            return comments[:n]
    
    def _jaccard_sim(self, a: str, b: str) -> float:
        """Jaccard 相似度"""
        set_a = set(a)
        set_b = set(b)
        return len(set_a & set_b) / max(len(set_a | set_b), 1)


# ========== 评估脚本 ==========

def evaluate_summarizer(summarizer, test_queries: List[str], 
                        retrieval_results: List[list]) -> Dict:
    """
    评估摘要生成质量
    
    Args:
        summarizer: HierarchicalSummarizer 实例
        test_queries: 测试查询列表（5-10个）
        retrieval_results: 每个查询对应的检索结果（评论列表）
    """
    results = []
    
    for query, comments in zip(test_queries, retrieval_results):
        print(f"\n处理查询: {query}")
        
        # 生成摘要
        output = summarizer.summarize(comments, query, depth=3)
        
        # 收集指标
        results.append({
            'query': query,
            'metrics': output['metrics'],
            'summary': output['final_summary'][:200],
            'trace': output['trace']
        })
    
    # 汇总统计
    avg_metrics = {}
    for key in results[0]['metrics'].keys():
        if isinstance(results[0]['metrics'][key], (int, float)):
            avg_metrics[key] = np.mean([r['metrics'][key] for r in results])
    
    return {
        'per_query': results,
        'average': avg_metrics,
        'summary': f"平均压缩比 {avg_metrics['compression_ratio']:.2%} | "
                   f"覆盖率 {avg_metrics['coverage']:.1%} | "
                   f"信息密度 {avg_metrics['info_density']:.2f}"
    }
