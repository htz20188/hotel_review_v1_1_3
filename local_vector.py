"""本地向量库（纯 numpy 实现，零原生依赖）

背景：本项目原使用云端 DashVector 存放评论向量。云端集群失效后，这里改用
"本地 numpy 向量库"替代：把全部评论用 DashScope 嵌入后存成 .npz，查询时用
brute-force 余弦相似度取 topk。2171×1024 的规模下检索是毫秒级的，且不依赖
chromadb / onnxruntime 等在本机不稳定的原生组件。

对外暴露与 DashVector 一致的 `.query(vector=..., topk=..., filter=...)` 接口
（返回带 `.output` 列表、元素含 `.id` 与 `.fields`），因此 retriever.py 无需改动。

- LocalCommentCollection: 本地评论向量集合（评论 vector / hyde 路使用）。
- EmptyCollection:        空集合占位（反向 Query 路，已无本地数据）。
- EmptySummaryCollection: 空摘要集合占位（摘要路，沿用 chroma 查询风格的返回）。
"""

import re
import numpy as np

VECTORS_FILE = "comment_vectors.npz"  # 本地评论向量文件（位于 data/ 下）


# ── DashVector 风格的返回对象 ─────────────────────────────────────────

class _Doc:
    """模拟 DashVector 返回的单条文档：含 .id 与 .fields。"""

    def __init__(self, doc_id, fields=None):
        self.id = doc_id
        self.fields = fields or {}


class _Resp:
    """模拟 DashVector 返回对象：含 .output 列表。"""

    def __init__(self, output):
        self.output = output


_FILTER_RE = re.compile(r"^\s*(\w+)\s*=\s*'(.*)'\s*$")


def _parse_filter(filter_str):
    """把 DashVector 风格的过滤串 `field = 'value'` 解析为 (field, value)。

    解析失败或为空时返回 None（即不过滤）。
    """
    if not filter_str:
        return None
    m = _FILTER_RE.match(filter_str)
    if not m:
        return None
    return m.group(1), m.group(2)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class LocalCommentCollection:
    """本地评论向量集合：numpy brute-force 余弦检索，接口对齐 DashVector。"""

    def __init__(self, ids, embeddings, room_types, fuzzy_types):
        self.ids = list(ids)
        # 存归一化向量，查询时点积即为余弦相似度
        self.emb = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        self.room_types = np.asarray(room_types, dtype=object)
        self.fuzzy_types = np.asarray(fuzzy_types, dtype=object)

    @classmethod
    def load(cls, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        return cls(
            ids=data["ids"].tolist(),
            embeddings=data["emb"],
            room_types=data["room"],
            fuzzy_types=data["fuzzy"],
        )

    def query(self, vector, topk=10, filter=None):
        v = np.asarray(vector, dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        sims = self.emb @ v  # (N,) 余弦相似度

        # 房型过滤：把不满足条件的相似度置为 -inf
        parsed = _parse_filter(filter)
        if parsed:
            field, value = parsed
            if field == "room_type":
                mask = self.room_types == value
            elif field == "fuzzy_room_type":
                mask = self.fuzzy_types == value
            else:
                mask = np.ones(len(self.ids), dtype=bool)
            sims = np.where(mask, sims, -np.inf)

        k = min(topk, len(self.ids))
        if k <= 0:
            return _Resp([])
        # 取 topk（按相似度降序）
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]

        docs = []
        for i in idx:
            if not np.isfinite(sims[i]):
                continue  # 被过滤掉的项
            doc_id = self.ids[i]
            docs.append(_Doc(doc_id, {
                "comment_id": doc_id,
                "room_type": self.room_types[i],
                "fuzzy_room_type": self.fuzzy_types[i],
                "score": float(sims[i]),
            }))
        return _Resp(docs)


class EmptyCollection:
    """空集合占位：任何查询都返回空结果（用于停用的反向 Query 路）。"""

    def query(self, *args, **kwargs):
        return _Resp([])


class EmptySummaryCollection:
    """空摘要集合：沿用 chroma 的查询风格，返回与查询数等长的空结果。"""

    def query(self, query_embeddings=None, n_results=1, **kwargs):
        n = len(query_embeddings) if query_embeddings is not None else 1
        return {
            "ids": [[] for _ in range(n)],
            "documents": [[] for _ in range(n)],
            "metadatas": [[] for _ in range(n)],
        }
