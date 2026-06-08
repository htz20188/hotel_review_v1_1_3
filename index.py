"""基于 BM25 的倒排索引"""

import re
import math
import nltk
import jieba
import pickle
from pathlib import Path
from collections import Counter

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


class InvertedIndex:
    """基于 BM25 的倒排索引"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, stopwords_file: str = None):
        self.k1 = k1
        self.b = b
        self.index = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.num_docs = 0
        self.documents = {}

        self.stopwords = set()
        if stopwords_file and Path(stopwords_file).exists():
            with open(stopwords_file, encoding='utf-8') as f:
                self.stopwords.update([line.strip() for line in f])
            try:
                self.stopwords.update(nltk.corpus.stopwords.words('english'))
            except Exception:
                print("警告: 未能加载 NLTK 英文停用词")

        jieba.initialize()

    def tokenize(self, text: str) -> list[str]:
        text = re.sub(r'\s+', '', text)
        tokens = jieba.lcut(text)
        pattern = re.compile(r'[^一-鿿a-zA-Z]')
        tokens = [
            token.lower() for token in tokens
            if token.lower() not in self.stopwords and not pattern.search(token)
        ]
        return tokens

    def build(self, documents: dict[str, str]):
        self.documents = documents
        self.num_docs = len(documents)

        total_length = 0
        for doc_id, text in tqdm(documents.items(), desc="分词与统计"):
            tokens = self.tokenize(text)
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length

            term_freq = Counter(tokens)
            for term, freq in term_freq.items():
                if term not in self.index:
                    self.index[term] = {}
                self.index[term][doc_id] = freq

        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0
        print(f"倒排索引构建完成: {len(self.index)} 个词项, {self.num_docs} 篇文档")
        print(f"平均文档长度: {self.avg_doc_length:.2f} 个词")

    def search(self, query: str, topk: int = 10) -> list[tuple[str, float]]:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        idf = {}
        for term in query_tokens:
            if term in self.index:
                df = len(self.index[term])
                idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

        scores = {}
        for term in query_tokens:
            if term not in self.index:
                continue
            for doc_id, tf in self.index[term].items():
                if doc_id not in scores:
                    scores[doc_id] = 0
                doc_length = self.doc_lengths[doc_id]
                norm_factor = 1 - self.b + self.b * (doc_length / self.avg_doc_length)
                term_score = idf[term] * (tf * (self.k1 + 1)) / (tf + self.k1 * norm_factor)
                scores[doc_id] = scores.get(doc_id, 0) + term_score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return sorted_docs

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'num_docs': self.num_docs,
                'documents': self.documents,
                'k1': self.k1,
                'b': self.b,
                'stopwords': self.stopwords
            }, f)
        print(f"倒排索引已保存: {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.doc_lengths = data['doc_lengths']
            self.avg_doc_length = data['avg_doc_length']
            self.num_docs = data['num_docs']
            self.documents = data['documents']
            self.k1 = data['k1']
            self.b = data['b']
            self.stopwords = data.get('stopwords', set())
        print(f"倒排索引已加载")
