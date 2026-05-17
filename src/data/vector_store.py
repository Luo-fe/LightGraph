import json
import math
import logging
from collections import Counter
from pathlib import Path

import faiss
import jieba
import numpy as np

from src.config.settings import EMBEDDING_DIM, FAISS_INDEX_PATH
from src.graph.glm_embedder import GLMEmbedder

logger = logging.getLogger(__name__)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.doc_lengths: list[int] = []

    def _tokenize(self, text: str) -> list[str]:
        return [w for w in jieba.lcut(text) if w.strip()]

    def build(self, documents: list[dict], text_key: str = 'content'):
        self.corpus = []
        self.df = {}
        self.doc_lengths = []

        for doc in documents:
            text = doc.get(text_key, json.dumps(doc, ensure_ascii=False))
            tokens = self._tokenize(text)
            self.corpus.append(tokens)
            self.doc_lengths.append(len(tokens))

            seen = set()
            for token in tokens:
                if token not in seen:
                    self.df[token] = self.df.get(token, 0) + 1
                    seen.add(token)

        n = len(self.corpus)
        self.avgdl = sum(self.doc_lengths) / n if n > 0 else 0.0
        self.idf = {}
        for token, freq in self.df.items():
            self.idf[token] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        if not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for idx, doc_tokens in enumerate(self.corpus):
            score = 0.0
            tf_map = Counter(doc_tokens)
            dl = self.doc_lengths[idx]

            for token in query_tokens:
                if token not in tf_map:
                    continue
                tf = tf_map[token]
                idf = self.idf.get(token, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl > 0 else tf
                score += idf * numerator / denominator

            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class VectorStore:
    def __init__(self, embedder: GLMEmbedder | None = None, dim: int | None = None):
        self.embedder = embedder or GLMEmbedder()
        self.dim = dim or EMBEDDING_DIM
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict] = []
        self.bm25: BM25 = BM25()
        self._initialize_index()

    def _initialize_index(self):
        self.index = faiss.IndexFlatIP(self.dim)
        logger.info(f'Faiss索引已初始化，维度={self.dim}')

    def _rebuild_bm25(self):
        self.bm25.build(self.documents)

    async def add_documents(self, documents: list[dict], text_key: str = 'content'):
        texts = [doc.get(text_key, json.dumps(doc, ensure_ascii=False)) for doc in documents]
        embeddings = await self.embedder.embed_batch(texts)
        embeddings_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)
        self.documents.extend(documents)
        self._rebuild_bm25()
        logger.info(f'已添加 {len(documents)} 条文档到向量库，总计 {len(self.documents)} 条')

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning('向量库为空')
            return []

        query_embedding = await self.embedder.embed(query)
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)

        scores, indices = self.index.search(query_np, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results

    def _bm25_search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        raw_results = self.bm25.search(query, top_k)
        results = []
        for idx, score in raw_results:
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        return results

    async def hybrid_search(
        self, query: str, top_k: int = 5, vector_weight: float = 0.7, bm25_weight: float = 0.3
    ) -> list[tuple[dict, float]]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning('向量库为空')
            return []

        vector_results = await self.search(query, top_k=top_k * 3)
        bm25_results = self._bm25_search(query, top_k=top_k * 3)

        rrf_scores: dict[int, float] = {}
        k = 60

        for rank, (doc, _score) in enumerate(vector_results):
            doc_idx = self.documents.index(doc)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + vector_weight / (k + rank + 1)

        for rank, (doc, _score) in enumerate(bm25_results):
            doc_idx = self.documents.index(doc)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + bm25_weight / (k + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]

        results = []
        for idx in sorted_indices:
            results.append((self.documents[idx], rrf_scores[idx]))
        return results

    async def search_with_filter(
        self, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[tuple[dict, float]]:
        results = await self.search(query, top_k=top_k * 4)

        if not filters:
            return results[:top_k]

        filtered = []
        for doc, score in results:
            match = True
            for key, value in filters.items():
                if isinstance(value, list):
                    if doc.get(key) not in value:
                        match = False
                        break
                else:
                    if doc.get(key) != value:
                        match = False
                        break
            if match:
                filtered.append((doc, score))

        return filtered[:top_k]

    def delete_documents(self, doc_ids: list[str], id_key: str = 'id') -> int:
        id_set = set(doc_ids)
        keep_indices = []
        for i, doc in enumerate(self.documents):
            if doc.get(id_key) not in id_set:
                keep_indices.append(i)

        deleted_count = len(self.documents) - len(keep_indices)
        if deleted_count == 0:
            return 0

        self.documents = [self.documents[i] for i in keep_indices]

        new_index = faiss.IndexFlatIP(self.dim)
        if keep_indices:
            keep_np = np.array([keep_indices], dtype=np.int64)
            vectors = self.index.reconstruct_n(0, self.index.ntotal)
            selected = vectors[keep_np[0]]
            new_index.add(selected)
        self.index = new_index

        self._rebuild_bm25()
        logger.info(f'已删除 {deleted_count} 条文档，剩余 {len(self.documents)} 条')
        return deleted_count

    def get_document_count(self) -> int:
        return len(self.documents)

    def save(self, path: str | None = None):
        save_path = path or FAISS_INDEX_PATH
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, save_path)
        meta_path = Path(save_path).with_suffix('.meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        logger.info(f'向量索引已保存至 {save_path}')

    def load(self, path: str | None = None):
        load_path = path or FAISS_INDEX_PATH
        if not Path(load_path).exists():
            logger.warning(f'索引文件不存在: {load_path}')
            return False

        self.index = faiss.read_index(load_path)
        meta_path = Path(load_path).with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, encoding='utf-8') as f:
                self.documents = json.load(f)
        self._rebuild_bm25()
        logger.info(f'向量索引已加载，共 {self.index.ntotal} 条')
        return True

    async def close(self):
        await self.embedder.close()
