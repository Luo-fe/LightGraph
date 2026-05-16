import json
import logging
from pathlib import Path

import faiss
import numpy as np

from src.config.settings import EMBEDDING_DIM, FAISS_INDEX_PATH
from src.graph.glm_embedder import GLMEmbedder

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, embedder: GLMEmbedder | None = None, dim: int | None = None):
        self.embedder = embedder or GLMEmbedder()
        self.dim = dim or EMBEDDING_DIM
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict] = []
        self._initialize_index()

    def _initialize_index(self):
        self.index = faiss.IndexFlatIP(self.dim)
        logger.info(f'Faiss索引已初始化，维度={self.dim}')

    async def add_documents(self, documents: list[dict], text_key: str = 'content'):
        texts = [doc.get(text_key, json.dumps(doc, ensure_ascii=False)) for doc in documents]
        embeddings = await self.embedder.embed_batch(texts)
        embeddings_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)
        self.documents.extend(documents)
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
        logger.info(f'向量索引已加载，共 {self.index.ntotal} 条')
        return True

    async def close(self):
        await self.embedder.close()
