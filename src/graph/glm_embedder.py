import logging

from openai import AsyncOpenAI

from src.config.settings import EMBEDDING_DIM, GLM_API_KEY, GLM_BASE_URL, GLM_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class GLMEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_dim: int | None = None,
    ):
        self.api_key = api_key or GLM_API_KEY
        self.base_url = base_url or GLM_BASE_URL
        self.model = model or GLM_EMBEDDING_MODEL
        self.embedding_dim = embedding_dim or EMBEDDING_DIM

        if not self.api_key:
            raise ValueError('GLM_API_KEY未设置，请在.env文件中配置')

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def embed(self, text: str) -> list[float]:
        result = await self.client.embeddings.create(
            input=text, model=self.model
        )
        embedding = result.data[0].embedding
        return embedding[:self.embedding_dim]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = await self.client.embeddings.create(
            input=texts, model=self.model
        )
        return [d.embedding[:self.embedding_dim] for d in result.data]

    async def close(self):
        await self.client.close()
