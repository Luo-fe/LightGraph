import json
import logging
import typing

logger = logging.getLogger(__name__)

try:
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False


class GLMCompatibleClient(OpenAIGenericClient if GRAPHITI_AVAILABLE else object):
    MAX_RETRIES: int = 2

    def __init__(self, config: LLMConfig | None = None, **kwargs):
        if not GRAPHITI_AVAILABLE:
            raise RuntimeError('graphiti-core未安装')
        super().__init__(config=config, **kwargs)

    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=16384,
        model_size=None,
    ) -> dict[str, typing.Any]:
        openai_messages = []
        for m in messages:
            content = m.content if hasattr(m, 'content') else str(m)
            role = m.role if hasattr(m, 'role') else 'user'
            if role == 'user':
                openai_messages.append({'role': 'user', 'content': content})
            elif role == 'system':
                openai_messages.append({'role': 'system', 'content': content})

        response_format = {'type': 'json_object'}

        if response_model is not None:
            schema = response_model.model_json_schema()
            schema_hint = json.dumps(schema, ensure_ascii=False)
            last_msg = openai_messages[-1]
            last_msg['content'] += (
                f'\n\n请严格按照以下JSON Schema格式输出结果：\n{schema_hint}'
            )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )

        result = response.choices[0].message.content or ''
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
            raise ValueError(f'无法解析LLM返回的JSON: {result[:200]}') from None
