import json
import logging
from typing import Any, ClassVar

from openai import AsyncOpenAI

from src.config.settings import GLM_API_KEY, GLM_BASE_URL, GLM_MODEL

logger = logging.getLogger(__name__)


class GLMClient:
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.api_key = api_key or GLM_API_KEY
        self.base_url = base_url or GLM_BASE_URL
        self.model = model or GLM_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError('GLM_API_KEY未设置，请在.env文件中配置')

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def chat(self, messages: list[dict[str, str]], response_format: dict | None = None) -> str:
        kwargs: dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }
        if response_format:
            kwargs['response_format'] = response_format

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ''

    async def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        messages_copy = messages.copy()
        if not any('json' in m.get('content', '').lower() for m in messages_copy):
            messages_copy[-1]['content'] += '\n请以JSON格式输出结果。'

        response_format = {'type': 'json_object'}
        result = await self.chat(messages_copy, response_format=response_format)
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning('LLM返回非JSON格式，尝试提取JSON')
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
            raise ValueError(f'无法解析LLM返回的JSON: {result[:200]}') from None

    async def extract_entities_and_relations(self, text: str) -> dict[str, Any]:
        messages = [
            {
                'role': 'system',
                'content': (
                    '你是一个CAD工艺知识抽取专家。从给定的工艺文本中抽取实体和关系。'
                    '实体类型包括：加工特征、加工方法、刀具、机床、工艺参数、材料。'
                    '关系类型包括：使用、适用于、加工、包含、对应。'
                    '请以JSON格式输出，包含entities和relations两个列表。'
                ),
            },
            {'role': 'user', 'content': f'请从以下文本中抽取实体和关系：\n{text}'},
        ]
        return await self.chat_json(messages)

    async def recommend_process(self, feature_info: dict[str, Any], context: str = '') -> dict[str, Any]:
        feature_desc = ', '.join(f'{k}: {v}' for k, v in feature_info.items())
        messages = [
            {
                'role': 'system',
                'content': (
                    '你是一名资深中国工艺加工工程师。根据给定的加工特征与参数，'
                    '生成相应的加工方法与工艺路线，并给出推荐工艺参数。\n'
                    '严格要求：\n'
                    '1. 加工方法必须使用中文，如"粗铣-半精铣"、"钻-扩-铰"、"粗车-精车"等\n'
                    '2. 工艺路线必须使用中文描述，包含具体步骤和参数\n'
                    '3. 如果信息不足以确定具体方法，根据零件类型和特征给出最可能的推荐\n'
                    '4. parameters中每个字段必须是单个数字（整数或浮点数），禁止使用字符串、嵌套对象或字典\n'
                    '5. 参数合理范围：主轴转速200-8000rpm，进给速度30-800mm/min，'
                    '切削深度0.1-5mm，切削宽度0.5-10mm\n'
                    '6. 进给速度绝对不能小于30mm/min，这是机床最低进给速度\n'
                    '7. 刀具直径应与加工特征匹配（孔加工取孔径的0.8-1.0倍，铣削取槽宽的0.5-0.8倍）\n\n'
                    '【关键规则 - 加工方法选择】：\n'
                    '- 回转体特征（外圆、圆锥面、圆柱面、圆曲线、螺纹）→ 必须使用车削方法（粗车-精车/粗车-半精车-精车），禁止使用铣削\n'
                    '- 孔类特征（通孔、盲孔）→ 使用钻削方法（钻-扩-铰/钻-镗）\n'
                    '- 腔槽类特征（四边形腔、方形槽）→ 使用铣削方法（粗铣-半精铣/粗铣-精铣）\n'
                    '- 齿形特征 → 使用铣削方法\n'
                    '请严格按以下JSON格式输出：\n'
                    '{"machining_method": "中文加工方法", "process_route": "中文工艺路线", '
                    '"parameters": {"spindle_speed": 数字, "feed_rate": 数字, '
                    '"tool_diameter": 数字, "cutting_depth": 数字, "cutting_width": 数字}}'
                ),
            },
            {
                'role': 'user',
                'content': f'加工特征信息：{feature_desc}\n{context}',
            },
        ]
        return await self.chat_json(messages)

    async def close(self):
        await self.client.close()
