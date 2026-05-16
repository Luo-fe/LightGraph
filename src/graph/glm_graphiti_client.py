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


def _default_for_type(type_str: str):
    if type_str == 'array':
        return []
    if type_str == 'object':
        return {}
    if type_str == 'string':
        return ''
    if type_str in ('integer', 'number'):
        return 0
    if type_str == 'boolean':
        return False
    return None


def _fix_response_schema(data: dict, schema: dict) -> dict:
    properties = schema.get('properties', {})
    required_fields = schema.get('required', [])

    for key in required_fields:
        if key not in data:
            prop = properties.get(key, {})
            expected_type = prop.get('type', '')
            if expected_type == 'array':
                data[key] = []
            elif expected_type == 'object':
                items_schema = prop.get('properties', {})
                if items_schema:
                    data[key] = _build_default_object(prop)
                else:
                    data[key] = {}
            elif expected_type == 'string':
                data[key] = ''
            elif expected_type in ('integer', 'number'):
                data[key] = 0
            elif expected_type == 'boolean':
                data[key] = False
            else:
                if '$defs' in prop:
                    ref = prop.get('$ref', '')
                    if ref:
                        def_name = ref.split('/')[-1]
                        def_schema = schema.get('$defs', {}).get(def_name, {})
                        data[key] = _build_default_object(def_schema)
                    else:
                        data[key] = {}
                else:
                    data[key] = {}

    for key, prop in properties.items():
        if key not in data:
            continue
        expected_type = prop.get('type', '')
        if expected_type == 'array' and not isinstance(data[key], list):
            if isinstance(data[key], dict):
                items = []
                for v in data[key].values():
                    if isinstance(v, dict):
                        items.append(v)
                    elif isinstance(v, list):
                        items.extend(v)
                data[key] = items
            elif data[key] is None:
                data[key] = []
            else:
                data[key] = [data[key]]
        elif expected_type == 'array' and isinstance(data[key], list):
            items_schema = prop.get('items', {})
            item_type = items_schema.get('type', '')
            if item_type == 'object':
                data[key] = [
                    v if isinstance(v, dict) else {'value': v} for v in data[key]
                ]
            items_ref = items_schema.get('$ref', '')
            if items_ref:
                def_name = items_ref.split('/')[-1]
                def_schema = schema.get('$defs', {}).get(def_name, {})
                if def_schema:
                    data[key] = [
                        _fill_required_fields(item, def_schema, schema.get('$defs', {}))
                        for item in data[key]
                        if isinstance(item, dict)
                    ]

        elif expected_type == 'object' and isinstance(data[key], dict):
            nested_required = prop.get('required', [])
            for rk in nested_required:
                if rk not in data[key]:
                    nested_props = prop.get('properties', {})
                    rk_prop = nested_props.get(rk, {})
                    data[key][rk] = _default_for_type(rk_prop.get('type', ''))

    return data


def _build_default_object(schema: dict) -> dict:
    result = {}
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    defs = schema.get('$defs', {})

    for key in required:
        prop = properties.get(key, {})
        expected_type = prop.get('type', '')
        ref = prop.get('$ref', '')

        if expected_type == 'array':
            result[key] = []
        elif expected_type == 'object' or ref:
            if ref:
                def_name = ref.split('/')[-1]
                def_schema = defs.get(def_name, {})
                if def_schema:
                    result[key] = _build_default_object(def_schema)
                else:
                    result[key] = {}
            else:
                result[key] = _build_default_object(prop)
        elif expected_type == 'string':
            result[key] = ''
        elif expected_type in ('integer', 'number'):
            result[key] = 0
        elif expected_type == 'boolean':
            result[key] = False
        else:
            result[key] = None

    return result


def _fill_required_fields(data: dict, schema: dict, defs: dict) -> dict:
    required = schema.get('required', [])
    properties = schema.get('properties', {})

    for key in required:
        if key not in data:
            prop = properties.get(key, {})
            expected_type = prop.get('type', '')
            ref = prop.get('$ref', '')

            if expected_type == 'array':
                data[key] = []
            elif expected_type == 'string':
                data[key] = ''
            elif expected_type in ('integer', 'number'):
                data[key] = 0
            elif expected_type == 'boolean':
                data[key] = False
            elif ref:
                def_name = ref.split('/')[-1]
                def_schema = defs.get(def_name, {})
                if def_schema:
                    data[key] = _build_default_object(def_schema)
                else:
                    data[key] = {}
            else:
                data[key] = {}

    return data


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
        schema = None

        if response_model is not None:
            schema = response_model.model_json_schema()
            schema_hint = json.dumps(schema, ensure_ascii=False)
            last_msg = openai_messages[-1]
            last_msg['content'] += (
                f'\n\n请严格按照以下JSON Schema格式输出结果：\n{schema_hint}'
            )

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=response_format,
                )

                result = response.choices[0].message.content or ''
                try:
                    parsed = json.loads(result)
                except json.JSONDecodeError:
                    start = result.find('{')
                    end = result.rfind('}') + 1
                    if start >= 0 and end > start:
                        parsed = json.loads(result[start:end])
                    else:
                        if attempt < self.MAX_RETRIES:
                            logger.warning(f'JSON解析失败，重试({attempt+1}/{self.MAX_RETRIES})')
                            continue
                        raise ValueError(f'无法解析LLM返回的JSON: {result[:200]}') from None

                if schema is not None and isinstance(parsed, dict):
                    parsed = _fix_response_schema(parsed, schema)

                if response_model is not None:
                    try:
                        response_model.model_validate(parsed)
                    except Exception as validate_err:
                        if attempt < self.MAX_RETRIES:
                            logger.warning(
                                f'Pydantic验证失败({attempt+1}/{self.MAX_RETRIES}): {validate_err}'
                            )
                            continue
                        logger.warning(f'Pydantic验证最终失败，返回原始数据: {validate_err}')

                return parsed

            except (json.JSONDecodeError, ValueError):
                if attempt < self.MAX_RETRIES:
                    continue
                raise

        return parsed
