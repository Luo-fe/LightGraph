import json
import logging
import re

from src.config.settings import FEATURE_NAME_MAP
from src.data.structurer import MachiningFeature
from src.graph.glm_client import GLMClient

logger = logging.getLogger(__name__)


def _clean_numeric(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        m = re.search(r'[\d]+\.?[\d]*', value)
        if m:
            return float(m.group())
        return None
    return None


class FeatureExtractor:
    def __init__(self, glm_client: GLMClient | None = None):
        self.glm_client = glm_client or GLMClient()

    async def extract_from_text(self, text: str) -> list[MachiningFeature]:
        messages = [
            {
                'role': 'system',
                'content': (
                    '你是一个CAD加工特征识别专家。从给定的工艺文本中识别加工特征。\n'
                    f'支持的加工特征类型：{", ".join(FEATURE_NAME_MAP.values())}。\n'
                    '严格要求：\n'
                    '1. 必须从支持的类型中选择最匹配的特征类型，使用中文名称\n'
                    '2. 对于每种特征，根据文本信息推断合理的尺寸参数（单位mm）\n'
                    '3. 如果文本没有明确尺寸，根据零件类型推断典型值（如螺栓直径6-24mm，轴直径20-80mm）\n'
                    '4. 精度默认IT7-IT8，粗糙度默认1.6-6.3\n'
                    '5. 每个特征必须包含feature_type(中文)、length、width、diameter、depth、precision、roughness字段\n'
                    '6. 所有数值字段(length/width/diameter/depth/roughness)必须是纯数字，不要包含单位或文字\n'
                    '请以JSON格式输出，包含一个features列表。'
                ),
            },
            {'role': 'user', 'content': f'请识别以下文本中的加工特征：\n{text}'},
        ]
        result = await self.glm_client.chat_json(messages)
        if isinstance(result, list):
            feature_list = result
        elif isinstance(result, dict):
            feature_list = result.get('features', result.get('特征列表', []))
            if not isinstance(feature_list, list):
                feature_list = [result]
        else:
            feature_list = []
        features = []
        for item in feature_list:
            try:
                feature_type = item.get('feature_type', item.get('特征类型', ''))
                feature_type = self._normalize_feature_type(feature_type)
                feature = MachiningFeature(
                    feature_type=feature_type,
                    length=_clean_numeric(item.get('length', item.get('长度'))),
                    width=_clean_numeric(item.get('width', item.get('宽度'))),
                    diameter=_clean_numeric(item.get('diameter', item.get('直径'))),
                    depth=_clean_numeric(item.get('depth', item.get('深度'))),
                    precision=item.get('precision', item.get('精度')),
                    roughness=_clean_numeric(item.get('roughness', item.get('粗糙度'))),
                )
                features.append(feature)
            except Exception as e:
                logger.warning(f'特征解析失败: {e}')
        return features

    async def extract_from_json(self, data: dict) -> list[MachiningFeature]:
        text = json.dumps(data, ensure_ascii=False)
        return await self.extract_from_text(text)

    def _normalize_feature_type(self, feature_type: str) -> str:
        for eng, chn in FEATURE_NAME_MAP.items():
            if chn in feature_type or eng in feature_type.lower():
                return eng
        return feature_type

    def extract_from_structured_data(self, data: dict) -> MachiningFeature | None:
        try:
            feature_type = self._normalize_feature_type(
                data.get('feature_type', data.get('特征类型', ''))
            )
            return MachiningFeature(
                feature_type=feature_type,
                length=_clean_numeric(data.get('length', data.get('长度'))),
                width=_clean_numeric(data.get('width', data.get('宽度'))),
                diameter=_clean_numeric(data.get('diameter', data.get('直径'))),
                depth=_clean_numeric(data.get('depth', data.get('深度'))),
                precision=data.get('precision', data.get('精度')),
                roughness=_clean_numeric(data.get('roughness', data.get('粗糙度'))),
            )
        except Exception as e:
            logger.warning(f'结构化特征提取失败: {e}')
            return None
