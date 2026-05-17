import asyncio
import json
import logging
import re

from src.config.settings import FEATURE_NAME_MAP
from src.data.structurer import MachiningFeature
from src.graph.glm_client import GLMClient

logger = logging.getLogger(__name__)

_NUMERIC_RANGES = {
    'length': (0.1, 2000.0),
    'width': (0.1, 2000.0),
    'diameter': (0.5, 1000.0),
    'depth': (0.1, 500.0),
    'roughness': (0.025, 100.0),
}

_PRECISION_PATTERN = re.compile(r'^IT\d{1,2}$')

_CONFIDENCE_THRESHOLD = 0.6

_REFINE_MAX_ROUNDS = 2


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


def _clean_precision(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        m = re.search(r'IT\s*(\d+)', value, re.IGNORECASE)
        if m:
            return f'IT{m.group(1)}'
        m = re.search(r'(\d+)', value)
        if m:
            return f'IT{m.group(1)}'
        return value
    if isinstance(value, (int, float)):
        return f'IT{int(value)}'
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
                    precision=_clean_precision(item.get('precision', item.get('精度'))),
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
                precision=_clean_precision(data.get('precision', data.get('精度'))),
                roughness=_clean_numeric(data.get('roughness', data.get('粗糙度'))),
            )
        except Exception as e:
            logger.warning(f'结构化特征提取失败: {e}')
            return None

    def _compute_feature_confidence(self, feature: MachiningFeature) -> float:
        total_fields = 7
        filled_fields = 0
        numeric_score = 0.0
        numeric_count = 0

        if feature.feature_type and feature.feature_type.strip():
            filled_fields += 1
            if feature.feature_type in FEATURE_NAME_MAP:
                numeric_score += 1.0
            else:
                numeric_score += 0.3
            numeric_count += 1

        for field_name, (low, high) in _NUMERIC_RANGES.items():
            value = getattr(feature, field_name, None)
            if value is not None:
                filled_fields += 1
                numeric_count += 1
                if low <= value <= high:
                    numeric_score += 1.0
                elif value < low * 0.5 or value > high * 2.0:
                    numeric_score += 0.0
                else:
                    numeric_score += 0.5

        if feature.precision is not None:
            filled_fields += 1
            numeric_count += 1
            if _PRECISION_PATTERN.match(feature.precision):
                it_num = int(feature.precision[2:])
                if 1 <= it_num <= 18:
                    numeric_score += 1.0
                else:
                    numeric_score += 0.3
            else:
                numeric_score += 0.2

        completeness = filled_fields / total_fields
        if numeric_count > 0:
            reasonableness = numeric_score / numeric_count
        else:
            reasonableness = 0.0

        confidence = completeness * 0.4 + reasonableness * 0.6
        return round(confidence, 4)

    def validate_feature(self, feature: MachiningFeature) -> dict:
        issues = []

        if not feature.feature_type or not feature.feature_type.strip():
            issues.append('特征类型为空')
        elif feature.feature_type not in FEATURE_NAME_MAP:
            issues.append(
                f'特征类型"{feature.feature_type}"不在支持列表中，'
                f'支持的类型：{", ".join(FEATURE_NAME_MAP.keys())}'
            )

        for field_name, (low, high) in _NUMERIC_RANGES.items():
            value = getattr(feature, field_name, None)
            if value is not None:
                if value < low:
                    issues.append(f'{field_name}={value}低于合理范围下限{low}')
                elif value > high:
                    issues.append(f'{field_name}={value}超过合理范围上限{high}')

        if feature.precision is not None:
            if not _PRECISION_PATTERN.match(feature.precision):
                issues.append(f'精度格式"{feature.precision}"不正确，应为IT1-IT18格式')
            else:
                it_num = int(feature.precision[2:])
                if it_num < 1 or it_num > 18:
                    issues.append(f'精度等级IT{it_num}超出有效范围(IT1-IT18)')

        confidence = self._compute_feature_confidence(feature)

        return {
            'valid': len(issues) == 0,
            'confidence': confidence,
            'issues': issues,
            'feature': feature.model_dump(),
        }

    async def refine_features(
        self,
        features: list[MachiningFeature],
        original_text: str,
        threshold: float = _CONFIDENCE_THRESHOLD,
        max_rounds: int = _REFINE_MAX_ROUNDS,
    ) -> list[MachiningFeature]:
        refined = list(features)

        for round_idx in range(max_rounds):
            low_confidence_indices = []
            low_confidence_features = []
            for i, feature in enumerate(refined):
                confidence = self._compute_feature_confidence(feature)
                if confidence < threshold:
                    low_confidence_indices.append(i)
                    low_confidence_features.append(feature)

            if not low_confidence_indices:
                logger.info(f'第{round_idx + 1}轮精炼：所有特征置信度均达标，跳过')
                break

            logger.info(
                f'第{round_idx + 1}轮精炼：发现{len(low_confidence_features)}个低置信度特征'
            )

            feature_summaries = []
            for idx, feat in enumerate(low_confidence_features):
                feat_dict = feat.model_dump()
                confidence = self._compute_feature_confidence(feat)
                feat_dict['confidence'] = confidence
                feature_summaries.append(feat_dict)

            messages = [
                {
                    'role': 'system',
                    'content': (
                        '你是一个CAD加工特征验证与修正专家。以下是从工艺文本中提取的加工特征，'
                        '但部分特征的置信度较低，可能存在错误或缺失。\n'
                        f'支持的加工特征类型：{", ".join(FEATURE_NAME_MAP.values())}。\n'
                        '请根据原始文本修正这些特征，确保：\n'
                        '1. 特征类型必须从支持列表中选择\n'
                        '2. 数值参数在合理范围内（长度0.1-2000mm，宽度0.1-2000mm，直径0.5-1000mm，深度0.1-500mm，粗糙度0.025-100μm）\n'
                        '3. 精度格式为IT1-IT18\n'
                        '4. 尽量补全缺失的字段\n'
                        '请以JSON格式输出，包含一个features列表，每个特征包含feature_type、length、width、diameter、depth、precision、roughness字段。'
                    ),
                },
                {
                    'role': 'user',
                    'content': (
                        f'原始文本：\n{original_text}\n\n'
                        f'需要修正的特征（含置信度）：\n{json.dumps(feature_summaries, ensure_ascii=False, indent=2)}\n\n'
                        '请修正上述特征并返回完整的特征列表。'
                    ),
                },
            ]

            try:
                result = await self.glm_client.chat_json(messages)
                if isinstance(result, list):
                    corrected_list = result
                elif isinstance(result, dict):
                    corrected_list = result.get('features', result.get('特征列表', []))
                    if not isinstance(corrected_list, list):
                        corrected_list = [result]
                else:
                    corrected_list = []

                corrected_features = []
                for item in corrected_list:
                    try:
                        feature_type = item.get('feature_type', item.get('特征类型', ''))
                        feature_type = self._normalize_feature_type(feature_type)
                        feat = MachiningFeature(
                            feature_type=feature_type,
                            length=_clean_numeric(item.get('length', item.get('长度'))),
                            width=_clean_numeric(item.get('width', item.get('宽度'))),
                            diameter=_clean_numeric(item.get('diameter', item.get('直径'))),
                            depth=_clean_numeric(item.get('depth', item.get('深度'))),
                            precision=_clean_precision(item.get('precision', item.get('精度'))),
                            roughness=_clean_numeric(item.get('roughness', item.get('粗糙度'))),
                        )
                        corrected_features.append(feat)
                    except Exception as e:
                        logger.warning(f'精炼特征解析失败: {e}')

                if corrected_features:
                    for i, idx in enumerate(low_confidence_indices):
                        if i < len(corrected_features):
                            new_confidence = self._compute_feature_confidence(corrected_features[i])
                            old_confidence = self._compute_feature_confidence(refined[idx])
                            if new_confidence > old_confidence:
                                refined[idx] = corrected_features[i]
                                logger.info(
                                    f'特征{idx}置信度从{old_confidence}提升至{new_confidence}'
                                )
            except Exception as e:
                logger.warning(f'第{round_idx + 1}轮精炼失败: {e}')

        return refined

    async def extract_and_validate(
        self,
        text: str,
        threshold: float = _CONFIDENCE_THRESHOLD,
        max_rounds: int = _REFINE_MAX_ROUNDS,
    ) -> dict:
        features = await self.extract_from_text(text)

        validations = [self.validate_feature(f) for f in features]

        refined_features = await self.refine_features(
            features, text, threshold=threshold, max_rounds=max_rounds
        )

        refined_validations = [self.validate_feature(f) for f in refined_features]

        return {
            'features': refined_features,
            'validations': refined_validations,
            'original_features': features,
            'original_validations': validations,
            'refined': features != refined_features,
        }

    async def batch_extract(self, texts: list[str]) -> list[list[MachiningFeature]]:
        tasks = [self.extract_from_text(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f'批量提取第{i}个文本失败: {result}')
                batch_results.append([])
            else:
                batch_results.append(result)
        return batch_results
