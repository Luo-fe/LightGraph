import hashlib
import json
import logging
import re
from pathlib import Path

from src.config.settings import (
    FEATURE_NAME_MAP,
    PARAMETER_NAME_MAP,
    PROCESSED_DATA_DIR,
    PROCESS_PARAMETERS,
)

logger = logging.getLogger(__name__)

_NUMERIC_FIELD_RANGES: dict[str, tuple[float, float]] = {
    'spindle_speed': (100, 10000),
    'feed_rate': (10, 1000),
    'tool_diameter': (0.5, 500),
    'cutting_depth': (0.01, 100),
    'cutting_width': (0.01, 500),
    '主轴转速': (100, 10000),
    '进给速度': (10, 1000),
    '刀具直径': (0.5, 500),
    '切削深度': (0.01, 100),
    '切削宽度': (0.01, 500),
}

_REVERSE_FEATURE_MAP: dict[str, str] = {v: k for k, v in FEATURE_NAME_MAP.items()}
_REVERSE_PARAM_MAP: dict[str, str] = {v: k for k, v in PARAMETER_NAME_MAP.items()}
_FIELD_NAME_MAP: dict[str, str] = {}
_FIELD_NAME_MAP.update(_REVERSE_FEATURE_MAP)
_FIELD_NAME_MAP.update(_REVERSE_PARAM_MAP)


class DataCleaner:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seen_hashes: set[str] = set()

    def clean_text(self, text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\u4e00-\u9fff\w\s\.\,\;\:\-\(\)\[\]\{\}\/\\\%\+\=\@\#\!\?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_sensitive_info(self, text: str) -> str:
        text = re.sub(r'\b\d{17}[\dXx]\b', '[ID_REDACTED]', text)
        text = re.sub(r'\b1[3-9]\d{9}\b', '[PHONE_REDACTED]', text)
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL_REDACTED]', text)
        return text

    def compute_hash(self, data: dict) -> str:
        content = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()

    def deduplicate(self, data_list: list[dict]) -> list[dict]:
        unique = []
        for item in data_list:
            h = self.compute_hash(item)
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                unique.append(item)
        removed = len(data_list) - len(unique)
        if removed > 0:
            logger.info(f'去重完成，移除 {removed} 条重复数据')
        return unique

    def clean_process_record(self, record: dict) -> dict | None:
        cleaned = {}
        for key, value in record.items():
            if isinstance(value, str):
                value = self.clean_text(value)
                value = self.remove_sensitive_info(value)
            if value is not None and value != '':
                cleaned[key] = value

        if not cleaned:
            return None
        return cleaned

    def clean_dataset(self, data: list[dict]) -> list[dict]:
        cleaned = []
        for record in data:
            result = self.clean_process_record(record)
            if result:
                cleaned.append(result)

        cleaned = self.deduplicate(cleaned)
        logger.info(f'数据清洗完成: {len(data)} -> {len(cleaned)} 条')
        return cleaned

    def compute_quality_score(self, record: dict) -> float:
        if not record:
            return 0.0

        completeness_score = self._compute_completeness(record)
        numeric_score = self._compute_numeric_reasonability(record)
        text_score = self._compute_text_quality(record)

        score = 0.4 * completeness_score + 0.3 * numeric_score + 0.3 * text_score
        return round(max(0.0, min(1.0, score)), 4)

    def _compute_completeness(self, record: dict) -> float:
        expected_fields = set(PROCESS_PARAMETERS) | set(FEATURE_NAME_MAP.keys()) | set(PARAMETER_NAME_MAP.values())
        present = 0
        relevant = 0
        for field in expected_fields:
            if field in record:
                relevant += 1
                value = record[field]
                if value is not None and value != '' and value != []:
                    present += 1
        for key, value in record.items():
            if key not in expected_fields:
                relevant += 1
                if value is not None and value != '' and value != []:
                    present += 1
        if relevant == 0:
            return 0.0
        return present / relevant

    def _compute_numeric_reasonability(self, record: dict) -> float:
        numeric_fields = []
        for key, value in record.items():
            if isinstance(value, (int, float)) and key in _NUMERIC_FIELD_RANGES:
                numeric_fields.append((key, value))
        if not numeric_fields:
            return 1.0
        valid = 0
        for key, value in numeric_fields:
            low, high = _NUMERIC_FIELD_RANGES[key]
            if low <= value <= high:
                valid += 1
        return valid / len(numeric_fields)

    def _compute_text_quality(self, record: dict) -> float:
        text_fields = [v for v in record.values() if isinstance(v, str)]
        if not text_fields:
            return 1.0
        total_score = 0.0
        for text in text_fields:
            field_score = 1.0
            if len(text) == 0:
                field_score = 0.0
            elif len(text) > 5000:
                field_score *= 0.7
            control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
            if control_chars > 0:
                field_score *= max(0.5, 1.0 - control_chars / len(text))
            if re.search(r'(.)\1{5,}', text):
                field_score *= 0.8
            total_score += field_score
        return total_score / len(text_fields)

    def validate_numeric_fields(self, record: dict) -> dict[str, bool]:
        results = {}
        for key, value in record.items():
            if key in _NUMERIC_FIELD_RANGES and isinstance(value, (int, float)):
                low, high = _NUMERIC_FIELD_RANGES[key]
                results[key] = low <= value <= high
        return results

    def normalize_fields(self, record: dict) -> dict:
        normalized = {}
        for key, value in record.items():
            mapped_key = _FIELD_NAME_MAP.get(key, key)
            normalized[mapped_key] = value
        return normalized

    def clean_and_score_dataset(
        self,
        data: list[dict],
        quality_threshold: float = 0.3,
    ) -> list[dict]:
        scored = []
        for record in data:
            normalized = self.normalize_fields(record)
            cleaned = self.clean_process_record(normalized)
            if cleaned is None:
                continue
            score = self.compute_quality_score(cleaned)
            if score >= quality_threshold:
                cleaned['quality_score'] = score
                scored.append(cleaned)

        scored = self.deduplicate(scored)
        filtered = len(data) - len(scored)
        logger.info(
            f'清洗评分完成: {len(data)} -> {len(scored)} 条, '
            f'过滤 {filtered} 条低质量数据 (阈值={quality_threshold})'
        )
        return scored

    def filter_by_quality(
        self,
        data: list[dict],
        threshold: float = 0.5,
    ) -> list[dict]:
        filtered = [
            r for r in data
            if r.get('quality_score', 0.0) >= threshold
        ]
        removed = len(data) - len(filtered)
        if removed > 0:
            logger.info(f'质量过滤: 移除 {removed} 条低于阈值 {threshold} 的数据')
        return filtered

    def save_cleaned_data(self, data: list[dict], filename: str = 'cleaned_data.json'):
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f'清洗数据已保存至 {filepath}')
        return filepath
