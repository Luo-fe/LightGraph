import hashlib
import json
import logging
import re
from pathlib import Path

from src.config.settings import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


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

    def save_cleaned_data(self, data: list[dict], filename: str = 'cleaned_data.json'):
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f'清洗数据已保存至 {filepath}')
        return filepath
