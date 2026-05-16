import json
import logging
from pathlib import Path

from src.config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, raw_dir: Path | None = None, processed_dir: Path | None = None):
        self.raw_dir = raw_dir or RAW_DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_directory(self, directory: Path, file_extensions: list[str] | None = None) -> list[Path]:
        if file_extensions is None:
            file_extensions = ['.json', '.csv', '.txt', '.xml', '.step', '.stp']

        collected = []
        for ext in file_extensions:
            collected.extend(directory.rglob(f'*{ext}'))

        logger.info(f'从 {directory} 收集到 {len(collected)} 个文件')
        return collected

    def collect_process_documents(self, directory: Path) -> list[dict]:
        documents = []
        json_files = list(directory.rglob('*.json'))
        for f in json_files:
            try:
                with open(f, encoding='utf-8') as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f'跳过无效文件 {f}: {e}')

        logger.info(f'从 {directory} 收集到 {len(documents)} 条工艺文档')
        return documents

    def save_raw_data(self, data: list[dict], filename: str = 'collected_raw.json'):
        filepath = self.raw_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f'原始数据已保存至 {filepath}，共 {len(data)} 条')
        return filepath

    def load_raw_data(self, filename: str = 'collected_raw.json') -> list[dict]:
        filepath = self.raw_dir / filename
        if not filepath.exists():
            logger.warning(f'文件不存在: {filepath}')
            return []
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
