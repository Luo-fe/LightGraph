import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class CollectionStats:
    total_files: int = 0
    json_files: int = 0
    csv_files: int = 0
    step_files: int = 0
    other_files: int = 0
    total_records: int = 0
    invalid_files: int = 0
    validation_errors: int = 0
    directories_scanned: int = 0
    collection_time: str = ''


class DataCollector:
    def __init__(self, raw_dir: Path | None = None, processed_dir: Path | None = None):
        self.raw_dir = raw_dir or RAW_DATA_DIR
        self.processed_dir = processed_dir or PROCESSED_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._stats = CollectionStats()

    def collect_from_directory(self, directory: Path, file_extensions: list[str] | None = None) -> list[Path]:
        if file_extensions is None:
            file_extensions = ['.json', '.csv', '.txt', '.xml', '.step', '.stp']

        collected = []
        for ext in file_extensions:
            collected.extend(directory.rglob(f'*{ext}'))

        self._stats.total_files += len(collected)
        for f in collected:
            suffix = f.suffix.lower()
            if suffix == '.json':
                self._stats.json_files += 1
            elif suffix == '.csv':
                self._stats.csv_files += 1
            elif suffix in ('.step', '.stp'):
                self._stats.step_files += 1
            else:
                self._stats.other_files += 1

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
                self._stats.invalid_files += 1

        self._stats.total_records += len(documents)
        logger.info(f'从 {directory} 收集到 {len(documents)} 条工艺文档')
        return documents

    def collect_from_csv(self, filepath: Path, encoding: str = 'utf-8') -> list[dict]:
        records = []
        try:
            with open(filepath, newline='', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))
        except (UnicodeDecodeError, csv.Error) as e:
            logger.warning(f'跳过无效CSV文件 {filepath}: {e}')
            self._stats.invalid_files += 1
            return []

        self._stats.csv_files += 1
        self._stats.total_records += len(records)
        logger.info(f'从CSV文件 {filepath} 收集到 {len(records)} 条记录')
        return records

    def collect_all_csv_from_directory(self, directory: Path, encoding: str = 'utf-8') -> list[dict]:
        all_records = []
        csv_files = list(directory.rglob('*.csv'))
        for f in csv_files:
            records = self.collect_from_csv(f, encoding=encoding)
            all_records.extend(records)
        logger.info(f'从目录 {directory} 的 {len(csv_files)} 个CSV文件中共收集到 {len(all_records)} 条记录')
        return all_records

    def collect_step_metadata(self, filepath: Path) -> dict:
        metadata = {
            'file_path': str(filepath),
            'file_name': filepath.name,
            'file_size': 0,
            'file_extension': filepath.suffix.lower(),
            'header_info': '',
            'schema_name': '',
            'encoding': '',
            'modified_time': '',
        }
        try:
            stat = filepath.stat()
            metadata['file_size'] = stat.st_size
            metadata['modified_time'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            with open(filepath, encoding='utf-8', errors='ignore') as f:
                in_header = False
                for line in f:
                    stripped = line.strip()
                    if stripped == 'HEADER;':
                        in_header = True
                        continue
                    if in_header:
                        if stripped == 'ENDSEC;':
                            break
                        if 'FILE_SCHEMA' in stripped:
                            metadata['schema_name'] = stripped
                        elif 'FILE_DESCRIPTION' in stripped:
                            metadata['header_info'] = stripped
                        elif 'FILE_NAME' in stripped:
                            metadata['encoding'] = stripped
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f'无法读取STEP文件元数据 {filepath}: {e}')
            self._stats.invalid_files += 1

        self._stats.step_files += 1
        logger.info(f'收集STEP文件元数据: {filepath.name}')
        return metadata

    def collect_all_step_metadata(self, directory: Path) -> list[dict]:
        metadata_list = []
        step_files = list(directory.rglob('*.step')) + list(directory.rglob('*.stp'))
        for f in step_files:
            metadata = self.collect_step_metadata(f)
            metadata_list.append(metadata)
        logger.info(f'从目录 {directory} 收集到 {len(metadata_list)} 个STEP文件元数据')
        return metadata_list

    def validate_data_quality(
        self,
        data: list[dict],
        required_fields: list[str] | None = None,
        field_types: dict[str, type] | None = None,
        value_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> dict:
        if required_fields is None:
            required_fields = []
        if field_types is None:
            field_types = {}
        if value_ranges is None:
            value_ranges = {}

        report = {
            'total_records': len(data),
            'missing_fields': {},
            'type_errors': {},
            'range_errors': {},
            'valid_records': 0,
            'invalid_records': 0,
        }

        for idx, record in enumerate(data):
            is_valid = True

            for fld in required_fields:
                if fld not in record or record[fld] is None or record[fld] == '':
                    report['missing_fields'].setdefault(fld, []).append(idx)
                    is_valid = False

            for fld, expected_type in field_types.items():
                if fld in record and record[fld] is not None:
                    if not isinstance(record[fld], expected_type):
                        try:
                            if expected_type in (int, float):
                                expected_type(record[fld])
                            else:
                                raise ValueError
                        except (ValueError, TypeError):
                            report['type_errors'].setdefault(fld, []).append(idx)
                            is_valid = False

            for fld, (min_val, max_val) in value_ranges.items():
                if fld in record and record[fld] is not None:
                    try:
                        val = float(record[fld])
                        if val < min_val or val > max_val:
                            report['range_errors'].setdefault(fld, []).append(idx)
                            is_valid = False
                    except (ValueError, TypeError):
                        pass

            if is_valid:
                report['valid_records'] += 1
            else:
                report['invalid_records'] += 1

        self._stats.validation_errors = report['invalid_records']
        logger.info(
            f'数据质量校验完成: 总计{report["total_records"]}条, '
            f'有效{report["valid_records"]}条, 无效{report["invalid_records"]}条'
        )
        return report

    def collect_from_multiple_sources(
        self,
        directories: list[Path],
        file_extensions: list[str] | None = None,
    ) -> dict[str, list[Path]]:
        results: dict[str, list[Path]] = {}
        self._stats.directories_scanned = len(directories)

        for directory in directories:
            dir_key = str(directory)
            results[dir_key] = self.collect_from_directory(directory, file_extensions)

        logger.info(f'从 {len(directories)} 个目录收集到 {self._stats.total_files} 个文件')
        return results

    def get_stats(self) -> CollectionStats:
        self._stats.collection_time = datetime.now().isoformat()
        return self._stats

    def reset_stats(self) -> None:
        self._stats = CollectionStats()

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
