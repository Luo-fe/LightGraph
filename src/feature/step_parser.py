import logging
import re
from pathlib import Path

from src.config.settings import PART_CATEGORY_MAP, TMCAD_DATASET_PATH

logger = logging.getLogger(__name__)


class STEPParser:
    def __init__(self, dataset_path: Path | None = None):
        self.dataset_path = dataset_path or TMCAD_DATASET_PATH

    def parse_step_file(self, filepath: Path) -> dict:
        try:
            with open(filepath, encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f'无法读取文件 {filepath}: {e}')
            return {}

        info = {
            'filename': filepath.name,
            'filepath': str(filepath),
            'category': self._get_category(filepath),
            'category_cn': PART_CATEGORY_MAP.get(self._get_category(filepath), ''),
            'file_size': filepath.stat().st_size,
        }

        info['protocol'] = self._extract_protocol(content)
        info['author'] = self._extract_author(content)
        info['organization'] = self._extract_organization(content)

        geometry = self._extract_geometry_info(content)
        info.update(geometry)

        info['summary'] = self._generate_summary(info)
        return info

    def _get_category(self, filepath: Path) -> str:
        for parent in filepath.parents:
            if parent.name in PART_CATEGORY_MAP:
                return parent.name
        return filepath.parent.name

    def _extract_protocol(self, content: str) -> str:
        match = re.search(r'FILE_SCHEMA\s*\(\s*\'([^\']+)\'\s*\)', content)
        if match:
            return match.group(1)
        return ''

    def _extract_author(self, content: str) -> str:
        match = re.search(r'AUTHOR\s*\(\s*\'([^\']*)\'', content)
        if match:
            return match.group(1)
        return ''

    def _extract_organization(self, content: str) -> str:
        match = re.search(r'ORGANIZATION\s*\(\s*\'[^\']*\'\s*,\s*\'([^\']*)\'', content)
        if match:
            return match.group(1)
        return ''

    def _extract_geometry_info(self, content: str) -> dict:
        info = {}

        closed_shell = len(re.findall(r'CLOSED_SHELL', content))
        info['shell_count'] = closed_shell

        advanced_face = len(re.findall(r'ADVANCED_FACE', content))
        info['face_count'] = advanced_face

        cylindrical = len(re.findall(r'CYLINDRICAL_SURFACE', content))
        planar = len(re.findall(r'PLANAR_SURFACE', content))
        conical = len(re.findall(r'CONICAL_SURFACE', content))
        toroidal = len(re.findall(r'TOROIDAL_SURFACE', content))
        bspline = len(re.findall(r'B_SPLINE_SURFACE', content))

        info['cylindrical_surfaces'] = cylindrical
        info['planar_surfaces'] = planar
        info['conical_surfaces'] = conical
        info['toroidal_surfaces'] = toroidal
        info['bspline_surfaces'] = bspline

        circle_curves = len(re.findall(r'CIRCLE\s*\(', content))
        line_curves = len(re.findall(r'\bLINE\s*\(', content))
        ellipse_curves = len(re.findall(r'ELLIPSE\s*\(', content))

        info['circle_curves'] = circle_curves
        info['line_curves'] = line_curves
        info['ellipse_curves'] = ellipse_curves

        info['has_hole_features'] = cylindrical >= 2 and circle_curves >= 2
        info['has_rotation_features'] = (cylindrical > 0 or conical > 0) and circle_curves > 0
        info['has_flat_features'] = planar >= 3

        return info

    def _generate_summary(self, info: dict) -> str:
        parts = [f"零件类型: {info.get('category_cn', '未知')}"]

        if info.get('face_count'):
            parts.append(f"面数: {info['face_count']}")
        if info.get('cylindrical_surfaces'):
            parts.append(f"圆柱面: {info['cylindrical_surfaces']}")
        if info.get('planar_surfaces'):
            parts.append(f"平面: {info['planar_surfaces']}")
        if info.get('conical_surfaces'):
            parts.append(f"圆锥面: {info['conical_surfaces']}")
        if info.get('circle_curves'):
            parts.append(f"圆曲线: {info['circle_curves']}")

        features = []
        if info.get('has_hole_features'):
            features.append('孔类特征')
        if info.get('has_rotation_features'):
            features.append('回转体特征')
        if info.get('has_flat_features'):
            features.append('平面特征')
        if features:
            parts.append(f"推断加工特征: {', '.join(features)}")

        return '；'.join(parts)

    def scan_dataset(self, max_per_category: int = 0) -> list[dict]:
        results = []
        if not self.dataset_path.exists():
            logger.error(f'数据集路径不存在: {self.dataset_path}')
            return results

        for category_dir in sorted(self.dataset_path.iterdir()):
            if not category_dir.is_dir():
                continue
            if category_dir.name not in PART_CATEGORY_MAP:
                continue

            count = 0
            step_files = sorted(
                list(category_dir.glob('*.stp')) + list(category_dir.glob('*.step'))
            )
            for step_file in step_files:
                info = self.parse_step_file(step_file)
                if info:
                    results.append(info)
                    count += 1
                if max_per_category > 0 and count >= max_per_category:
                    break

            logger.info(f'扫描 {category_dir.name}: {count} 个文件')

        logger.info(f'数据集扫描完成: 共 {len(results)} 个零件')
        return results

    def scan_category(self, category: str, max_files: int = 0) -> list[dict]:
        cat_dir = self.dataset_path / category
        if not cat_dir.exists():
            logger.error(f'类别目录不存在: {cat_dir}')
            return []

        results = []
        step_files = sorted(list(cat_dir.glob('*.stp')) + list(cat_dir.glob('*.step')))
        for step_file in step_files:
            info = self.parse_step_file(step_file)
            if info:
                results.append(info)
            if max_files > 0 and len(results) >= max_files:
                break

        return results
