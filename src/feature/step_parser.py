import logging
import re
from pathlib import Path

from src.config.settings import PART_CATEGORY_MAP, TMCAD_DATASET_PATH

logger = logging.getLogger(__name__)

CATEGORY_FEATURE_MAP = {
    'bolt': {
        'primary_features': ['螺纹', '外圆', '圆柱面'],
        'typical_diameter': (6, 30),
        'typical_length': (20, 200),
        'material': '45号钢',
        'machine_tool': '数控车床',
        'process_type': 'turning',
    },
    'gear': {
        'primary_features': ['齿形', '外圆', '通孔', '圆柱面'],
        'typical_diameter': (20, 200),
        'typical_length': (10, 80),
        'material': '40Cr钢',
        'machine_tool': '数控车床+滚齿机',
        'process_type': 'turning+milling',
    },
    'nut': {
        'primary_features': ['螺纹', '通孔', '外圆', '圆柱面'],
        'typical_diameter': (6, 52),
        'typical_length': (5, 40),
        'material': '45号钢',
        'machine_tool': '数控车床',
        'process_type': 'turning',
    },
    'shaft': {
        'primary_features': ['外圆', '圆柱面', '圆锥面', '圆曲线'],
        'typical_diameter': (15, 100),
        'typical_length': (50, 500),
        'material': '45号钢',
        'machine_tool': '数控车床',
        'process_type': 'turning',
    },
    'flange': {
        'primary_features': ['外圆', '通孔', '圆柱面'],
        'typical_diameter': (50, 400),
        'typical_length': (10, 50),
        'material': 'Q235钢',
        'machine_tool': '数控车床',
        'process_type': 'turning',
    },
}


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
        info['file_description'] = self._extract_file_description(content)
        info['file_name_raw'] = self._extract_file_name(content)
        info['cad_software'] = self._extract_cad_software(content)

        geometry = self._extract_geometry_info(content)
        info.update(geometry)

        dimensions = self._extract_dimensions(content)
        info.update(dimensions)

        info['summary'] = self._generate_summary(info)
        info['inferred_features'] = self._infer_features(info)
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

    def _extract_file_description(self, content: str) -> str:
        match = re.search(r"FILE_DESCRIPTION\s*\(\s*\(\s*'([^']*)'", content)
        if match:
            return match.group(1)
        return ''

    def _extract_file_name(self, content: str) -> str:
        match = re.search(r"FILE_NAME\s*\(\s*'([^']*)'", content)
        if match:
            return match.group(1)
        return ''

    def _extract_cad_software(self, content: str) -> str:
        match = re.search(r"'(SolidWorks\s*\d+)'", content)
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
        info['has_thread_features'] = toroidal > 0 or (cylindrical > 0 and circle_curves > cylindrical)
        info['has_gear_features'] = bspline > 0 or (ellipse_curves > 0 and cylindrical > 0)

        return info

    def _extract_dimensions(self, content: str) -> dict:
        info = {}

        circle_radii = []
        for m in re.finditer(r'CIRCLE\s*\([^)]*,\s*([^,]+)\s*\)', content):
            try:
                radius = abs(float(m.group(1).strip()))
                if 0.1 < radius < 500:
                    circle_radii.append(radius)
            except (ValueError, IndexError):
                pass

        if circle_radii:
            info['circle_radii'] = sorted(set(round(r, 2) for r in circle_radii))
            info['max_radius'] = max(circle_radii)
            info['min_radius'] = min(circle_radii)
            info['estimated_diameter'] = round(max(circle_radii) * 2, 2)
        else:
            info['circle_radii'] = []
            info['max_radius'] = None
            info['min_radius'] = None
            info['estimated_diameter'] = None

        coords = re.findall(
            r'CARTESIAN_POINT\s*\(\s*\'[^\']*\'\s*,\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*\)',
            content,
        )
        if coords:
            x_vals = [abs(float(c[0])) for c in coords if self._is_valid_coord(c[0])]
            y_vals = [abs(float(c[1])) for c in coords if self._is_valid_coord(c[1])]
            z_vals = [abs(float(c[2])) for c in coords if self._is_valid_coord(c[2])]

            if x_vals and y_vals and z_vals:
                info['bbox_x'] = round(max(x_vals), 2)
                info['bbox_y'] = round(max(y_vals), 2)
                info['bbox_z'] = round(max(z_vals), 2)
                info['estimated_length'] = round(max(x_vals + y_vals + z_vals), 2)
            else:
                info['bbox_x'] = None
                info['bbox_y'] = None
                info['bbox_z'] = None
                info['estimated_length'] = None
        else:
            info['bbox_x'] = None
            info['bbox_y'] = None
            info['bbox_z'] = None
            info['estimated_length'] = None

        return info

    @staticmethod
    def _is_valid_coord(val: str) -> bool:
        try:
            v = float(val)
            return abs(v) < 10000
        except ValueError:
            return False

    def _infer_features(self, info: dict) -> list[str]:
        category = info.get('category', '')
        cat_info = CATEGORY_FEATURE_MAP.get(category, {})
        features = list(cat_info.get('primary_features', []))

        if info.get('has_hole_features') and '通孔' not in features:
            features.append('通孔')
        if info.get('has_thread_features') and '螺纹' not in features:
            features.append('螺纹')
        if info.get('has_gear_features') and '齿形' not in features:
            features.append('齿形')
        if info.get('conical_surfaces', 0) > 0 and '圆锥面' not in features:
            features.append('圆锥面')

        return features

    def _generate_summary(self, info: dict) -> str:
        parts = [f"零件类型: {info.get('category_cn', '未知')}"]

        if info.get('file_name_raw'):
            parts.append(f"文件名: {info['file_name_raw']}")

        if info.get('face_count'):
            parts.append(f"面数: {info['face_count']}")
        if info.get('cylindrical_surfaces'):
            parts.append(f"圆柱面: {info['cylindrical_surfaces']}")
        if info.get('planar_surfaces'):
            parts.append(f"平面: {info['planar_surfaces']}")
        if info.get('conical_surfaces'):
            parts.append(f"圆锥面: {info['conical_surfaces']}")
        if info.get('toroidal_surfaces'):
            parts.append(f"环面: {info['toroidal_surfaces']}")
        if info.get('circle_curves'):
            parts.append(f"圆曲线: {info['circle_curves']}")

        if info.get('estimated_diameter'):
            parts.append(f"估算最大直径: {info['estimated_diameter']}mm")
        if info.get('estimated_length'):
            parts.append(f"估算最大长度: {info['estimated_length']}mm")

        features = info.get('inferred_features', [])
        if not features:
            features = []
            if info.get('has_hole_features'):
                features.append('孔类特征')
            if info.get('has_rotation_features'):
                features.append('回转体特征')
            if info.get('has_flat_features'):
                features.append('平面特征')
            if info.get('has_thread_features'):
                features.append('螺纹特征')
            if info.get('has_gear_features'):
                features.append('齿形特征')
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
