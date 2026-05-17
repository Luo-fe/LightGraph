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

TOLERANCE_ENTITY_PATTERNS = [
    (r'GEOMETRIC_TOLERANCE\s*\(', 'geometric_tolerance'),
    (r'PLUS_MINUS_TOLERANCE\s*\(', 'plus_minus_tolerance'),
    (r'TOLERANCE_VALUE\s*\(', 'tolerance_value'),
    (r'DIMENSIONAL_TOLERANCE\s*\(', 'dimensional_tolerance'),
    (r'FLATNESS_TOLERANCE\s*\(', 'flatness_tolerance'),
    (r'CYLINDRICITY_TOLERANCE\s*\(', 'cylindricity_tolerance'),
    (r'ROUNDNESS_TOLERANCE\s*\(', 'roundness_tolerance'),
    (r'STRAIGHTNESS_TOLERANCE\s*\(', 'straightness_tolerance'),
    (r'POSITION_TOLERANCE\s*\(', 'position_tolerance'),
    (r'PERPENDICULARITY_TOLERANCE\s*\(', 'perpendicularity_tolerance'),
    (r'PARALLELISM_TOLERANCE\s*\(', 'parallelism_tolerance'),
    (r'ANGULARITY_TOLERANCE\s*\(', 'angularity_tolerance'),
    (r'CIRCULAR_RUNOUT_TOLERANCE\s*\(', 'circular_runout_tolerance'),
    (r'TOTAL_RUNOUT_TOLERANCE\s*\(', 'total_runout_tolerance'),
]

SURFACE_FINISH_PATTERNS = [
    (r'SURFACE_ROUGHNESS\s*\(', 'surface_roughness'),
    (r'SURFACE_TEXTURE\s*\(', 'surface_texture'),
    (r'MEASURE_REPRESENTATION_ITEM\s*\(\s*\'[^\']*RA[^\']*\'', 'ra_measure'),
    (r'MEASURE_REPRESENTATION_ITEM\s*\(\s*\'[^\']*ROUGHNESS[^\']*\'', 'roughness_measure'),
]

MATERIAL_SIZE_RULES = {
    '45号钢': {'diameter_range': (5, 150), 'length_range': (10, 600)},
    '40Cr钢': {'diameter_range': (15, 250), 'length_range': (10, 300)},
    'Q235钢': {'diameter_range': (30, 500), 'length_range': (5, 100)},
    '20CrMnTi钢': {'diameter_range': (20, 200), 'length_range': (10, 150)},
    'HT200铸铁': {'diameter_range': (50, 600), 'length_range': (10, 200)},
    '铝合金6061': {'diameter_range': (5, 200), 'length_range': (10, 400)},
}

FEATURE_CONFIDENCE_THRESHOLD = 0.3


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

        info['tolerance_info'] = self._extract_tolerance_info(content)
        info['surface_finish'] = self._extract_surface_finish(content)

        classified = self._classify_part_type(info)
        info['classified_category'] = classified
        if classified and classified != info.get('category', ''):
            info['category_cn'] = PART_CATEGORY_MAP.get(classified, info.get('category_cn', ''))

        info['estimated_material'] = self._estimate_material(info)

        info['summary'] = self._generate_summary(info)
        info['inferred_features'] = self._infer_features(info)
        info['feature_confidence'] = self._compute_feature_confidence(info)
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

    def _extract_tolerance_info(self, content: str) -> dict:
        result = {
            'has_tolerance': False,
            'tolerance_types': [],
            'tolerance_count': 0,
            'tolerance_values': [],
        }

        found_types = []
        for pattern, tol_type in TOLERANCE_ENTITY_PATTERNS:
            count = len(re.findall(pattern, content))
            if count > 0:
                found_types.append(tol_type)
                result['tolerance_count'] += count

        if found_types:
            result['has_tolerance'] = True
            result['tolerance_types'] = found_types

        tolerance_value_matches = re.findall(
            r'TOLERANCE_VALUE\s*\(\s*[^,]*\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)',
            content,
        )
        for match in tolerance_value_matches:
            try:
                lower = float(match[0].strip())
                upper = float(match[1].strip())
                result['tolerance_values'].append({
                    'lower': lower,
                    'upper': upper,
                    'range': round(abs(upper - lower), 4),
                })
            except (ValueError, IndexError):
                pass

        plus_minus_matches = re.findall(
            r'PLUS_MINUS_TOLERANCE\s*\(\s*[^,]*\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)',
            content,
        )
        for match in plus_minus_matches:
            try:
                upper = float(match[0].strip())
                lower = float(match[1].strip())
                result['tolerance_values'].append({
                    'lower': -abs(lower),
                    'upper': abs(upper),
                    'range': round(abs(upper) + abs(lower), 4),
                })
            except (ValueError, IndexError):
                pass

        return result

    def _extract_surface_finish(self, content: str) -> dict:
        result = {
            'has_surface_finish': False,
            'finish_types': [],
            'roughness_values': [],
            'min_roughness': None,
            'max_roughness': None,
        }

        found_types = []
        for pattern, finish_type in SURFACE_FINISH_PATTERNS:
            count = len(re.findall(pattern, content, re.IGNORECASE))
            if count > 0:
                found_types.append(finish_type)

        if found_types:
            result['has_surface_finish'] = True
            result['finish_types'] = found_types

        roughness_matches = re.findall(
            r'(?:RA|ROUGHNESS)\s*[:=]?\s*([0-9]+\.?[0-9]*)',
            content,
            re.IGNORECASE,
        )
        for val_str in roughness_matches:
            try:
                val = float(val_str)
                if 0.001 < val < 100:
                    result['roughness_values'].append(val)
            except ValueError:
                pass

        measure_roughness = re.findall(
            r'MEASURE_REPRESENTATION_ITEM\s*\(\s*\'[^\']*\'\s*,\s*([0-9]+\.?[0-9]*)',
            content,
        )
        for val_str in measure_roughness:
            try:
                val = float(val_str)
                if 0.001 < val < 100:
                    result['roughness_values'].append(val)
            except ValueError:
                pass

        if result['roughness_values']:
            result['roughness_values'] = sorted(set(round(v, 4) for v in result['roughness_values']))
            result['min_roughness'] = min(result['roughness_values'])
            result['max_roughness'] = max(result['roughness_values'])

        return result

    def _classify_part_type(self, info: dict) -> str:
        scores = {}

        for category, cat_info in CATEGORY_FEATURE_MAP.items():
            score = 0.0

            if category == 'bolt':
                if info.get('has_thread_features'):
                    score += 0.35
                if info.get('cylindrical_surfaces', 0) >= 1:
                    score += 0.15
                if info.get('circle_curves', 0) > info.get('cylindrical_surfaces', 0):
                    score += 0.15
                if not info.get('has_hole_features'):
                    score += 0.1
                if not info.get('has_gear_features'):
                    score += 0.05

            elif category == 'gear':
                if info.get('has_gear_features'):
                    score += 0.35
                if info.get('has_hole_features'):
                    score += 0.15
                if info.get('cylindrical_surfaces', 0) >= 2:
                    score += 0.1
                if info.get('bspline_surfaces', 0) > 0:
                    score += 0.15
                if info.get('ellipse_curves', 0) > 0:
                    score += 0.05

            elif category == 'nut':
                if info.get('has_thread_features'):
                    score += 0.3
                if info.get('has_hole_features'):
                    score += 0.25
                if info.get('cylindrical_surfaces', 0) >= 2:
                    score += 0.1
                if not info.get('has_gear_features'):
                    score += 0.05

            elif category == 'shaft':
                if info.get('has_rotation_features'):
                    score += 0.25
                if info.get('cylindrical_surfaces', 0) >= 2:
                    score += 0.15
                if info.get('conical_surfaces', 0) > 0:
                    score += 0.15
                if not info.get('has_hole_features'):
                    score += 0.1
                if not info.get('has_thread_features'):
                    score += 0.05

            elif category == 'flange':
                if info.get('has_hole_features'):
                    score += 0.3
                if info.get('has_flat_features'):
                    score += 0.2
                if info.get('cylindrical_surfaces', 0) >= 2:
                    score += 0.1
                if not info.get('has_thread_features'):
                    score += 0.1
                if not info.get('has_gear_features'):
                    score += 0.05

            diameter = info.get('estimated_diameter')
            length = info.get('estimated_length')
            typ_d = cat_info.get('typical_diameter')
            typ_l = cat_info.get('typical_length')

            if diameter and typ_d:
                if typ_d[0] <= diameter <= typ_d[1]:
                    score += 0.1
                else:
                    dist = min(abs(diameter - typ_d[0]), abs(diameter - typ_d[1]))
                    score += max(0, 0.1 - dist / 100)

            if length and typ_l:
                if typ_l[0] <= length <= typ_l[1]:
                    score += 0.1
                else:
                    dist = min(abs(length - typ_l[0]), abs(length - typ_l[1]))
                    score += max(0, 0.1 - dist / 100)

            scores[category] = score

        if not scores:
            return ''

        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        if best_score < 0.2:
            return ''

        return best_category

    def _estimate_material(self, info: dict) -> str:
        classified = info.get('classified_category', '') or info.get('category', '')
        cat_info = CATEGORY_FEATURE_MAP.get(classified, {})
        default_material = cat_info.get('material', '')

        diameter = info.get('estimated_diameter')
        length = info.get('estimated_length')

        if not diameter and not length:
            return default_material

        best_material = default_material
        best_score = 0.0

        for material, rules in MATERIAL_SIZE_RULES.items():
            score = 0.0
            d_range = rules.get('diameter_range')
            l_range = rules.get('length_range')

            if diameter and d_range:
                if d_range[0] <= diameter <= d_range[1]:
                    score += 0.5
                else:
                    dist = min(abs(diameter - d_range[0]), abs(diameter - d_range[1]))
                    score += max(0, 0.5 - dist / 100)

            if length and l_range:
                if l_range[0] <= length <= l_range[1]:
                    score += 0.5
                else:
                    dist = min(abs(length - l_range[0]), abs(length - l_range[1]))
                    score += max(0, 0.5 - dist / 100)

            if material == default_material:
                score += 0.2

            if score > best_score:
                best_score = score
                best_material = material

        return best_material

    def _compute_feature_confidence(self, info: dict) -> dict[str, float]:
        confidence = {}

        cyl = info.get('cylindrical_surfaces', 0)
        pla = info.get('planar_surfaces', 0)
        con = info.get('conical_surfaces', 0)
        tor = info.get('toroidal_surfaces', 0)
        bsp = info.get('bspline_surfaces', 0)
        cir = info.get('circle_curves', 0)
        ell = info.get('ellipse_curves', 0)

        if cyl > 0:
            confidence['圆柱面'] = min(1.0, 0.4 + cyl * 0.1)
        if pla > 0:
            confidence['平面'] = min(1.0, 0.3 + pla * 0.05)
        if con > 0:
            confidence['圆锥面'] = min(1.0, 0.5 + con * 0.15)
        if cir > 0:
            confidence['圆曲线'] = min(1.0, 0.3 + cir * 0.05)

        if info.get('has_hole_features'):
            hole_conf = 0.4
            if cyl >= 3:
                hole_conf += 0.2
            if cir >= 3:
                hole_conf += 0.15
            if pla >= 2:
                hole_conf += 0.1
            confidence['通孔'] = min(1.0, hole_conf)
        else:
            if cyl >= 2 and cir >= 2:
                confidence['通孔'] = 0.25

        if info.get('has_thread_features'):
            thread_conf = 0.3
            if tor > 0:
                thread_conf += 0.4
            if cyl > 0 and cir > cyl:
                thread_conf += 0.2
            confidence['螺纹'] = min(1.0, thread_conf)
        else:
            if tor > 0:
                confidence['螺纹'] = 0.35

        if info.get('has_gear_features'):
            gear_conf = 0.3
            if bsp > 0:
                gear_conf += 0.35
            if ell > 0 and cyl > 0:
                gear_conf += 0.2
            confidence['齿形'] = min(1.0, gear_conf)
        else:
            if bsp > 0:
                confidence['齿形'] = 0.35

        if info.get('has_rotation_features'):
            rot_conf = 0.4
            if cyl > 1:
                rot_conf += 0.2
            if con > 0:
                rot_conf += 0.15
            if cir > 2:
                rot_conf += 0.1
            confidence['外圆'] = min(1.0, rot_conf)

        return confidence

    def _infer_features(self, info: dict) -> list[str]:
        category = info.get('classified_category', '') or info.get('category', '')
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

        confidence = self._compute_feature_confidence(info)

        scored = []
        for feat in features:
            score = confidence.get(feat, 0.5)
            scored.append((feat, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        filtered = [feat for feat, score in scored if score >= FEATURE_CONFIDENCE_THRESHOLD]

        return filtered

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

        if info.get('estimated_material'):
            parts.append(f"推断材料: {info['estimated_material']}")

        tol_info = info.get('tolerance_info', {})
        if tol_info.get('has_tolerance'):
            parts.append(f"公差类型: {', '.join(tol_info.get('tolerance_types', []))}")

        sf_info = info.get('surface_finish', {})
        if sf_info.get('has_surface_finish'):
            parts.append(f"表面粗糙度: 有")

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
