import pytest

from src.data.structurer import MachiningFeature, ProcessCase, ProcessParameter, DataStructurer
from src.data.cleaner import DataCleaner
from src.config.settings import MACHINING_FEATURES, FEATURE_NAME_MAP


def test_machining_feature_creation():
    feature = MachiningFeature(
        feature_type='rectangular_pocket',
        length=120.0,
        width=100.0,
        depth=50.0,
        precision='IT8',
        roughness=6.3,
    )
    assert feature.feature_type == 'rectangular_pocket'
    assert feature.length == 120.0
    assert feature.precision == 'IT8'


def test_process_case_creation():
    feature = MachiningFeature(
        feature_type='through_hole',
        diameter=10.0,
        depth=25.0,
        precision='IT7',
        roughness=1.6,
    )
    params = ProcessParameter(
        spindle_speed=2000,
        feed_rate=150,
        tool_diameter=10,
    )
    case = ProcessCase(
        id='test_001',
        feature=feature,
        machining_method='钻-扩-铰',
        process_route='钻孔→扩孔→铰孔',
        parameters=params,
    )
    assert case.machining_method == '钻-扩-铰'
    assert case.parameters.spindle_speed == 2000


def test_data_cleaner():
    cleaner = DataCleaner()
    text = '这是一段  <b>测试</b>  文本   '
    result = cleaner.clean_text(text)
    assert '<b>' not in result
    assert '  ' not in result


def test_data_cleaner_dedup():
    cleaner = DataCleaner()
    data = [{'id': '1', 'name': 'test'}, {'id': '1', 'name': 'test'}, {'id': '2', 'name': 'other'}]
    result = cleaner.deduplicate(data)
    assert len(result) == 2


def test_structurer():
    structurer = DataStructurer()
    raw = {
        'id': 'test_001',
        'feature_type': '四边形腔',
        'length': 120, 'width': 100,
        'depth': 50, 'precision': 'IT8', 'roughness': 6.3,
        'machining_method': '粗铣-半精铣',
        'process_route': '粗铣→半精铣',
        'spindle_speed': 3000, 'feed_rate': 800,
        'tool_diameter': 12,
    }
    case = structurer.structure_process_case(raw)
    assert case is not None
    assert case.feature.feature_type == 'rectangular_pocket'
    assert case.machining_method == '粗铣-半精铣'


def test_training_data_format():
    structurer = DataStructurer()
    feature = MachiningFeature(
        feature_type='through_hole',
        diameter=10.0,
        depth=25.0,
        precision='IT7',
        roughness=1.6,
    )
    case = ProcessCase(
        id='test_001',
        feature=feature,
        machining_method='钻-扩-铰',
        process_route='钻孔→扩孔→铰孔',
    )
    item = structurer.convert_to_training_format(case)
    assert 'instruction' in item.model_dump()
    assert '加工特征' in item.input_text


def test_feature_name_map():
    assert 'rectangular_pocket' in FEATURE_NAME_MAP
    assert FEATURE_NAME_MAP['rectangular_pocket'] == '四边形腔'
    assert len(MACHINING_FEATURES) == 12
