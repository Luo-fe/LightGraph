import pytest

from src.data.structurer import (
    MachiningFeature,
    ProcessCase,
    ProcessParameter,
    DataStructurer,
    KnowledgeTriple,
)
from src.data.cleaner import DataCleaner
from src.data.collector import DataCollector, CollectionStats
from src.config.settings import MACHINING_FEATURES, FEATURE_NAME_MAP, MACHINING_METHODS
from src.feature.extractor import FeatureExtractor, _clean_numeric, _clean_precision
from src.feature.step_parser import STEPParser


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


def test_data_cleaner_quality_score():
    cleaner = DataCleaner()
    record = {
        'id': 'test_001',
        'feature_type': '通孔',
        'diameter': 10,
        'depth': 25,
        'precision': 'IT7',
        'roughness': 1.6,
        'machining_method': '钻-扩-铰',
        'spindle_speed': 2000,
        'feed_rate': 150,
    }
    score = cleaner.compute_quality_score(record)
    assert 0 <= score <= 1
    assert score > 0.5


def test_data_cleaner_normalize_fields():
    cleaner = DataCleaner()
    record = {'通孔': 1, '主轴转速': 2000}
    normalized = cleaner.normalize_fields(record)
    assert 'spindle_speed' in normalized


def test_data_cleaner_validate_numeric():
    cleaner = DataCleaner()
    record = {'spindle_speed': 5000, 'feed_rate': 200}
    result = cleaner.validate_numeric_fields(record)
    assert result.get('spindle_speed') is True
    assert result.get('feed_rate') is True


def test_data_cleaner_validate_numeric_out_of_range():
    cleaner = DataCleaner()
    record = {'spindle_speed': 99999}
    result = cleaner.validate_numeric_fields(record)
    assert result.get('spindle_speed') is False


def test_data_cleaner_clean_and_score():
    cleaner = DataCleaner()
    data = [
        {'id': '1', 'feature_type': '通孔', 'diameter': 10, 'depth': 25},
        {'id': '2', 'feature_type': '外圆', 'diameter': 50, 'length': 80},
    ]
    result = cleaner.clean_and_score_dataset(data, quality_threshold=0.1)
    assert len(result) >= 1
    for item in result:
        assert 'quality_score' in item


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


def test_structurer_knowledge_triples():
    structurer = DataStructurer()
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
        material='45号钢',
        machine_tool='CNC加工中心',
    )
    triples = structurer.generate_knowledge_triples(case)
    assert len(triples) > 0
    triple_types = {t.relation for t in triples}
    assert '使用' in triple_types
    assert '适用于' in triple_types


def test_structurer_auto_annotate():
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
        machining_method='',
        process_route='',
    )
    annotated = structurer.auto_annotate(case)
    assert annotated.machining_method != ''


def test_structurer_structure_and_annotate():
    structurer = DataStructurer()
    raw = {
        'id': 'test_002',
        'feature_type': '通孔',
        'diameter': 10,
        'depth': 25,
        'precision': 'IT7',
        'roughness': 1.6,
    }
    case = structurer.structure_and_annotate(raw)
    assert case is not None
    assert case.machining_method != ''


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


def test_knowledge_triple_model():
    triple = KnowledgeTriple(
        source_entity='通孔',
        relation='使用',
        target_entity='钻-扩-铰',
        attributes={'diameter': 10},
    )
    assert triple.source_entity == '通孔'
    assert triple.relation == '使用'
    assert triple.target_entity == '钻-扩-铰'


def test_feature_name_map():
    assert 'rectangular_pocket' in FEATURE_NAME_MAP
    assert FEATURE_NAME_MAP['rectangular_pocket'] == '四边形腔'
    assert len(MACHINING_FEATURES) == 12


def test_machining_methods_config():
    for feature in MACHINING_FEATURES:
        assert feature in MACHINING_METHODS
        assert len(MACHINING_METHODS[feature]) > 0


def test_clean_numeric():
    assert _clean_numeric(10) == 10.0
    assert _clean_numeric('25.5') == 25.5
    assert _clean_numeric('直径10mm') == 10.0
    assert _clean_numeric(None) is None
    assert _clean_numeric('abc') is None


def test_clean_precision():
    assert _clean_precision('IT7') == 'IT7'
    assert _clean_precision('it8') == 'IT8'
    assert _clean_precision(7) == 'IT7'
    assert _clean_precision(None) is None


def test_collector_stats():
    stats = CollectionStats()
    assert stats.total_files == 0
    assert stats.total_records == 0
    stats.total_files = 5
    stats.total_records = 10
    assert stats.total_files == 5


def test_collector_validate_quality():
    collector = DataCollector()
    data = [
        {'id': '1', 'feature_type': '通孔', 'diameter': 10},
        {'id': '2'},
    ]
    report = collector.validate_data_quality(
        data,
        required_fields=['id', 'feature_type'],
        field_types={'diameter': float},
        value_ranges={'diameter': (0.1, 500)},
    )
    assert report['total_records'] == 2
    assert report['valid_records'] >= 1


def test_step_parser_category_feature_map():
    from src.feature.step_parser import CATEGORY_FEATURE_MAP
    assert 'bolt' in CATEGORY_FEATURE_MAP
    assert 'gear' in CATEGORY_FEATURE_MAP
    assert 'shaft' in CATEGORY_FEATURE_MAP
    assert 'primary_features' in CATEGORY_FEATURE_MAP['bolt']


def test_extractor_normalize_feature_type():
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    assert extractor._normalize_feature_type('通孔') == 'through_hole'
    assert extractor._normalize_feature_type('through_hole') == 'through_hole'
    assert extractor._normalize_feature_type('四边形腔') == 'rectangular_pocket'


def test_extractor_validate_feature():
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    feature = MachiningFeature(
        feature_type='through_hole',
        diameter=10.0,
        depth=25.0,
        precision='IT7',
        roughness=1.6,
    )
    result = extractor.validate_feature(feature)
    assert 'valid' in result
    assert 'confidence' in result
    assert 'issues' in result


def test_extractor_validate_invalid_feature():
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    feature = MachiningFeature(
        feature_type='unknown_type',
        diameter=0.001,
        precision='INVALID',
    )
    result = extractor.validate_feature(feature)
    assert result['valid'] is False
    assert len(result['issues']) > 0


def test_extractor_compute_confidence():
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    feature = MachiningFeature(
        feature_type='through_hole',
        diameter=10.0,
        depth=25.0,
        precision='IT7',
        roughness=1.6,
    )
    confidence = extractor._compute_feature_confidence(feature)
    assert 0 <= confidence <= 1
    assert confidence > 0.5


def test_recommender_flatten_param():
    from src.recommend.recommender import _flatten_param, _clamp_param, _determine_process_type
    assert _flatten_param(100) == 100.0
    assert _flatten_param('200rpm') == 200.0
    assert _flatten_param(None) is None
    assert _flatten_param({'a': 100, 'b': 200}) == 150.0


def test_recommender_clamp_param():
    from src.recommend.recommender import _clamp_param
    assert _clamp_param(50, 0, 100) == 50
    assert _clamp_param(-10, 0, 100) == 0
    assert _clamp_param(200, 0, 100) == 100
    assert _clamp_param(None, 0, 100) == 50


def test_recommender_determine_process_type():
    from src.recommend.recommender import _determine_process_type
    assert _determine_process_type('粗铣-精铣') == 'milling'
    assert _determine_process_type('钻-扩-铰') == 'drilling'
    assert _determine_process_type('粗车-精车') == 'turning'


def test_pipeline_state():
    from src.pipeline.run import PipelineState, StepState
    state = PipelineState()
    state.start_step('test_step')
    assert state.steps['test_step'].status == 'running'
    state.finish_step('test_step', 'ok')
    assert state.steps['test_step'].status == 'success'
    summary = state.get_summary()
    assert summary['total_steps'] == 1
    assert summary['success'] == 1


def test_pipeline_state_failure():
    from src.pipeline.run import PipelineState
    state = PipelineState()
    state.start_step('fail_step')
    state.fail_step('fail_step', 'error occurred')
    assert state.steps['fail_step'].status == 'failed'
    assert state.steps['fail_step'].error_info == 'error occurred'


def test_pipeline_state_skip():
    from src.pipeline.run import PipelineState
    state = PipelineState()
    state.skip_step('skip_step', 'not needed')
    assert state.steps['skip_step'].status == 'skipped'
