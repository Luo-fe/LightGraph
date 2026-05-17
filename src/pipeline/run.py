import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import (
    DATA_DIR,
    PART_CATEGORY_MAP,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TMCAD_DATASET_PATH,
)
from src.data.cleaner import DataCleaner
from src.data.collector import DataCollector
from src.data.structurer import DataStructurer
from src.data.vector_store import VectorStore
from src.feature.extractor import FeatureExtractor
from src.feature.step_parser import STEPParser
from src.graph.glm_client import GLMClient
from src.graph.glm_embedder import GLMEmbedder
from src.graph.knowledge_graph import CADKnowledgeGraph
from src.recommend.recommender import ProcessRecommender
from src.validation import TYPICAL_PART_TEST_CASES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
for noisy in [
    'neo4j.notifications',
    'httpx',
    'httpcore',
    'graphiti_core',
    'openai._base_client',
    'sentence_transformers',
    'urllib3',
]:
    logging.getLogger(noisy).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


@dataclass
class StepState:
    name: str
    status: str = 'pending'
    start_time: datetime | None = None
    end_time: datetime | None = None
    result_summary: str = ''
    error_info: str = ''

    @property
    def duration_seconds(self) -> float | None:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'result_summary': self.result_summary,
            'error_info': self.error_info,
        }


class PipelineState:
    def __init__(self):
        self.steps: dict[str, StepState] = {}
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    def start_step(self, name: str):
        if name not in self.steps:
            self.steps[name] = StepState(name=name)
        step = self.steps[name]
        step.status = 'running'
        step.start_time = datetime.now(timezone.utc)
        logger.info(f'[PipelineState] 步骤 "{name}" 开始执行')

    def finish_step(self, name: str, result_summary: str = ''):
        if name not in self.steps:
            return
        step = self.steps[name]
        step.status = 'success'
        step.end_time = datetime.now(timezone.utc)
        step.result_summary = result_summary
        logger.info(
            f'[PipelineState] 步骤 "{name}" 完成 '
            f'(耗时{step.duration_seconds:.1f}s): {result_summary}'
        )

    def fail_step(self, name: str, error_info: str):
        if name not in self.steps:
            self.steps[name] = StepState(name=name)
        step = self.steps[name]
        step.status = 'failed'
        step.end_time = datetime.now(timezone.utc)
        step.error_info = error_info
        logger.error(f'[PipelineState] 步骤 "{name}" 失败: {error_info}')

    def skip_step(self, name: str, reason: str = ''):
        if name not in self.steps:
            self.steps[name] = StepState(name=name)
        step = self.steps[name]
        step.status = 'skipped'
        step.end_time = datetime.now(timezone.utc)
        step.result_summary = reason
        logger.info(f'[PipelineState] 步骤 "{name}" 跳过: {reason}')

    def get_summary(self) -> dict:
        total = len(self.steps)
        success = sum(1 for s in self.steps.values() if s.status == 'success')
        failed = sum(1 for s in self.steps.values() if s.status == 'failed')
        skipped = sum(1 for s in self.steps.values() if s.status == 'skipped')
        pending = sum(1 for s in self.steps.values() if s.status in ('pending', 'running'))
        return {
            'total_steps': total,
            'success': success,
            'failed': failed,
            'skipped': skipped,
            'pending_or_running': pending,
            'pipeline_start': self.start_time.isoformat() if self.start_time else None,
            'pipeline_end': self.end_time.isoformat() if self.end_time else None,
        }

    def to_dict(self) -> dict:
        return {
            'summary': self.get_summary(),
            'steps': {name: step.to_dict() for name, step in self.steps.items()},
        }

    def format_report(self) -> str:
        lines = []
        lines.append('=' * 70)
        lines.append('流程执行报告')
        lines.append('=' * 70)
        summary = self.get_summary()
        lines.append(
            f'总步骤: {summary["total_steps"]}, '
            f'成功: {summary["success"]}, '
            f'失败: {summary["failed"]}, '
            f'跳过: {summary["skipped"]}, '
            f'未完成: {summary["pending_or_running"]}'
        )
        lines.append('-' * 70)
        for name, step in self.steps.items():
            duration = f'{step.duration_seconds:.1f}s' if step.duration_seconds is not None else 'N/A'
            status_icon = {
                'success': '✓',
                'failed': '✗',
                'skipped': '○',
                'running': '⟳',
                'pending': '·',
            }.get(step.status, '?')
            lines.append(
                f'  [{status_icon}] {name}: {step.status} ({duration})'
            )
            if step.result_summary:
                lines.append(f'      结果: {step.result_summary}')
            if step.error_info:
                lines.append(f'      错误: {step.error_info}')
        lines.append('=' * 70)
        return '\n'.join(lines)


async def run_data_pipeline(raw_data_path: Path | None = None) -> dict:
    logger.info('=' * 60)
    logger.info('工作1：历史数据收集、清洗、结构化、标注和入库')
    logger.info('=' * 60)

    collector = DataCollector()
    cleaner = DataCleaner()
    structurer = DataStructurer()

    if raw_data_path and raw_data_path.exists():
        raw_data = collector.collect_process_documents(raw_data_path)
    else:
        logger.info('未指定数据路径，使用示例数据')
        raw_data = _generate_sample_data()

    collector.save_raw_data(raw_data)

    cleaned_data = cleaner.clean_and_score_dataset(raw_data)
    cleaner.save_cleaned_data(cleaned_data)

    cases = []
    for item in cleaned_data:
        case = structurer.structure_and_annotate(item)
        if case:
            cases.append(case)
    structurer.save_structured_data(cases)

    training_data = structurer.generate_training_data(cases)
    structurer.save_training_data(training_data)

    all_triples = []
    for case in cases:
        triples = structurer.generate_knowledge_triples(case)
        all_triples.extend(triples)
    if all_triples:
        structurer.save_knowledge_triples(all_triples)
        logger.info(f'知识三元组生成完成: {len(all_triples)} 条')

    embedder = GLMEmbedder()
    vector_store = VectorStore(embedder=embedder)
    docs = [
        {'content': json.dumps(c.model_dump(), ensure_ascii=False), 'id': c.id}
        for c in cases
    ]
    await vector_store.add_documents(docs, text_key='content')
    vector_store.save()
    logger.info(f'向量库写入完成: {len(docs)} 条文档')
    await embedder.close()

    kg = CADKnowledgeGraph()
    kg_connected = await kg.initialize()
    kg_count = 0
    if kg_connected:
        case_dicts = [c.model_dump() for c in cases]
        kg_count = await kg.add_process_cases_bulk(case_dicts) or 0
        await kg.close()

    logger.info(
        f'数据处理完成: 原始{len(raw_data)}条 -> 清洗评分{len(cleaned_data)}条 '
        f'-> 结构化标注{len(cases)}条 -> 训练数据{len(training_data)}条 '
        f'-> 三元组{len(all_triples)}条 -> 向量库{len(docs)}条 -> 知识图谱{kg_count}条'
    )
    return {
        'raw_count': len(raw_data),
        'cleaned_count': len(cleaned_data),
        'structured_count': len(cases),
        'training_count': len(training_data),
        'triple_count': len(all_triples),
        'vector_count': len(docs),
        'kg_count': kg_count,
    }


async def run_tmcad_pipeline(max_per_category: int = 3) -> dict:
    logger.info('=' * 60)
    logger.info('工作1+2：TMCAD数据集处理流程')
    logger.info('=' * 60)

    parser = STEPParser()
    logger.info(f'扫描TMCAD数据集: {TMCAD_DATASET_PATH}')
    parts = parser.scan_dataset(max_per_category=max_per_category)
    logger.info(f'共扫描 {len(parts)} 个零件')

    parts_path = RAW_DATA_DIR / 'tmcad_parts.json'
    parts_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parts_path, 'w', encoding='utf-8') as f:
        json.dump(parts, f, ensure_ascii=False, indent=2)
    logger.info(f'零件信息已保存至 {parts_path}')

    glm_client = GLMClient()
    all_features = []
    for part in parts:
        summary = part.get('summary', '')
        if not summary:
            continue
        logger.info(f'正在识别零件 {part["filename"]} 的加工特征...')
        try:
            features = await FeatureExtractor(glm_client).extract_from_text(summary)
            for feat in features:
                feat_dict = feat.model_dump()
                feat_dict['source_file'] = part['filename']
                feat_dict['source_category'] = part.get('category_cn', '')
                feat_dict['estimated_diameter'] = part.get('estimated_diameter')
                feat_dict['estimated_length'] = part.get('estimated_length')
                all_features.append(feat_dict)
        except Exception as e:
            logger.warning(f'特征识别失败 {part["filename"]}: {e}')

    features_path = PROCESSED_DATA_DIR / 'tmcad_features.json'
    features_path.parent.mkdir(parents=True, exist_ok=True)
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2)
    logger.info(f'特征识别完成: {len(all_features)} 个特征，保存至 {features_path}')

    await glm_client.close()

    return {
        'parts_scanned': len(parts),
        'features_extracted': len(all_features),
    }


async def run_feature_recognition(feature_input: dict | str) -> list:
    logger.info('=' * 60)
    logger.info('工作2：加工特征识别')
    logger.info('=' * 60)

    extractor = FeatureExtractor()

    if isinstance(feature_input, dict):
        features = await extractor.extract_from_json(feature_input)
    else:
        features = await extractor.extract_from_text(feature_input)

    for f in features:
        logger.info(f'识别到特征: {f.feature_type} (深度={f.depth}, 精度={f.precision})')

    await extractor.glm_client.close()
    return features


async def run_process_recommendation(features: list) -> list[dict]:
    logger.info('=' * 60)
    logger.info('工作2：工艺参数推荐')
    logger.info('=' * 60)

    kg = CADKnowledgeGraph()
    connected = await kg.initialize()
    if not connected:
        logger.warning('知识图谱未连接，将仅使用LLM和向量库')
        kg = None

    embedder = GLMEmbedder()
    vector_store = VectorStore(embedder=embedder)
    try:
        vector_store.load()
    except Exception:
        logger.warning('向量库加载失败，将仅使用LLM')

    recommender = ProcessRecommender(
        knowledge_graph=kg,
        vector_store=vector_store,
    )

    results = []
    for feature in features:
        recommendation = await recommender.recommend_with_validation(feature)
        json_output = recommender.format_json_output(recommendation)
        results.append(json_output)
        logger.info(f'推荐结果: {json.dumps(json_output, ensure_ascii=False, indent=2)}')

    if kg:
        await kg.close()
    await embedder.close()

    return results


async def run_end_to_end(feature_input: dict | str) -> list[dict]:
    logger.info('=' * 60)
    logger.info('工作4：端到端流程')
    logger.info('=' * 60)

    extractor = FeatureExtractor()

    if isinstance(feature_input, dict):
        text_input = json.dumps(feature_input, ensure_ascii=False)
    else:
        text_input = feature_input

    extract_result = await extractor.extract_and_validate(text_input)
    features = extract_result['features']

    if not features:
        logger.error('未识别到任何加工特征')
        await extractor.glm_client.close()
        return []

    intermediate_path = DATA_DIR / 'output'
    intermediate_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')

    feature_result_path = intermediate_path / f'feature_result_{timestamp}.json'
    feature_save_data = {
        'features': [f.model_dump() for f in features],
        'validations': extract_result['validations'],
        'refined': extract_result['refined'],
    }
    with open(feature_result_path, 'w', encoding='utf-8') as f:
        json.dump(feature_save_data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f'特征识别中间结果已保存至 {feature_result_path}')

    kg = CADKnowledgeGraph()
    connected = await kg.initialize()
    if not connected:
        logger.warning('知识图谱未连接，将仅使用LLM和向量库')
        kg = None

    embedder = GLMEmbedder()
    vector_store = VectorStore(embedder=embedder)
    with contextlib.suppress(Exception):
        vector_store.load()

    recommender = ProcessRecommender(
        knowledge_graph=kg,
        vector_store=vector_store,
    )

    results = []
    for feature in features:
        recommendation = await recommender.recommend_with_full_validation(feature)
        detailed_output = recommender.format_detailed_output(recommendation)
        results.append(detailed_output)

        provenance = recommendation.get('provenance', {})
        retrieval_path = intermediate_path / f'retrieval_result_{timestamp}_{feature.feature_type}.json'
        retrieval_save_data = {
            'feature_type': feature.feature_type,
            'provenance': provenance,
        }
        with open(retrieval_path, 'w', encoding='utf-8') as f:
            json.dump(retrieval_save_data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f'检索结果中间结果已保存至 {retrieval_path}')

        logger.info(f'推荐结果: {json.dumps(detailed_output, ensure_ascii=False, indent=2, default=str)[:400]}...')

    if kg:
        await kg.close()
    await embedder.close()
    await extractor.glm_client.close()

    filepath = intermediate_path / f'result_{timestamp}.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f'结果已保存至 {filepath}')

    return results


async def run_tmcad_end_to_end(max_per_category: int = 2) -> list[dict]:
    logger.info('=' * 60)
    logger.info('工作4：TMCAD数据集端到端流程')
    logger.info('=' * 60)

    parser = STEPParser()
    parts = parser.scan_dataset(max_per_category=max_per_category)
    logger.info(f'选取 {len(parts)} 个零件进行端到端测试')

    glm_client = GLMClient()
    extractor = FeatureExtractor(glm_client)

    embedder = GLMEmbedder()
    vector_store = VectorStore(embedder=embedder)
    with contextlib.suppress(Exception):
        vector_store.load()

    kg = CADKnowledgeGraph()
    connected = await kg.initialize()
    if not connected:
        logger.warning('知识图谱未连接')
        kg = None

    recommender = ProcessRecommender(knowledge_graph=kg, vector_store=vector_store)

    all_results = []
    for part in parts:
        summary = part.get('summary', '')
        if not summary:
            continue

        logger.info(f'处理零件: {part["filename"]} ({part.get("category_cn", "")})')

        try:
            features = await extractor.extract_from_text(summary)
            if not features:
                logger.warning(f'未识别到特征: {part["filename"]}')
                continue

            for feature in features:
                recommendation = await recommender.recommend_with_validation(feature)
                json_output = recommender.format_json_output(recommendation)
                json_output['source_file'] = part['filename']
                json_output['source_category'] = part.get('category_cn', '')
                all_results.append(json_output)
                logger.info(
                    f'  推荐: {json_output.get("加工方法", "")} - '
                    f'{json_output.get("加工工艺路线", "")[:50]}...'
                )
        except Exception as e:
            logger.error(f'处理失败 {part["filename"]}: {e}')

    output_path = DATA_DIR / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    filepath = output_path / f'tmcad_result_{timestamp}.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f'TMCAD端到端结果已保存至 {filepath}，共 {len(all_results)} 条')

    await glm_client.close()
    await embedder.close()
    if kg:
        await kg.close()

    return all_results


async def run_tmcad_knowledge_build(max_per_category: int = 5) -> dict:
    logger.info('=' * 60)
    logger.info('TMCAD数据集知识库构建（向量库+知识图谱）')
    logger.info('=' * 60)

    parser = STEPParser()
    parts = parser.scan_dataset(max_per_category=max_per_category)
    logger.info(f'扫描 {len(parts)} 个零件用于知识库构建')

    glm_client = GLMClient()
    extractor = FeatureExtractor(glm_client)

    tmcad_cases = []
    for part in parts:
        summary = part.get('summary', '')
        if not summary:
            continue
        try:
            features = await extractor.extract_from_text(summary)
            for feat in features:
                feat_dict = feat.model_dump()
                category = part.get('category', '')
                cat_info = _get_category_info(category)
                diameter = feat_dict.get('diameter') or part.get('estimated_diameter')
                length = feat_dict.get('length') or part.get('estimated_length')

                case = {
                    'id': f"tmcad_{category}_{part['filename'].replace('.stp', '').replace('.step', '')}_{feat_dict['feature_type']}",
                    'feature_type': feat_dict['feature_type'],
                    'length': length,
                    'width': feat_dict.get('width'),
                    'diameter': diameter,
                    'depth': feat_dict.get('depth'),
                    'precision': feat_dict.get('precision', 'IT7'),
                    'roughness': feat_dict.get('roughness', 3.2),
                    'machining_method': _infer_machining_method(feat_dict['feature_type'], category),
                    'process_route': _infer_process_route(feat_dict['feature_type'], category, diameter, length),
                    'spindle_speed': _infer_spindle_speed(feat_dict['feature_type'], category),
                    'feed_rate': _infer_feed_rate(feat_dict['feature_type'], category),
                    'tool_diameter': _infer_tool_diameter(feat_dict['feature_type'], diameter),
                    'cutting_depth': _infer_cutting_depth(feat_dict['feature_type']),
                    'cutting_width': _infer_cutting_width(feat_dict['feature_type']),
                    'material': cat_info.get('material', '45号钢'),
                    'machine_tool': cat_info.get('machine_tool', '数控车床'),
                }
                tmcad_cases.append(case)
        except Exception as e:
            logger.warning(f'特征识别失败 {part["filename"]}: {e}')

    await glm_client.close()
    logger.info(f'生成 {len(tmcad_cases)} 条TMCAD工艺案例')

    embedder = GLMEmbedder()
    vector_store = VectorStore(embedder=embedder)
    docs = [
        {'content': json.dumps(c, ensure_ascii=False), 'id': c['id']}
        for c in tmcad_cases
    ]
    await vector_store.add_documents(docs, text_key='content')
    vector_store.save()
    logger.info(f'TMCAD向量库写入完成: {len(docs)} 条')
    await embedder.close()

    kg = CADKnowledgeGraph()
    kg_connected = await kg.initialize()
    kg_count = 0
    if kg_connected:
        kg_count = await kg.add_process_cases_bulk(tmcad_cases) or 0
        await kg.close()

    cases_path = PROCESSED_DATA_DIR / 'tmcad_cases.json'
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cases_path, 'w', encoding='utf-8') as f:
        json.dump(tmcad_cases, f, ensure_ascii=False, indent=2)

    return {
        'parts_scanned': len(parts),
        'cases_generated': len(tmcad_cases),
        'vector_count': len(docs),
        'kg_count': kg_count,
    }


def _get_category_info(category: str) -> dict:
    from src.feature.step_parser import CATEGORY_FEATURE_MAP

    return CATEGORY_FEATURE_MAP.get(category, {
        'material': '45号钢',
        'machine_tool': '数控车床',
        'process_type': 'turning',
    })


def _infer_machining_method(feature_type: str, category: str) -> str:
    from src.config.settings import MACHINING_METHODS

    methods = MACHINING_METHODS.get(feature_type, [])
    if methods:
        return methods[0]
    if category in ('bolt', 'nut', 'shaft', 'flange'):
        return '粗车-精车'
    if category == 'gear':
        return '粗车-精车-滚齿'
    return '粗铣-精铣'


def _infer_process_route(feature_type: str, category: str, diameter=None, length=None) -> str:
    d_str = f'直径{diameter}mm' if diameter else '相应直径'
    l_str = f'长度{length}mm' if length else ''

    routes = {
        'outer_circle': f'选择棒料→车床装夹→粗车外圆至{d_str}留0.5mm余量→精车外圆至{d_str}',
        'conical_surface': f'选择棒料→车床装夹→粗车圆锥面留0.5mm余量→精车圆锥面至IT7精度',
        'cylindrical_surface': f'选择棒料→车床装夹→粗车圆柱面至{d_str}留0.5mm余量→精车至{d_str}',
        'circular_curve': f'选择棒料→车床装夹→粗车圆弧面→精车圆弧面至IT7精度',
        'thread': f'选择棒料→车床装夹→车螺纹底径→精车螺纹至标准',
        'gear_tooth': f'选择棒料→车床装夹→粗车外圆→精车外圆→滚齿加工齿形→剃齿精加工',
        'through_hole': f'选择工件→CNC定位→钻头钻孔→扩孔钻扩孔→铰刀铰孔至{d_str}',
        'blind_hole': f'选择工件→CNC定位→钻头钻孔→镗刀镗孔至{d_str}',
        'rectangular_pocket': f'选择工件→CNC定位→铣刀粗铣腔体→精铣至尺寸',
        'square_slot': f'选择工件→CNC定位→铣刀粗铣槽→精铣至尺寸',
    }
    return routes.get(feature_type, f'选择工件→装夹→加工{feature_type}特征至要求尺寸')


def _infer_spindle_speed(feature_type: str, category: str) -> float:
    if feature_type in ('thread',):
        return 800
    if feature_type in ('gear_tooth',):
        return 600
    if category in ('bolt', 'nut', 'shaft', 'flange'):
        return 1500
    if feature_type in ('through_hole', 'blind_hole'):
        return 2000
    return 3000


def _infer_feed_rate(feature_type: str, category: str) -> float:
    if feature_type in ('thread',):
        return 40
    if feature_type in ('gear_tooth',):
        return 60
    if category in ('bolt', 'nut', 'shaft', 'flange'):
        return 150
    if feature_type in ('through_hole', 'blind_hole'):
        return 120
    return 600


def _infer_tool_diameter(feature_type: str, diameter=None) -> float | None:
    if feature_type in ('through_hole', 'blind_hole'):
        return diameter if diameter else 10
    if feature_type in ('rectangular_pocket', 'square_slot'):
        return 8
    return None


def _infer_cutting_depth(feature_type: str) -> float:
    if feature_type in ('thread',):
        return 0.5
    if feature_type in ('gear_tooth',):
        return 1.5
    return 2.0


def _infer_cutting_width(feature_type: str) -> float | None:
    if feature_type in ('rectangular_pocket', 'square_slot'):
        return 4.0
    return None


async def run_validation(test_cases: list[dict]) -> dict:
    logger.info('=' * 60)
    logger.info('工作3：典型零件原型验证')
    logger.info('=' * 60)

    extractor = FeatureExtractor()
    kg = CADKnowledgeGraph()
    connected = await kg.initialize()
    if not connected:
        kg = None

    embedder = GLMEmbedder()
    vector_store = VectorStore(embedder=embedder)
    with contextlib.suppress(Exception):
        vector_store.load()

    recommender = ProcessRecommender(knowledge_graph=kg, vector_store=vector_store)

    results = []
    correct_method = 0
    correct_tool = 0
    total = len(test_cases)

    for i, test_case in enumerate(test_cases):
        expected_method = test_case.get('expected_method', '')
        expected_tool_diameter = test_case.get('expected_tool_diameter')

        feature = extractor.extract_from_structured_data(test_case)
        if feature is None:
            logger.warning(f'测试用例 {i} 特征提取失败')
            continue

        recommendation = await recommender.recommend_with_validation(feature)
        json_output = recommender.format_json_output(recommendation)

        recommended_method = json_output.get('加工方法', '')
        if expected_method and expected_method in recommended_method:
            correct_method += 1

        if expected_tool_diameter:
            params = json_output.get('加工参数', {})
            tool_d = params.get('刀具直径(mm)')
            if tool_d and abs(float(tool_d) - float(expected_tool_diameter)) / float(
                expected_tool_diameter
            ) < 0.2:
                correct_tool += 1

        results.append({
            'test_case': i,
            'feature_type': test_case.get('feature_type', ''),
            'expected_method': expected_method,
            'recommended_method': recommended_method,
            'method_match': expected_method in recommended_method if expected_method else None,
            'output': json_output,
        })
        logger.info(
            f'  用例{i}: 特征={test_case.get("feature_type", "")}, '
            f'期望={expected_method}, 推荐={recommended_method}, '
            f'匹配={expected_method in recommended_method if expected_method else "N/A"}'
        )

    mdpm = correct_method / total if total > 0 else 0
    mdmt = (
        correct_tool / total
        if total > 0 and any(tc.get('expected_tool_diameter') for tc in test_cases)
        else None
    )

    validation_report = {
        'total_cases': total,
        'mdpm': mdpm,
        'mdmt': mdmt,
        'results': results,
    }

    report_path = DATA_DIR / 'validation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, ensure_ascii=False, indent=2)
    logger.info(f'验证报告已保存至 {report_path}')
    logger.info(f'MDPM(加工方法匹配度)={mdpm:.2%}, MDMT(加工刀具匹配度)={mdmt}')

    if kg:
        await kg.close()
    await extractor.glm_client.close()
    await embedder.close()

    return validation_report


async def run_typical_part_validation() -> dict:
    logger.info('=' * 60)
    logger.info('工作3+：典型零件完整验证（轴类+齿轮）')
    logger.info('=' * 60)

    extractor = FeatureExtractor()
    kg = CADKnowledgeGraph()
    connected = await kg.initialize()
    if not connected:
        logger.warning('知识图谱未连接，将仅使用LLM和向量库')
        kg = None

    embedder = GLMEmbedder()
    vector_store = VectorStore(embedder=embedder)
    with contextlib.suppress(Exception):
        vector_store.load()

    recommender = ProcessRecommender(
        knowledge_graph=kg,
        vector_store=vector_store,
    )

    all_results = {}
    overall_correct_method = 0
    overall_correct_tool = 0
    overall_total = 0

    for part_key, test_cases in TYPICAL_PART_TEST_CASES.items():
        part_name = PART_CATEGORY_MAP.get(part_key, part_key)
        logger.info(f'\n--- 验证 {part_name} ({len(test_cases)}个特征) ---')

        part_results = []
        correct_method = 0
        correct_tool = 0
        total = len(test_cases)

        for i, test_case in enumerate(test_cases):
            expected_method = test_case.get('expected_method', '')
            expected_tool_diameter = test_case.get('expected_tool_diameter')

            feature = extractor.extract_from_structured_data(test_case)
            if feature is None:
                logger.warning(f'  {part_name} 测试用例 {i} 特征提取失败')
                part_results.append({
                    'test_case': i,
                    'feature_type': test_case.get('feature_type', ''),
                    'status': 'extraction_failed',
                })
                continue

            try:
                recommendation = await recommender.recommend_with_full_validation(feature)
                detailed_output = recommender.format_detailed_output(recommendation)

                recommended_method = detailed_output.get('加工方法', '')
                method_match = expected_method in recommended_method if expected_method else None
                if method_match:
                    correct_method += 1

                tool_match = None
                if expected_tool_diameter:
                    params = detailed_output.get('加工参数', {})
                    tool_d = params.get('刀具直径(mm)')
                    if tool_d:
                        tool_match = (
                            abs(float(tool_d) - float(expected_tool_diameter))
                            / float(expected_tool_diameter)
                            < 0.2
                        )
                        if tool_match:
                            correct_tool += 1

                part_results.append({
                    'test_case': i,
                    'feature_type': test_case.get('feature_type', ''),
                    'expected_method': expected_method,
                    'recommended_method': recommended_method,
                    'method_match': method_match,
                    'tool_match': tool_match,
                    'confidence': detailed_output.get('置信度', {}),
                    'output': detailed_output,
                })
                logger.info(
                    f'  {part_name} 用例{i}: 特征={test_case.get("feature_type", "")}, '
                    f'期望={expected_method}, 推荐={recommended_method}, '
                    f'方法匹配={method_match}, 刀具匹配={tool_match}'
                )
            except Exception as e:
                logger.error(f'  {part_name} 用例{i} 推荐失败: {e}')
                part_results.append({
                    'test_case': i,
                    'feature_type': test_case.get('feature_type', ''),
                    'status': 'recommendation_failed',
                    'error': str(e),
                })

        part_mdpm = correct_method / total if total > 0 else 0
        part_mdmt = (
            correct_tool / total
            if total > 0 and any(tc.get('expected_tool_diameter') for tc in test_cases)
            else None
        )

        all_results[part_key] = {
            'name': part_name,
            'total_cases': total,
            'mdpm': part_mdpm,
            'mdmt': part_mdmt,
            'results': part_results,
        }

        overall_correct_method += correct_method
        overall_correct_tool += correct_tool
        overall_total += total

        logger.info(
            f'  {part_name} 验证结果: MDPM={part_mdpm:.2%}, '
            f'MDMT={part_mdmt}'
        )

    overall_mdpm = overall_correct_method / overall_total if overall_total > 0 else 0
    overall_mdmt = (
        overall_correct_tool / overall_total
        if overall_total > 0
        else None
    )

    report = {
        'overall_mdpm': overall_mdpm,
        'overall_mdmt': overall_mdmt,
        'overall_total': overall_total,
        'parts': all_results,
    }

    report_path = DATA_DIR / 'typical_part_validation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f'典型零件验证报告已保存至 {report_path}')
    logger.info(
        f'总体验证结果: MDPM={overall_mdpm:.2%}, MDMT={overall_mdmt}, '
        f'总用例数={overall_total}'
    )

    if kg:
        await kg.close()
    await extractor.glm_client.close()
    await embedder.close()

    return report


def _generate_sample_data() -> list[dict]:
    return [
        {
            'id': 'sample_001',
            'feature_type': '四边形腔',
            'length': 120,
            'width': 100,
            'diameter': None,
            'depth': 50,
            'precision': 'IT8',
            'roughness': 6.3,
            'machining_method': '粗铣-半精铣',
            'process_route': '选择130mm长110mm宽60mm深铝合金工件→CNC定位夹紧→直径12mm铣刀粗铣→直径4mm铣刀半精铣',
            'spindle_speed': 3000,
            'feed_rate': 800,
            'tool_diameter': 12,
            'cutting_depth': 3,
            'cutting_width': 6,
            'material': '铝合金7075',
            'machine_tool': 'CNC加工中心',
        },
        {
            'id': 'sample_002',
            'feature_type': '通孔',
            'length': None,
            'width': None,
            'diameter': 10,
            'depth': 25,
            'precision': 'IT7',
            'roughness': 1.6,
            'machining_method': '钻-扩-铰',
            'process_route': '选择铝合金工件→CNC定位→直径9.8mm钻头钻孔→直径9.95mm扩孔钻扩孔→直径10mm铰刀铰孔',
            'spindle_speed': 2000,
            'feed_rate': 150,
            'tool_diameter': 10,
            'cutting_depth': 25,
            'cutting_width': None,
            'material': '铝合金6061',
            'machine_tool': 'CNC加工中心',
        },
        {
            'id': 'sample_003',
            'feature_type': '外圆',
            'length': 80,
            'width': None,
            'diameter': 50,
            'depth': None,
            'precision': 'IT6',
            'roughness': 0.8,
            'machining_method': '粗车-精车',
            'process_route': '选择45号钢棒料→车床装夹→粗车外圆至直径50.5mm→精车外圆至直径50mm',
            'spindle_speed': 1500,
            'feed_rate': 200,
            'tool_diameter': None,
            'cutting_depth': 2,
            'cutting_width': None,
            'material': '45号钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_004',
            'feature_type': '方形槽',
            'length': 60,
            'width': 30,
            'diameter': None,
            'depth': 15,
            'precision': 'IT7',
            'roughness': 3.2,
            'machining_method': '粗铣-精铣',
            'process_route': '选择铝合金工件→CNC定位→直径8mm铣刀粗铣→直径4mm铣刀精铣',
            'spindle_speed': 4000,
            'feed_rate': 600,
            'tool_diameter': 8,
            'cutting_depth': 2,
            'cutting_width': 4,
            'material': '铝合金2024',
            'machine_tool': 'CNC加工中心',
        },
        {
            'id': 'sample_005',
            'feature_type': '盲孔',
            'length': None,
            'width': None,
            'diameter': 8,
            'depth': 20,
            'precision': 'IT8',
            'roughness': 3.2,
            'machining_method': '钻-镗',
            'process_route': '选择钢件→CNC定位→直径7.8mm钻头钻孔→直径8mm镗刀镗孔至深度20mm',
            'spindle_speed': 2500,
            'feed_rate': 120,
            'tool_diameter': 8,
            'cutting_depth': 20,
            'cutting_width': None,
            'material': '40Cr钢',
            'machine_tool': 'CNC加工中心',
        },
        {
            'id': 'sample_006',
            'feature_type': '圆锥面',
            'length': 30,
            'width': None,
            'diameter': 40,
            'depth': None,
            'precision': 'IT7',
            'roughness': 1.6,
            'machining_method': '粗车-精车',
            'process_route': '选择45号钢棒料→数控车床装夹→粗车圆锥面留0.5mm余量→精车圆锥面至IT7精度',
            'spindle_speed': 1200,
            'feed_rate': 80,
            'tool_diameter': None,
            'cutting_depth': 1.5,
            'cutting_width': None,
            'material': '45号钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_007',
            'feature_type': '圆柱面',
            'length': 100,
            'width': None,
            'diameter': 30,
            'depth': None,
            'precision': 'IT7',
            'roughness': 1.6,
            'machining_method': '粗车-精车',
            'process_route': '选择40Cr钢棒料→数控车床装夹→粗车圆柱面至直径30.5mm→精车圆柱面至直径30mm',
            'spindle_speed': 1800,
            'feed_rate': 100,
            'tool_diameter': None,
            'cutting_depth': 2,
            'cutting_width': None,
            'material': '40Cr钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_008',
            'feature_type': '圆曲线',
            'length': 50,
            'width': None,
            'diameter': 25,
            'depth': None,
            'precision': 'IT7',
            'roughness': 1.6,
            'machining_method': '粗车-精车',
            'process_route': '选择铝合金棒料→数控车床装夹→粗车圆弧面→精车圆弧面至IT7精度',
            'spindle_speed': 2000,
            'feed_rate': 60,
            'tool_diameter': None,
            'cutting_depth': 1,
            'cutting_width': None,
            'material': '铝合金6061',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_009',
            'feature_type': '圆锥面',
            'length': 50,
            'width': None,
            'diameter': 60,
            'depth': None,
            'precision': 'IT6',
            'roughness': 0.8,
            'machining_method': '粗车-半精车-精车',
            'process_route': '选择合金钢棒料→数控车床装夹→粗车圆锥面→半精车留0.2mm余量→精车至IT6精度',
            'spindle_speed': 1000,
            'feed_rate': 50,
            'tool_diameter': None,
            'cutting_depth': 1,
            'cutting_width': None,
            'material': '合金钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_010',
            'feature_type': '螺纹',
            'length': 20,
            'width': None,
            'diameter': 10,
            'depth': None,
            'precision': 'IT7',
            'roughness': 3.2,
            'machining_method': '车螺纹',
            'process_route': '选择钢棒料→数控车床装夹→车螺纹底径→精车螺纹至M10标准',
            'spindle_speed': 800,
            'feed_rate': 40,
            'tool_diameter': None,
            'cutting_depth': 0.5,
            'cutting_width': None,
            'material': '45号钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_bolt_001',
            'feature_type': '螺纹',
            'length': 30,
            'width': None,
            'diameter': 12,
            'depth': None,
            'precision': 'IT7',
            'roughness': 3.2,
            'machining_method': '车螺纹',
            'process_route': '选择45号钢棒料→数控车床装夹→粗车外圆至直径12mm→车螺纹底径→精车M12螺纹',
            'spindle_speed': 800,
            'feed_rate': 40,
            'tool_diameter': None,
            'cutting_depth': 0.5,
            'cutting_width': None,
            'material': '45号钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_bolt_002',
            'feature_type': '外圆',
            'length': 60,
            'width': None,
            'diameter': 12,
            'depth': None,
            'precision': 'IT7',
            'roughness': 1.6,
            'machining_method': '粗车-精车',
            'process_route': '选择45号钢棒料→数控车床装夹→粗车外圆至直径12.5mm→精车至直径12mm',
            'spindle_speed': 1800,
            'feed_rate': 120,
            'tool_diameter': None,
            'cutting_depth': 1.5,
            'cutting_width': None,
            'material': '45号钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_bolt_003',
            'feature_type': '圆柱面',
            'length': 25,
            'width': None,
            'diameter': 20,
            'depth': None,
            'precision': 'IT8',
            'roughness': 3.2,
            'machining_method': '粗车-精车',
            'process_route': '选择45号钢棒料→数控车床装夹→粗车圆柱面至直径20.5mm→精车至直径20mm',
            'spindle_speed': 1500,
            'feed_rate': 100,
            'tool_diameter': None,
            'cutting_depth': 2,
            'cutting_width': None,
            'material': '45号钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_gear_001',
            'feature_type': '齿形',
            'length': 30,
            'width': None,
            'diameter': 80,
            'depth': None,
            'precision': 'IT6',
            'roughness': 0.8,
            'machining_method': '滚齿-剃齿',
            'process_route': '选择40Cr钢锻坯→粗车外圆→精车外圆→滚齿加工齿形→剃齿精加工',
            'spindle_speed': 600,
            'feed_rate': 60,
            'tool_diameter': None,
            'cutting_depth': 1.5,
            'cutting_width': None,
            'material': '40Cr钢',
            'machine_tool': '数控车床+滚齿机',
        },
        {
            'id': 'sample_gear_002',
            'feature_type': '外圆',
            'length': 30,
            'width': None,
            'diameter': 80,
            'depth': None,
            'precision': 'IT7',
            'roughness': 1.6,
            'machining_method': '粗车-精车',
            'process_route': '选择40Cr钢锻坯→数控车床装夹→粗车外圆至直径80.5mm→精车至直径80mm',
            'spindle_speed': 1200,
            'feed_rate': 80,
            'tool_diameter': None,
            'cutting_depth': 2,
            'cutting_width': None,
            'material': '40Cr钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_gear_003',
            'feature_type': '通孔',
            'length': None,
            'width': None,
            'diameter': 25,
            'depth': 30,
            'precision': 'IT7',
            'roughness': 1.6,
            'machining_method': '钻-扩-铰',
            'process_route': '选择40Cr钢工件→CNC定位→直径24.8mm钻头钻孔→直径24.95mm扩孔→直径25mm铰刀铰孔',
            'spindle_speed': 1500,
            'feed_rate': 100,
            'tool_diameter': 25,
            'cutting_depth': 30,
            'cutting_width': None,
            'material': '40Cr钢',
            'machine_tool': 'CNC加工中心',
        },
        {
            'id': 'sample_nut_001',
            'feature_type': '螺纹',
            'length': 15,
            'width': None,
            'diameter': 16,
            'depth': None,
            'precision': 'IT7',
            'roughness': 3.2,
            'machining_method': '车螺纹',
            'process_route': '选择45号钢棒料→数控车床装夹→车外圆→钻孔→车M16内螺纹',
            'spindle_speed': 800,
            'feed_rate': 40,
            'tool_diameter': None,
            'cutting_depth': 0.5,
            'cutting_width': None,
            'material': '45号钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_nut_002',
            'feature_type': '通孔',
            'length': None,
            'width': None,
            'diameter': 14,
            'depth': 15,
            'precision': 'IT8',
            'roughness': 3.2,
            'machining_method': '钻-扩-铰',
            'process_route': '选择45号钢工件→CNC定位→直径13.8mm钻头钻孔→直径13.95mm扩孔→直径14mm铰刀铰孔',
            'spindle_speed': 2000,
            'feed_rate': 120,
            'tool_diameter': 14,
            'cutting_depth': 15,
            'cutting_width': None,
            'material': '45号钢',
            'machine_tool': 'CNC加工中心',
        },
        {
            'id': 'sample_nut_003',
            'feature_type': '外圆',
            'length': 15,
            'width': None,
            'diameter': 24,
            'depth': None,
            'precision': 'IT8',
            'roughness': 3.2,
            'machining_method': '粗车-精车',
            'process_route': '选择45号钢棒料→数控车床装夹→粗车外圆至直径24.5mm→精车至直径24mm',
            'spindle_speed': 1500,
            'feed_rate': 100,
            'tool_diameter': None,
            'cutting_depth': 1.5,
            'cutting_width': None,
            'material': '45号钢',
            'machine_tool': '数控车床',
        },
        {
            'id': 'sample_nut_004',
            'feature_type': '圆柱面',
            'length': 10,
            'width': None,
            'diameter': 52,
            'depth': None,
            'precision': 'IT7',
            'roughness': 1.6,
            'machining_method': '粗车-精车',
            'process_route': '选择45号钢棒料→数控车床装夹→粗车圆柱面至直径52.5mm→精车至直径52mm',
            'spindle_speed': 1200,
            'feed_rate': 80,
            'tool_diameter': None,
            'cutting_depth': 2,
            'cutting_width': None,
            'material': '45号钢',
            'machine_tool': '数控车床',
        },
    ]


def _get_validation_test_cases() -> list[dict]:
    return [
        {
            'feature_type': '通孔',
            'diameter': 10,
            'depth': 25,
            'precision': 'IT7',
            'roughness': 1.6,
            'expected_method': '钻-扩-铰',
            'expected_tool_diameter': 10,
        },
        {
            'feature_type': '四边形腔',
            'length': 100,
            'width': 80,
            'depth': 40,
            'precision': 'IT8',
            'roughness': 6.3,
            'expected_method': '粗铣-半精铣',
            'expected_tool_diameter': 12,
        },
        {
            'feature_type': '外圆',
            'diameter': 50,
            'length': 80,
            'precision': 'IT6',
            'roughness': 0.8,
            'expected_method': '粗车-精车',
        },
        {
            'feature_type': '螺纹',
            'diameter': 12,
            'length': 30,
            'precision': 'IT7',
            'roughness': 3.2,
            'expected_method': '车螺纹',
        },
        {
            'feature_type': '齿形',
            'diameter': 80,
            'length': 30,
            'precision': 'IT6',
            'roughness': 0.8,
            'expected_method': '滚齿',
        },
        {
            'feature_type': '圆锥面',
            'diameter': 40,
            'length': 30,
            'precision': 'IT7',
            'roughness': 1.6,
            'expected_method': '粗车-精车',
        },
        {
            'feature_type': '盲孔',
            'diameter': 8,
            'depth': 20,
            'precision': 'IT8',
            'roughness': 3.2,
            'expected_method': '钻-镗',
        },
        {
            'feature_type': '圆柱面',
            'diameter': 30,
            'length': 100,
            'precision': 'IT7',
            'roughness': 1.6,
            'expected_method': '粗车-精车',
        },
        {
            'feature_type': '方形槽',
            'length': 60,
            'width': 30,
            'depth': 15,
            'precision': 'IT7',
            'roughness': 3.2,
            'expected_method': '粗铣-精铣',
        },
        {
            'feature_type': '圆曲线',
            'diameter': 25,
            'length': 50,
            'precision': 'IT7',
            'roughness': 1.6,
            'expected_method': '粗车-精车',
        },
    ]


def save_pipeline_report(pipeline_state: PipelineState, step_results: dict) -> Path:
    report_path = DATA_DIR / 'pipeline_report.json'
    report = {
        'pipeline_state': pipeline_state.to_dict(),
        'step_results': step_results,
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f'流程报告已保存至 {report_path}')
    return report_path


async def main():
    logger.info('CAD工艺知识图谱系统启动')

    pipeline_state = PipelineState()
    pipeline_state.start_time = datetime.now(timezone.utc)
    step_results = {}

    logger.info('\n步骤0: 清空Neo4j旧数据')
    pipeline_state.start_step('clear_neo4j')
    try:
        kg = CADKnowledgeGraph()
        cleared = await kg.clear_all_data()
        if cleared:
            logger.info('Neo4j旧数据已清空')
        else:
            logger.info('Neo4j清空跳过（可能未连接或graphiti不可用）')
        await kg.close()
        pipeline_state.finish_step('clear_neo4j', f'清空结果: {cleared}')
    except Exception as e:
        logger.warning(f'清空Neo4j失败: {e}')
        pipeline_state.fail_step('clear_neo4j', str(e))

    logger.info('\n步骤1: 测试GLM API连接')
    pipeline_state.start_step('test_glm_api')
    try:
        glm_client = GLMClient()
        response = await glm_client.chat([
            {'role': 'user', 'content': '请用一句话介绍数控加工'}
        ])
        logger.info(f'GLM API连接成功，响应: {response[:80]}...')
        await glm_client.close()
        pipeline_state.finish_step('test_glm_api', 'GLM API连接成功')
    except Exception as e:
        logger.error(f'GLM API连接失败: {e}')
        pipeline_state.fail_step('test_glm_api', str(e))

    logger.info('\n步骤2: 工作1 - 填充知识库（数据收集→清洗评分→结构化标注→三元组→向量库→知识图谱）')
    pipeline_state.start_step('data_pipeline')
    try:
        data_result = await run_data_pipeline()
        step_results['data_pipeline'] = data_result
        logger.info(f'数据处理结果: {data_result}')
        pipeline_state.finish_step(
            'data_pipeline',
            f'原始{data_result["raw_count"]}条->清洗{data_result["cleaned_count"]}条'
            f'->结构化{data_result["structured_count"]}条->三元组{data_result["triple_count"]}条'
        )
    except Exception as e:
        logger.error(f'数据处理流程失败: {e}')
        pipeline_state.fail_step('data_pipeline', str(e))

    logger.info('\n步骤3: 工作1补充 - TMCAD数据集知识库构建')
    pipeline_state.start_step('tmcad_knowledge_build')
    try:
        tmcad_kg_result = await run_tmcad_knowledge_build(max_per_category=3)
        step_results['tmcad_knowledge_build'] = tmcad_kg_result
        logger.info(f'TMCAD知识库构建结果: {tmcad_kg_result}')
        pipeline_state.finish_step(
            'tmcad_knowledge_build',
            f'扫描{tmcad_kg_result["parts_scanned"]}个零件,'
            f'生成{tmcad_kg_result["cases_generated"]}条案例'
        )
    except Exception as e:
        logger.error(f'TMCAD知识库构建失败: {e}')
        pipeline_state.fail_step('tmcad_knowledge_build', str(e))

    logger.info('\n步骤4: 工作3 - 典型零件原型验证')
    pipeline_state.start_step('validation')
    try:
        test_cases = _get_validation_test_cases()
        validation_report = await run_validation(test_cases)
        step_results['validation'] = validation_report
        logger.info(
            f'验证结果: MDPM={validation_report["mdpm"]:.2%}, '
            f'MDMT={validation_report["mdmt"]}'
        )
        pipeline_state.finish_step(
            'validation',
            f'MDPM={validation_report["mdpm"]:.2%}, MDMT={validation_report["mdmt"]}'
        )
    except Exception as e:
        logger.error(f'典型零件原型验证失败: {e}')
        pipeline_state.fail_step('validation', str(e))

    logger.info('\n步骤5: 工作3+ - 典型零件完整验证（轴类+齿轮）')
    pipeline_state.start_step('typical_part_validation')
    try:
        typical_report = await run_typical_part_validation()
        step_results['typical_part_validation'] = typical_report
        pipeline_state.finish_step(
            'typical_part_validation',
            f'总体MDPM={typical_report["overall_mdpm"]:.2%}, '
            f'MDMT={typical_report["overall_mdmt"]}, '
            f'总用例={typical_report["overall_total"]}'
        )
    except Exception as e:
        logger.error(f'典型零件完整验证失败: {e}')
        pipeline_state.fail_step('typical_part_validation', str(e))

    logger.info('\n步骤6: 工作4 - 示例数据端到端流程')
    pipeline_state.start_step('end_to_end')
    try:
        test_input = {
            'feature_type': '四边形腔',
            'length': 100,
            'width': 80,
            'diameter': None,
            'depth': 40,
            'precision': 'IT8',
            'roughness': 6.3,
        }
        results = await run_end_to_end(test_input)
        step_results['end_to_end'] = {'result_count': len(results)}
        logger.info(f'端到端流程完成，生成 {len(results)} 条推荐结果')
        if results:
            logger.info(
                f'推荐结果示例: {json.dumps(results[0], ensure_ascii=False, indent=2, default=str)[:400]}...'
            )
        pipeline_state.finish_step('end_to_end', f'生成{len(results)}条推荐结果')
    except Exception as e:
        logger.error(f'端到端流程失败: {e}')
        pipeline_state.fail_step('end_to_end', str(e))

    logger.info('\n步骤7: 工作4 - TMCAD数据集端到端流程（每类2个零件）')
    pipeline_state.start_step('tmcad_end_to_end')
    try:
        tmcad_results = await run_tmcad_end_to_end(max_per_category=2)
        step_results['tmcad_end_to_end'] = {'result_count': len(tmcad_results)}
        logger.info(f'TMCAD流程完成，共 {len(tmcad_results)} 条推荐结果')
        pipeline_state.finish_step('tmcad_end_to_end', f'共{len(tmcad_results)}条推荐结果')
    except Exception as e:
        logger.error(f'TMCAD流程失败: {e}')
        pipeline_state.fail_step('tmcad_end_to_end', str(e))

    pipeline_state.end_time = datetime.now(timezone.utc)

    report_text = pipeline_state.format_report()
    logger.info(f'\n{report_text}')

    save_pipeline_report(pipeline_state, step_results)

    logger.info('\n所有步骤执行完毕')


if __name__ == '__main__':
    asyncio.run(main())
