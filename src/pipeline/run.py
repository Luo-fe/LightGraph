import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import (
    DATA_DIR,
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

    cleaned_data = cleaner.clean_dataset(raw_data)
    cleaner.save_cleaned_data(cleaned_data)

    cases = structurer.structure_dataset(cleaned_data)
    structurer.save_structured_data(cases)

    training_data = structurer.generate_training_data(cases)
    structurer.save_training_data(training_data)

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
        f'数据处理完成: 原始{len(raw_data)}条 -> 清洗{len(cleaned_data)}条 '
        f'-> 结构化{len(cases)}条 -> 向量库{len(docs)}条 -> 知识图谱{kg_count}条'
    )
    return {
        'raw_count': len(raw_data),
        'cleaned_count': len(cleaned_data),
        'structured_count': len(cases),
        'training_count': len(training_data),
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

    features = await run_feature_recognition(feature_input)

    if not features:
        logger.error('未识别到任何加工特征')
        return []

    results = await run_process_recommendation(features)

    output_path = DATA_DIR / 'output'
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    filepath = output_path / f'result_{timestamp}.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
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
    ]


async def main():
    logger.info('CAD工艺知识图谱系统启动')

    logger.info('\n步骤1: 测试GLM API连接')
    try:
        glm_client = GLMClient()
        response = await glm_client.chat([
            {'role': 'user', 'content': '请用一句话介绍数控加工'}
        ])
        logger.info(f'GLM API连接成功，响应: {response[:80]}...')
        await glm_client.close()
    except Exception as e:
        logger.error(f'GLM API连接失败: {e}')
        return

    logger.info('\n步骤2: 工作1 - 填充知识库（数据收集→清洗→结构化→向量库→知识图谱）')
    data_result = await run_data_pipeline()
    logger.info(f'数据处理结果: {data_result}')

    logger.info('\n步骤3: 工作3 - 典型零件原型验证')
    test_cases = _get_validation_test_cases()
    validation_report = await run_validation(test_cases)
    logger.info(
        f'验证结果: MDPM={validation_report["mdpm"]:.2%}, '
        f'MDMT={validation_report["mdmt"]}'
    )

    logger.info('\n步骤4: 工作4 - 示例数据端到端流程')
    test_input = {
        'feature_type': '四边形腔',
        'length': 100,
        'width': 80,
        'diameter': None,
        'depth': 40,
        'precision': 'IT8',
        'roughness': 6.3,
    }
    try:
        results = await run_end_to_end(test_input)
        logger.info(f'端到端流程完成，生成 {len(results)} 条推荐结果')
        if results:
            logger.info(
                f'推荐结果示例: {json.dumps(results[0], ensure_ascii=False, indent=2)[:400]}...'
            )
    except Exception as e:
        logger.error(f'端到端流程失败: {e}')

    logger.info('\n步骤5: 工作4 - TMCAD数据集端到端流程（每类1个零件）')
    try:
        tmcad_results = await run_tmcad_end_to_end(max_per_category=1)
        logger.info(f'TMCAD流程完成，共 {len(tmcad_results)} 条推荐结果')
    except Exception as e:
        logger.error(f'TMCAD流程失败: {e}')

    logger.info('\n所有步骤执行完毕')


if __name__ == '__main__':
    asyncio.run(main())
