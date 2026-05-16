import contextlib
import json
import logging

from src.config.settings import DATA_DIR
from src.data.structurer import MachiningFeature as MachiningFeature
from src.data.vector_store import VectorStore
from src.feature.extractor import FeatureExtractor
from src.graph.glm_client import GLMClient as GLMClient
from src.graph.glm_embedder import GLMEmbedder
from src.graph.knowledge_graph import CADKnowledgeGraph
from src.recommend.recommender import ProcessRecommender

logger = logging.getLogger(__name__)


class ValidationReport:
    def __init__(
        self,
        extractor: FeatureExtractor,
        recommender: ProcessRecommender,
        embedder: GLMEmbedder,
        kg: CADKnowledgeGraph | None,
    ):
        self.extractor = extractor
        self.recommender = recommender
        self.embedder = embedder
        self.kg = kg

    async def validate(self, test_cases: list[dict]) -> dict:
        results = []
        correct_method = 0
        correct_tool = 0
        total = len(test_cases)

        for i, test_case in enumerate(test_cases):
            expected_method = test_case.get('expected_method', '')
            expected_tool_diameter = test_case.get('expected_tool_diameter')

            feature = self.extractor.extract_from_structured_data(test_case)
            if feature is None:
                logger.warning(f'测试用例 {i} 特征提取失败')
                continue

            recommendation = await self.recommender.recommend_with_validation(feature)
            json_output = self.recommender.format_json_output(recommendation)

            recommended_method = json_output.get('加工方法', '')
            method_match = expected_method in recommended_method if expected_method else None
            if method_match:
                correct_method += 1

            tool_match = None
            if expected_tool_diameter:
                params = json_output.get('加工参数', {})
                tool_d = params.get('刀具直径(mm)')
                if tool_d:
                    tool_match = (
                        abs(float(tool_d) - float(expected_tool_diameter))
                        / float(expected_tool_diameter)
                        < 0.2
                    )
                    if tool_match:
                        correct_tool += 1

            results.append({
                'test_case': i,
                'feature_type': test_case.get('feature_type', ''),
                'expected_method': expected_method,
                'recommended_method': recommended_method,
                'method_match': method_match,
                'tool_match': tool_match,
                'output': json_output,
            })
            logger.info(
                f'  用例{i}: 特征={test_case.get("feature_type", "")}, '
                f'期望={expected_method}, 推荐={recommended_method}, '
                f'方法匹配={method_match}, 刀具匹配={tool_match}'
            )

        mdpm = correct_method / total if total > 0 else 0
        mdmt = (
            correct_tool / total
            if total > 0 and any(tc.get('expected_tool_diameter') for tc in test_cases)
            else None
        )

        report = {
            'total_cases': total,
            'mdpm': mdpm,
            'mdmt': mdmt,
            'results': results,
        }

        report_path = DATA_DIR / 'validation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f'验证报告已保存至 {report_path}')
        logger.info(f'MDPM(加工方法匹配度)={mdpm:.2%}, MDMT(加工刀具匹配度)={mdmt}')

        return report

    async def cleanup(self):
        await self.extractor.glm_client.close()
        await self.embedder.close()
        if self.kg:
            await self.kg.close()


async def create_validator() -> ValidationReport:
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

    return ValidationReport(
        extractor=extractor,
        recommender=recommender,
        embedder=embedder,
        kg=kg,
    )


DEFAULT_TEST_CASES = [
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
