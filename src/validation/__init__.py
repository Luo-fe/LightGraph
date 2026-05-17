import contextlib
import json
import logging
from datetime import datetime

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

    async def validate_typical_part(self, part_name: str, test_cases: list[dict]) -> dict:
        results = []
        correct_method = 0
        correct_tool = 0
        correct_spindle = 0
        correct_feed = 0
        total = len(test_cases)

        for i, test_case in enumerate(test_cases):
            expected_method = test_case.get('expected_method', '')
            expected_tool_diameter = test_case.get('expected_tool_diameter')
            expected_spindle_range = test_case.get('expected_spindle_speed_range')
            expected_feed_range = test_case.get('expected_feed_rate_range')

            feature = self.extractor.extract_from_structured_data(test_case)
            if feature is None:
                logger.warning(f'零件 {part_name} 测试用例 {i} 特征提取失败')
                results.append({
                    'test_case': i,
                    'feature_type': test_case.get('feature_type', ''),
                    'status': 'extraction_failed',
                    'match_scores': {},
                })
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

            spindle_match = None
            if expected_spindle_range:
                params = json_output.get('加工参数', {})
                spindle_speed = params.get('主轴转速(r/min)')
                if spindle_speed:
                    spindle_val = float(spindle_speed)
                    spindle_match = (
                        expected_spindle_range[0] <= spindle_val <= expected_spindle_range[1]
                    )
                    if spindle_match:
                        correct_spindle += 1

            feed_match = None
            if expected_feed_range:
                params = json_output.get('加工参数', {})
                feed_rate = params.get('进给量(mm/r)') or params.get('进给速度(mm/min)')
                if feed_rate:
                    feed_val = float(feed_rate)
                    feed_match = expected_feed_range[0] <= feed_val <= expected_feed_range[1]
                    if feed_match:
                        correct_feed += 1

            match_scores = {}
            if method_match is not None:
                match_scores['method'] = 1.0 if method_match else 0.0
            if tool_match is not None:
                match_scores['tool'] = 1.0 if tool_match else 0.0
            if spindle_match is not None:
                match_scores['spindle_speed'] = 1.0 if spindle_match else 0.0
            if feed_match is not None:
                match_scores['feed_rate'] = 1.0 if feed_match else 0.0

            results.append({
                'test_case': i,
                'feature_type': test_case.get('feature_type', ''),
                'expected_method': expected_method,
                'recommended_method': recommended_method,
                'method_match': method_match,
                'tool_match': tool_match,
                'spindle_match': spindle_match,
                'feed_match': feed_match,
                'match_scores': match_scores,
                'output': json_output,
            })
            logger.info(
                f'  零件{part_name} 用例{i}: 特征={test_case.get("feature_type", "")}, '
                f'期望={expected_method}, 推荐={recommended_method}, '
                f'方法匹配={method_match}, 刀具匹配={tool_match}, '
                f'转速匹配={spindle_match}, 进给匹配={feed_match}'
            )

        mdpm = correct_method / total if total > 0 else 0
        mdmt = correct_tool / total if total > 0 else 0
        mdpp = (
            (correct_spindle + correct_feed)
            / (total * 2)
            if total > 0
            else 0
        )

        return {
            'part_name': part_name,
            'total_features': total,
            'results': results,
            'mdpm': mdpm,
            'mdmt': mdmt,
            'mdpp': mdpp,
        }

    async def generate_validation_report(
        self,
        test_cases: list[dict] | None = None,
        typical_parts: list[str] | None = None,
    ) -> dict:
        if test_cases is None:
            test_cases = DEFAULT_TEST_CASES
        if typical_parts is None:
            typical_parts = list(TYPICAL_PART_TEST_CASES.keys())

        basic_report = await self.validate(test_cases)

        typical_reports = []
        for part_name in typical_parts:
            if part_name not in TYPICAL_PART_TEST_CASES:
                logger.warning(f'未找到典型零件 {part_name} 的测试用例')
                continue
            part_cases = TYPICAL_PART_TEST_CASES[part_name]
            part_report = await self.validate_typical_part(part_name, part_cases)
            typical_reports.append(part_report)

        mdpm_values = [basic_report['mdpm']]
        mdmt_values = [v for v in [basic_report['mdmt']] if v is not None]
        mdpp_values = []

        for tr in typical_reports:
            mdpm_values.append(tr['mdpm'])
            mdmt_values.append(tr['mdmt'])
            mdpp_values.append(tr['mdpp'])

        overall_mdpm = sum(mdpm_values) / len(mdpm_values) if mdpm_values else 0
        overall_mdmt = sum(mdmt_values) / len(mdmt_values) if mdmt_values else 0
        overall_mdpp = sum(mdpp_values) / len(mdpp_values) if mdpp_values else 0

        overall_score = overall_mdpm * 0.4 + overall_mdmt * 0.3 + overall_mdpp * 0.3

        suggestions = []
        if overall_mdpm < 0.8:
            suggestions.append('加工方法匹配度偏低，建议优化知识图谱中的加工方法规则和语义匹配策略')
        if overall_mdmt < 0.8:
            suggestions.append('加工刀具匹配度偏低，建议完善刀具选择知识库和刀具直径推荐逻辑')
        if overall_mdpp < 0.8:
            suggestions.append('加工参数匹配度偏低，建议调整切削参数推荐模型和参数范围约束')
        if overall_score >= 0.9:
            suggestions.append('整体验证表现优秀，可关注边缘用例的进一步优化')
        if not suggestions:
            suggestions.append('各维度匹配度良好，系统推荐质量达标')

        report = {
            'timestamp': datetime.now().isoformat(),
            'basic_validation': basic_report,
            'typical_part_validation': typical_reports,
            'metrics': {
                'MDPM': round(overall_mdpm, 4),
                'MDMT': round(overall_mdmt, 4),
                'MDPP': round(overall_mdpp, 4),
                'overall_score': round(overall_score, 4),
            },
            'suggestions': suggestions,
        }

        return report

    def save_validation_report(self, report: dict, filename: str | None = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'validation_report_{timestamp}.json'

        report_path = DATA_DIR / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f'验证报告已保存至 {report_path}')
        return str(report_path)

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
    {
        'feature_type': '外圆',
        'diameter': 50,
        'length': 120,
        'precision': 'IT7',
        'roughness': 1.6,
        'expected_method': '粗车-半精车-精车',
        'expected_tool_diameter': 50,
        'expected_spindle_speed_range': [600, 1200],
        'expected_feed_rate_range': [0.1, 0.3],
    },
    {
        'feature_type': '盲孔',
        'diameter': 8,
        'depth': 15,
        'precision': 'IT8',
        'roughness': 3.2,
        'expected_method': '钻-扩',
        'expected_tool_diameter': 8,
        'expected_spindle_speed_range': [800, 1500],
        'expected_feed_rate_range': [0.05, 0.2],
    },
    {
        'feature_type': '台阶面',
        'length': 60,
        'width': 40,
        'depth': 5,
        'precision': 'IT9',
        'roughness': 6.3,
        'expected_method': '粗铣-半精铣',
        'expected_tool_diameter': 16,
        'expected_spindle_speed_range': [500, 1000],
        'expected_feed_rate_range': [0.1, 0.4],
    },
    {
        'feature_type': '圆柱面',
        'diameter': 30,
        'length': 80,
        'precision': 'IT6',
        'roughness': 0.8,
        'expected_method': '粗车-半精车-精车-磨削',
        'expected_tool_diameter': 30,
        'expected_spindle_speed_range': [800, 1500],
        'expected_feed_rate_range': [0.05, 0.15],
    },
    {
        'feature_type': '圆形腔',
        'diameter': 25,
        'depth': 20,
        'precision': 'IT8',
        'roughness': 3.2,
        'expected_method': '粗铣-半精铣',
        'expected_tool_diameter': 10,
        'expected_spindle_speed_range': [600, 1200],
        'expected_feed_rate_range': [0.1, 0.3],
    },
    {
        'feature_type': '螺纹孔',
        'diameter': 6,
        'depth': 12,
        'precision': 'IT7',
        'roughness': 3.2,
        'expected_method': '钻-攻丝',
        'expected_tool_diameter': 6,
        'expected_spindle_speed_range': [200, 600],
        'expected_feed_rate_range': [0.5, 1.5],
    },
]

TYPICAL_PART_TEST_CASES = {
    'shaft': [
        {
            'feature_type': '外圆',
            'diameter': 45,
            'length': 200,
            'precision': 'IT7',
            'roughness': 1.6,
            'expected_method': '粗车-半精车-精车',
            'expected_tool_diameter': 45,
            'expected_spindle_speed_range': [600, 1200],
            'expected_feed_rate_range': [0.1, 0.3],
        },
        {
            'feature_type': '圆柱面',
            'diameter': 35,
            'length': 80,
            'precision': 'IT6',
            'roughness': 0.8,
            'expected_method': '粗车-半精车-精车-磨削',
            'expected_tool_diameter': 35,
            'expected_spindle_speed_range': [800, 1500],
            'expected_feed_rate_range': [0.05, 0.15],
        },
        {
            'feature_type': '圆锥面',
            'diameter': 30,
            'length': 50,
            'precision': 'IT7',
            'roughness': 1.6,
            'expected_method': '粗车-半精车-精车',
            'expected_tool_diameter': 30,
            'expected_spindle_speed_range': [600, 1000],
            'expected_feed_rate_range': [0.08, 0.25],
        },
        {
            'feature_type': '螺纹',
            'diameter': 20,
            'length': 30,
            'precision': 'IT7',
            'roughness': 3.2,
            'expected_method': '车螺纹',
            'expected_tool_diameter': 20,
            'expected_spindle_speed_range': [200, 600],
            'expected_feed_rate_range': [1.0, 3.0],
        },
        {
            'feature_type': '键槽',
            'length': 40,
            'width': 8,
            'depth': 4,
            'precision': 'IT9',
            'roughness': 6.3,
            'expected_method': '粗铣-半精铣',
            'expected_tool_diameter': 8,
            'expected_spindle_speed_range': [500, 1000],
            'expected_feed_rate_range': [0.1, 0.3],
        },
        {
            'feature_type': '外圆',
            'diameter': 25,
            'length': 60,
            'precision': 'IT6',
            'roughness': 0.4,
            'expected_method': '粗车-半精车-精车-磨削',
            'expected_tool_diameter': 25,
            'expected_spindle_speed_range': [1000, 2000],
            'expected_feed_rate_range': [0.03, 0.1],
        },
    ],
    'gear': [
        {
            'feature_type': '齿形',
            'diameter': 120,
            'length': 30,
            'precision': 'IT6',
            'roughness': 0.8,
            'expected_method': '滚齿-剃齿',
            'expected_tool_diameter': 80,
            'expected_spindle_speed_range': [100, 400],
            'expected_feed_rate_range': [0.5, 2.0],
        },
        {
            'feature_type': '外圆',
            'diameter': 124,
            'length': 30,
            'precision': 'IT7',
            'roughness': 1.6,
            'expected_method': '粗车-半精车-精车',
            'expected_tool_diameter': 124,
            'expected_spindle_speed_range': [300, 800],
            'expected_feed_rate_range': [0.1, 0.3],
        },
        {
            'feature_type': '通孔',
            'diameter': 30,
            'depth': 30,
            'precision': 'IT7',
            'roughness': 1.6,
            'expected_method': '钻-扩-铰',
            'expected_tool_diameter': 30,
            'expected_spindle_speed_range': [400, 800],
            'expected_feed_rate_range': [0.1, 0.3],
        },
        {
            'feature_type': '端面',
            'diameter': 124,
            'length': 30,
            'precision': 'IT8',
            'roughness': 3.2,
            'expected_method': '粗车-半精车',
            'expected_tool_diameter': 124,
            'expected_spindle_speed_range': [300, 800],
            'expected_feed_rate_range': [0.1, 0.3],
        },
        {
            'feature_type': '键槽',
            'length': 25,
            'width': 8,
            'depth': 3.3,
            'precision': 'IT9',
            'roughness': 6.3,
            'expected_method': '粗铣-半精铣',
            'expected_tool_diameter': 8,
            'expected_spindle_speed_range': [500, 1000],
            'expected_feed_rate_range': [0.1, 0.3],
        },
    ],
}
