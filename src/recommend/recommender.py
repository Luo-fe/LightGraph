import json
import logging

from src.config.settings import (
    FEATURE_NAME_MAP,
    FEED_RATE_RANGE,
    MACHINING_METHODS,
    SPINDLE_SPEED_RANGE,
)
from src.data.structurer import MachiningFeature
from src.data.vector_store import VectorStore
from src.graph.glm_client import GLMClient
from src.graph.knowledge_graph import CADKnowledgeGraph

logger = logging.getLogger(__name__)


def _flatten_param(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace('rpm', '').replace('mm/min', '').replace('mm', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    if isinstance(value, dict):
        vals = []
        for v in value.values():
            f = _flatten_param(v)
            if f is not None:
                vals.append(f)
        return sum(vals) / len(vals) if vals else None
    return None


def _clamp_param(value: float | None, min_val: float, max_val: float) -> float:
    if value is None:
        return (min_val + max_val) / 2
    return max(min_val, min(max_val, value))


def _determine_process_type(method: str) -> str:
    if any(k in method for k in ['铣', '腔', '槽']):
        return 'milling'
    if any(k in method for k in ['钻', '扩', '铰']):
        return 'drilling'
    if any(k in method for k in ['车', '圆']):
        return 'turning'
    return 'turning'


class ProcessRecommender:
    def __init__(
        self,
        glm_client: GLMClient | None = None,
        knowledge_graph: CADKnowledgeGraph | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.glm_client = glm_client or GLMClient()
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store

    async def _gather_context(self, feature_info: dict) -> tuple[str, dict]:
        context_parts = []
        provenance = {
            '向量库检索': [],
            '知识图谱检索': {
                '已连接': False,
                '事实': [],
                '实体': [],
                '案例': [],
            },
        }

        if self.vector_store:
            query = (
                f"{feature_info['feature_name_cn']} "
                f"精度{feature_info.get('precision')} "
                f"粗糙度{feature_info.get('roughness')}"
            )
            try:
                similar_docs = await self.vector_store.search(query, top_k=3)
                for doc, score in similar_docs:
                    summary = self._summarize_doc(doc)
                    context_parts.append(
                        f'[相似案例(相似度{score:.2f})]: {summary}'
                    )
                    provenance['向量库检索'].append({
                        '相似度': round(score, 4),
                        '摘要': summary,
                    })
            except Exception as e:
                logger.warning(f'向量检索失败: {e}')

        if self.knowledge_graph and self.knowledge_graph.is_connected:
            provenance['知识图谱检索']['已连接'] = True

            try:
                facts = await self.knowledge_graph.search_process_facts(
                    f"{feature_info['feature_name_cn']}加工工艺", limit=3
                )
                for fact in facts:
                    context_parts.append(f'[知识图谱事实]: {fact}')
                    provenance['知识图谱检索']['事实'].append(fact)
            except Exception as e:
                logger.warning(f'知识图谱事实检索失败: {e}')

            try:
                kg_items = await self.knowledge_graph.search_knowledge(
                    f"{feature_info['feature_name_cn']}加工参数推荐", limit=3
                )
                for item in kg_items:
                    if item['type'] == 'node':
                        summary = item.get('summary', '')
                        context_parts.append(
                            f'[知识图谱实体]: {item["name"]}'
                            + (f' - {summary}' if summary else '')
                        )
                        provenance['知识图谱检索']['实体'].append({
                            '名称': item['name'],
                            '摘要': summary,
                        })
                    elif item['type'] == 'episode':
                        context_parts.append(
                            f'[知识图谱案例]: {item.get("content", "")}'
                        )
                        provenance['知识图谱检索']['案例'].append(
                            item.get('content', '')
                        )
                    elif item['type'] == 'edge':
                        context_parts.append(
                            f'[知识图谱关系]: {item["name"]} - {item.get("fact", "")}'
                        )
                        provenance['知识图谱检索']['事实'].append(
                            f"{item['name']}: {item.get('fact', '')}"
                        )
            except Exception as e:
                logger.warning(f'知识图谱综合检索失败: {e}')

        return '\n'.join(context_parts), provenance

    @staticmethod
    def _summarize_doc(doc: dict) -> str:
        feature = doc.get('feature', {})
        if isinstance(feature, dict):
            ft = feature.get('feature_type', '')
            method = doc.get('machining_method', '')
            route = doc.get('process_route', '')
            return f'{ft} → {method}: {route}'
        return json.dumps(doc, ensure_ascii=False)[:200]

    async def recommend(self, feature: MachiningFeature) -> dict:
        feature_info = feature.model_dump()
        feature_info['feature_name_cn'] = FEATURE_NAME_MAP.get(
            feature.feature_type, feature.feature_type
        )

        context, provenance = await self._gather_context(feature_info)

        result = await self.glm_client.recommend_process(feature_info, context)
        result['feature'] = feature_info
        result['provenance'] = provenance
        return result

    async def recommend_with_validation(self, feature: MachiningFeature) -> dict:
        result = await self.recommend(feature)

        feature_type = feature.feature_type
        if feature_type in MACHINING_METHODS:
            valid_methods = MACHINING_METHODS[feature_type]
            recommended_method = result.get('machining_method', '')
            method_valid = any(m in recommended_method for m in valid_methods)
            result['method_validation'] = {
                'valid': method_valid,
                'recommended': recommended_method,
                'valid_methods': valid_methods,
            }

        return result

    def format_json_output(self, recommendation: dict) -> dict:
        output = {
            '加工特征': recommendation.get('feature', {}).get('feature_name_cn', ''),
            '特征参数': {},
            '加工方法': recommendation.get('machining_method', ''),
            '加工工艺路线': recommendation.get('process_route', ''),
            '加工参数': {},
        }

        feature = recommendation.get('feature', {})
        for key in ['length', 'width', 'diameter', 'depth', 'precision', 'roughness']:
            if feature.get(key) is not None:
                output['特征参数'][key] = feature[key]

        raw_params = recommendation.get('parameters', {})
        method = recommendation.get('machining_method', '')
        process_type = _determine_process_type(method)

        feed_range = FEED_RATE_RANGE.get(process_type, (30, 800))
        speed_range = SPINDLE_SPEED_RANGE.get(process_type, (200, 8000))

        param_map = {
            'spindle_speed': ('主轴转速(rpm)', speed_range[0], speed_range[1]),
            'feed_rate': ('进给速度(mm/min)', feed_range[0], feed_range[1]),
            'tool_diameter': ('刀具直径(mm)', 0.5, 200.0),
            'cutting_depth': ('切削深度(mm)', 0.05, 50.0),
            'cutting_width': ('切削宽度(mm)', 0.1, 50.0),
        }
        for eng, (chn, min_v, max_v) in param_map.items():
            raw_val = raw_params.get(eng)
            flat_val = _flatten_param(raw_val)
            clamped_val = _clamp_param(flat_val, min_v, max_v)
            if eng == 'feed_rate' and clamped_val < 30:
                clamped_val = 30.0
            output['加工参数'][chn] = round(clamped_val, 2)

        if 'method_validation' in recommendation:
            output['方法验证'] = recommendation['method_validation']

        provenance = recommendation.get('provenance', {})
        kg_info = provenance.get('知识图谱检索', {})
        vec_info = provenance.get('向量库检索', [])

        knowledge_sources = {}
        if vec_info:
            knowledge_sources['向量库(相似案例)'] = [
                v['摘要'] for v in vec_info
            ]
        if kg_info.get('已连接'):
            knowledge_sources['知识图谱(Neo4j)'] = {}
            if kg_info.get('事实'):
                knowledge_sources['知识图谱(Neo4j)']['事实'] = kg_info['事实']
            if kg_info.get('实体'):
                knowledge_sources['知识图谱(Neo4j)']['实体'] = [
                    e['名称'] for e in kg_info['实体']
                ]
            if kg_info.get('案例'):
                knowledge_sources['知识图谱(Neo4j)']['案例'] = kg_info['案例']
        elif kg_info and not kg_info.get('已连接'):
            knowledge_sources['知识图谱(Neo4j)'] = '未连接'

        if knowledge_sources:
            output['知识来源'] = knowledge_sources

        return output
