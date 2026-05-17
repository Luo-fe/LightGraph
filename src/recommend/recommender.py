import contextlib
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
        if 'content' in doc and isinstance(doc['content'], str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                doc = json.loads(doc['content'])
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

    def validate_parameters(self, recommendation: dict) -> dict:
        raw_params = recommendation.get('parameters', {})
        method = recommendation.get('machining_method', '')
        process_type = _determine_process_type(method)

        feed_range = FEED_RATE_RANGE.get(process_type, (30, 800))
        speed_range = SPINDLE_SPEED_RANGE.get(process_type, (200, 8000))

        param_constraints = {
            'spindle_speed': {
                'name': '主轴转速(rpm)',
                'min': speed_range[0],
                'max': speed_range[1],
            },
            'feed_rate': {
                'name': '进给速度(mm/min)',
                'min': feed_range[0],
                'max': feed_range[1],
            },
            'tool_diameter': {
                'name': '刀具直径(mm)',
                'min': 0.5,
                'max': 200.0,
            },
            'cutting_depth': {
                'name': '切削深度(mm)',
                'min': 0.05,
                'max': 50.0,
            },
            'cutting_width': {
                'name': '切削宽度(mm)',
                'min': 0.1,
                'max': 50.0,
            },
        }

        all_valid = True
        param_results = {}
        suggestions = {}

        for eng_key, constraint in param_constraints.items():
            raw_val = raw_params.get(eng_key)
            flat_val = _flatten_param(raw_val)
            chn_name = constraint['name']
            min_v = constraint['min']
            max_v = constraint['max']

            if flat_val is None:
                all_valid = False
                param_results[eng_key] = {
                    'valid': False,
                    'value': None,
                    'reason': f'{chn_name}参数缺失',
                }
                suggestions[eng_key] = {
                    'suggested_value': round((min_v + max_v) / 2, 2),
                    'reason': f'{chn_name}未提供，建议使用范围中值{(min_v + max_v) / 2:.2f}',
                }
            elif flat_val < min_v:
                all_valid = False
                param_results[eng_key] = {
                    'valid': False,
                    'value': flat_val,
                    'reason': f'{chn_name}={flat_val:.2f}低于最小值{min_v}',
                }
                suggestions[eng_key] = {
                    'suggested_value': round(min_v, 2),
                    'reason': f'{chn_name}低于下限，建议修正为{min_v}',
                }
            elif flat_val > max_v:
                all_valid = False
                param_results[eng_key] = {
                    'valid': False,
                    'value': flat_val,
                    'reason': f'{chn_name}={flat_val:.2f}超过最大值{max_v}',
                }
                suggestions[eng_key] = {
                    'suggested_value': round(max_v, 2),
                    'reason': f'{chn_name}超过上限，建议修正为{max_v}',
                }
            else:
                param_results[eng_key] = {
                    'valid': True,
                    'value': flat_val,
                    'reason': '参数在合理范围内',
                }

        return {
            'all_valid': all_valid,
            'process_type': process_type,
            'param_results': param_results,
            'suggestions': suggestions,
        }

    async def recommend_multi_candidates(self, feature: MachiningFeature, top_k: int = 3) -> list[dict]:
        feature_info = feature.model_dump()
        feature_info['feature_name_cn'] = FEATURE_NAME_MAP.get(
            feature.feature_type, feature.feature_type
        )

        context, provenance = await self._gather_context(feature_info)

        primary = await self.glm_client.recommend_process(feature_info, context)
        primary['feature'] = feature_info
        primary['provenance'] = provenance
        primary['candidate_source'] = 'primary'

        candidates = [primary]

        feature_type = feature.feature_type
        alt_methods = MACHINING_METHODS.get(feature_type, [])
        primary_method = primary.get('machining_method', '')

        for alt_method in alt_methods:
            if any(m in primary_method for m in [alt_method]):
                continue

            alt_feature_info = dict(feature_info)
            alt_feature_info['preferred_method'] = alt_method

            alt_context = context + f'\n[备选方法约束]: 请优先考虑加工方法"{alt_method}"'

            try:
                alt_result = await self.glm_client.recommend_process(
                    alt_feature_info, alt_context
                )
                alt_result['feature'] = feature_info
                alt_result['provenance'] = provenance
                alt_result['candidate_source'] = f'alternative_{alt_method}'
                candidates.append(alt_result)
            except Exception as e:
                logger.warning(f'备选方案生成失败({alt_method}): {e}')

            if len(candidates) >= top_k:
                break

        while len(candidates) < top_k:
            variant = dict(primary)
            variant_params = dict(primary.get('parameters', {}))
            process_type = _determine_process_type(primary_method)
            speed_range = SPINDLE_SPEED_RANGE.get(process_type, (200, 8000))
            feed_range = FEED_RATE_RANGE.get(process_type, (30, 800))

            variant_speed = _flatten_param(variant_params.get('spindle_speed'))
            variant_feed = _flatten_param(variant_params.get('feed_rate'))

            if variant_speed is not None:
                variant_params['spindle_speed'] = round(
                    _clamp_param(
                        variant_speed * (0.8 + 0.4 * len(candidates) / top_k),
                        speed_range[0], speed_range[1],
                    ), 2
                )
            if variant_feed is not None:
                variant_params['feed_rate'] = round(
                    _clamp_param(
                        variant_feed * (0.8 + 0.4 * len(candidates) / top_k),
                        feed_range[0], feed_range[1],
                    ), 2
                )

            variant['parameters'] = variant_params
            variant['candidate_source'] = f'variant_{len(candidates)}'
            candidates.append(variant)

        return candidates[:top_k]

    def rank_candidates(self, candidates: list[dict], feature: MachiningFeature) -> list[dict]:
        feature_type = feature.feature_type
        valid_methods = MACHINING_METHODS.get(feature_type, [])

        scored = []
        for candidate in candidates:
            score = 0.0
            details = {}

            method = candidate.get('machining_method', '')
            method_match = any(m in method for m in valid_methods) if valid_methods else True
            if method_match:
                score += 40.0
                details['method_score'] = 40.0
            else:
                score += 5.0
                details['method_score'] = 5.0
            details['method_valid'] = method_match

            validation = self.validate_parameters(candidate)
            valid_count = sum(
                1 for v in validation['param_results'].values() if v['valid']
            )
            total_count = len(validation['param_results'])
            param_score = (valid_count / total_count * 40.0) if total_count > 0 else 0.0
            score += param_score
            details['param_score'] = round(param_score, 2)
            details['param_valid_count'] = valid_count
            details['param_total_count'] = total_count

            provenance = candidate.get('provenance', {})
            vec_info = provenance.get('向量库检索', [])
            kg_info = provenance.get('知识图谱检索', {})

            kb_score = 0.0
            if vec_info:
                avg_sim = sum(v.get('相似度', 0) for v in vec_info) / len(vec_info)
                kb_score += avg_sim * 10.0

            if kg_info.get('已连接'):
                fact_count = len(kg_info.get('事实', []))
                entity_count = len(kg_info.get('实体', []))
                case_count = len(kg_info.get('案例', []))
                kb_score += min(fact_count * 2.0, 6.0)
                kb_score += min(entity_count * 1.5, 4.5)
                kb_score += min(case_count * 2.5, 5.0)

            kb_score = min(kb_score, 20.0)
            score += kb_score
            details['kb_score'] = round(kb_score, 2)

            candidate['rank_score'] = round(score, 2)
            candidate['rank_details'] = details
            scored.append(candidate)

        scored.sort(key=lambda x: x['rank_score'], reverse=True)

        for i, c in enumerate(scored):
            c['rank_position'] = i + 1

        return scored

    async def recommend_with_full_validation(self, feature: MachiningFeature) -> dict:
        candidates = await self.recommend_multi_candidates(feature, top_k=3)

        ranked = self.rank_candidates(candidates, feature)

        primary = ranked[0]

        validation = self.validate_parameters(primary)

        feature_type = feature.feature_type
        method_validation = None
        if feature_type in MACHINING_METHODS:
            valid_methods = MACHINING_METHODS[feature_type]
            recommended_method = primary.get('machining_method', '')
            method_valid = any(m in recommended_method for m in valid_methods)
            method_validation = {
                'valid': method_valid,
                'recommended': recommended_method,
                'valid_methods': valid_methods,
            }

        primary['parameter_validation'] = validation
        if method_validation is not None:
            primary['method_validation'] = method_validation

        for candidate in ranked[1:]:
            candidate['parameter_validation'] = self.validate_parameters(candidate)
            ft = feature.feature_type
            if ft in MACHINING_METHODS:
                vm = MACHINING_METHODS[ft]
                rm = candidate.get('machining_method', '')
                mv = any(m in rm for m in vm)
                candidate['method_validation'] = {
                    'valid': mv,
                    'recommended': rm,
                    'valid_methods': vm,
                }

        primary['candidates'] = ranked

        return primary

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

    def format_detailed_output(self, recommendation: dict) -> dict:
        output = self.format_json_output(recommendation)

        if 'parameter_validation' in recommendation:
            pv = recommendation['parameter_validation']
            validation_output = {
                '校验通过': pv.get('all_valid', False),
                '工艺类型': pv.get('process_type', ''),
                '参数校验详情': {},
                '修正建议': {},
            }
            for eng_key, result in pv.get('param_results', {}).items():
                validation_output['参数校验详情'][eng_key] = {
                    'valid': result['valid'],
                    'value': result.get('value'),
                    'reason': result.get('reason', ''),
                }
            for eng_key, sug in pv.get('suggestions', {}).items():
                validation_output['修正建议'][eng_key] = {
                    'suggested_value': sug['suggested_value'],
                    'reason': sug['reason'],
                }
            output['参数校验'] = validation_output

        candidates = recommendation.get('candidates', [])
        if candidates:
            candidate_list = []
            for c in candidates:
                candidate_item = {
                    '排名': c.get('rank_position', 0),
                    '加工方法': c.get('machining_method', ''),
                    '加工工艺路线': c.get('process_route', ''),
                    '评分': c.get('rank_score', 0),
                    '评分详情': c.get('rank_details', {}),
                    '来源': c.get('candidate_source', ''),
                    '加工参数': {},
                }

                c_params = c.get('parameters', {})
                c_method = c.get('machining_method', '')
                c_process_type = _determine_process_type(c_method)
                c_feed_range = FEED_RATE_RANGE.get(c_process_type, (30, 800))
                c_speed_range = SPINDLE_SPEED_RANGE.get(c_process_type, (200, 8000))

                c_param_map = {
                    'spindle_speed': ('主轴转速(rpm)', c_speed_range[0], c_speed_range[1]),
                    'feed_rate': ('进给速度(mm/min)', c_feed_range[0], c_feed_range[1]),
                    'tool_diameter': ('刀具直径(mm)', 0.5, 200.0),
                    'cutting_depth': ('切削深度(mm)', 0.05, 50.0),
                    'cutting_width': ('切削宽度(mm)', 0.1, 50.0),
                }
                for eng, (chn, min_v, max_v) in c_param_map.items():
                    raw_val = c_params.get(eng)
                    flat_val = _flatten_param(raw_val)
                    clamped_val = _clamp_param(flat_val, min_v, max_v)
                    if eng == 'feed_rate' and clamped_val < 30:
                        clamped_val = 30.0
                    candidate_item['加工参数'][chn] = round(clamped_val, 2)

                if 'method_validation' in c:
                    candidate_item['方法验证'] = c['method_validation']

                if 'parameter_validation' in c:
                    cpv = c['parameter_validation']
                    candidate_item['参数校验'] = {
                        '校验通过': cpv.get('all_valid', False),
                        '参数校验详情': {
                            k: {'valid': v['valid'], 'value': v.get('value'), 'reason': v.get('reason', '')}
                            for k, v in cpv.get('param_results', {}).items()
                        },
                    }

                candidate_list.append(candidate_item)

            output['候选方案'] = candidate_list

        primary_score = recommendation.get('rank_score', 0)
        primary_method_valid = recommendation.get('method_validation', {}).get('valid', False)
        pv = recommendation.get('parameter_validation', {})
        param_all_valid = pv.get('all_valid', False)

        if primary_method_valid and param_all_valid:
            confidence = 'high'
        elif primary_method_valid or param_all_valid:
            confidence = 'medium'
        else:
            confidence = 'low'

        output['置信度'] = {
            'level': confidence,
            'rank_score': primary_score,
            'method_valid': primary_method_valid,
            'parameters_valid': param_all_valid,
        }

        return output
