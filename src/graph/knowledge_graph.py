import contextlib
import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone

from src.config.settings import (
    EMBEDDING_DIM,
    GLM_API_KEY,
    GLM_BASE_URL,
    GLM_EMBEDDING_MODEL,
    GLM_MODEL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)

logger = logging.getLogger(__name__)

try:
    from graphiti_core import Graphiti
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

    from src.graph.glm_graphiti_client import GLMCompatibleClient

    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    logger.warning('graphiti-core未安装，知识图谱功能不可用')


class CADKnowledgeGraph:
    GROUP_ID = 'cad_process_knowledge'

    ENTITY_DEDUPE_SIMILARITY_THRESHOLD = 0.85
    RERANK_WEIGHT_KEYWORD = 0.3
    RERANK_WEIGHT_SEMANTIC = 0.5
    RERANK_WEIGHT_GRAPH = 0.2

    def __init__(self):
        self.graphiti: Graphiti | None = None
        self._connected = False
        self._neo4j_driver = None

    @staticmethod
    def normalize_entity_name(name: str) -> str:
        if not name:
            return ''
        name = unicodedata.normalize('NFKC', name)
        name = name.strip()
        name = name.lower()
        name = re.sub(r'[\s\u3000]+', '_', name)
        name = re.sub(r'[^\w\u4e00-\u9fff]', '', name)
        return name

    def _get_neo4j_driver(self):
        if self._neo4j_driver is not None:
            return self._neo4j_driver
        try:
            from neo4j import GraphDatabase

            self._neo4j_driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            return self._neo4j_driver
        except Exception as e:
            logger.warning(f'Neo4j驱动创建失败: {e}')
            return None

    async def initialize(self, timeout: float = 10.0) -> bool:
        if not GRAPHITI_AVAILABLE:
            logger.warning('graphiti-core不可用，跳过知识图谱初始化')
            return False

        logger.info('正在初始化知识图谱连接...')

        try:
            import asyncio

            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, driver.verify_connectivity),
                    timeout=timeout,
                )
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f'Neo4j连接超时或失败: {e}')
                driver.close()
                return False
            finally:
                driver.close()

            os.environ['OPENAI_API_KEY'] = GLM_API_KEY
            os.environ['OPENAI_BASE_URL'] = GLM_BASE_URL

            llm_config = LLMConfig(
                api_key=GLM_API_KEY,
                model=GLM_MODEL,
                base_url=GLM_BASE_URL,
                small_model=GLM_MODEL,
            )
            llm_client = GLMCompatibleClient(config=llm_config)

            embedder_config = OpenAIEmbedderConfig(
                api_key=GLM_API_KEY,
                base_url=GLM_BASE_URL,
                embedding_model=GLM_EMBEDDING_MODEL,
                embedding_dim=EMBEDDING_DIM,
            )
            embedder = OpenAIEmbedder(config=embedder_config)

            self.graphiti = Graphiti(
                uri=NEO4J_URI,
                user=NEO4J_USER,
                password=NEO4J_PASSWORD,
                llm_client=llm_client,
                embedder=embedder,
            )

            await self.graphiti.build_indices_and_constraints()

            self._connected = True
            logger.info('知识图谱连接成功（使用GLM API + Neo4j）')
            return True

        except Exception as e:
            logger.warning(f'知识图谱初始化失败: {e}，将仅使用LLM和向量库')
            self.graphiti = None
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.graphiti is not None

    async def _compute_embedding_similarity(self, text_a: str, text_b: str) -> float:
        try:
            import numpy as np
            from openai import OpenAI

            client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
            resp = client.embeddings.create(
                model=GLM_EMBEDDING_MODEL,
                input=[text_a, text_b],
            )
            emb_a = np.array(resp.data[0].embedding)
            emb_b = np.array(resp.data[1].embedding)
            norm_a = np.linalg.norm(emb_a)
            norm_b = np.linalg.norm(emb_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
        except Exception as e:
            logger.warning(f'计算嵌入相似度失败: {e}')
            return 0.0

    async def _extract_entities_from_text(self, text: str) -> dict:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=GLM_API_KEY, base_url=GLM_BASE_URL)
            prompt = (
                '从以下工艺知识文本中抽取实体和关系。'
                '请以JSON格式输出，包含"entities"和"relations"两个字段。\n'
                'entities是一个数组，每个元素包含"name"和"type"字段，type可选值为：'
                'FeatureType, Method, Material, MachineTool, Process, Parameter。\n'
                'relations是一个数组，每个元素包含"source", "relation", "target"字段。\n\n'
                f'文本：{text}\n\n请输出JSON：'
            )
            resp = client.chat.completions.create(
                model=GLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                response_format={'type': 'json_object'},
            )
            content = resp.choices[0].message.content or '{}'
            parsed = json.loads(content)
            entities = parsed.get('entities', [])
            relations = parsed.get('relations', [])
            return {'entities': entities, 'relations': relations}
        except Exception as e:
            logger.warning(f'LLM实体抽取失败: {e}')
            return {'entities': [], 'relations': []}

    async def _deduplicate_entities(self, entities: list[dict]) -> list[dict]:
        if not entities:
            return []

        normalized_map: dict[str, dict] = {}
        for entity in entities:
            norm_name = self.normalize_entity_name(entity.get('name', ''))
            if not norm_name:
                continue
            if norm_name not in normalized_map:
                normalized_map[norm_name] = entity
            else:
                existing = normalized_map[norm_name]
                if len(entity.get('name', '')) > len(existing.get('name', '')):
                    normalized_map[norm_name] = entity

        deduped = list(normalized_map.values())

        if len(deduped) <= 1:
            return deduped

        merge_groups: dict[int, list[dict]] = {}
        assigned: dict[int, int] = {}
        group_counter = 0

        for i, ent_a in enumerate(deduped):
            if i in assigned:
                continue
            group_id = group_counter
            group_counter += 1
            merge_groups[group_id] = [ent_a]
            assigned[i] = group_id

            for j in range(i + 1, len(deduped)):
                if j in assigned:
                    continue
                ent_b = deduped[j]
                name_a = ent_a.get('name', '')
                name_b = ent_b.get('name', '')
                similarity = await self._compute_embedding_similarity(name_a, name_b)
                if similarity >= self.ENTITY_DEDUPE_SIMILARITY_THRESHOLD:
                    merge_groups[group_id].append(ent_b)
                    assigned[j] = group_id

        result = []
        for group in merge_groups.values():
            representative = max(group, key=lambda e: len(e.get('name', '')))
            result.append(representative)

        return result

    async def _deduplicate_relations(
        self, relations: list[dict], entities: list[dict]
    ) -> list[dict]:
        entity_names = {self.normalize_entity_name(e.get('name', '')) for e in entities}
        seen: set[str] = set()
        result = []
        for rel in relations:
            src_norm = self.normalize_entity_name(rel.get('source', ''))
            tgt_norm = self.normalize_entity_name(rel.get('target', ''))
            relation = rel.get('relation', '')
            if src_norm not in entity_names or tgt_norm not in entity_names:
                continue
            key = f'{src_norm}|{relation}|{tgt_norm}'
            if key in seen:
                continue
            seen.add(key)
            result.append(rel)
        return result

    async def extract_and_deduplicate_entities(self, text: str) -> dict:
        extraction = await self._extract_entities_from_text(text)
        raw_entities = extraction.get('entities', [])
        raw_relations = extraction.get('relations', [])

        deduped_entities = await self._deduplicate_entities(raw_entities)
        deduped_relations = await self._deduplicate_relations(raw_relations, deduped_entities)

        logger.info(
            f'实体抽取与去重: 原始{len(raw_entities)}个实体/{len(raw_relations)}条关系 -> '
            f'去重后{len(deduped_entities)}个实体/{len(deduped_relations)}条关系'
        )
        return {'entities': deduped_entities, 'relations': deduped_relations}

    async def add_knowledge_triples(
        self, triples: list[tuple[str, str, str]], label: str = 'Entity'
    ) -> int:
        if not triples:
            return 0

        driver = self._get_neo4j_driver()
        if driver is None:
            return 0

        success = 0
        with driver.session() as session:
            for subject, relation, obj in triples:
                try:
                    norm_subj = self.normalize_entity_name(subject)
                    norm_obj = self.normalize_entity_name(obj)
                    if not norm_subj or not norm_obj:
                        continue

                    session.run(
                        f'MERGE (s:{label} {{name: $subj, normalizedName: $norm_subj}}) '
                        f'MERGE (o:{label} {{name: $obj, normalizedName: $norm_obj}}) '
                        f'MERGE (s)-[r:{relation}]->(o) '
                        'ON CREATE SET r.created_at = datetime(), r.group_id = $gid '
                        'ON MATCH SET r.updated_at = datetime()',
                        subj=subject,
                        norm_subj=norm_subj,
                        obj=obj,
                        norm_obj=norm_obj,
                        gid=self.GROUP_ID,
                    )
                    success += 1
                except Exception as e:
                    logger.warning(f'写入三元组失败 ({subject}-{relation}->{obj}): {e}')

        logger.info(f'知识三元组写入: {success}/{len(triples)} 条成功')
        return success

    async def add_process_episode(
        self,
        name: str,
        content: str | dict,
        description: str = '工艺知识',
        reference_time: datetime | None = None,
    ):
        if not self.is_connected:
            logger.warning('知识图谱未连接，跳过添加知识片段')
            return

        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        try:
            extraction = await self.extract_and_deduplicate_entities(content)
            deduped_entities = extraction.get('entities', [])
            deduped_relations = extraction.get('relations', [])

            result = await self.graphiti.add_episode(
                name=name,
                episode_body=content,
                source=EpisodeType.text,
                source_description=description,
                reference_time=reference_time,
                group_id=self.GROUP_ID,
            )
            logger.info(
                f'已添加知识片段: {name}, '
                f'提取实体{len(result.nodes)}个, 关系{len(result.edges)}条, '
                f'去重后实体{len(deduped_entities)}个, 关系{len(deduped_relations)}条'
            )
            return result
        except Exception as e:
            logger.warning(f'Graphiti添加知识片段失败，降级为直接Neo4j写入: {e}')
            return await self._add_episode_direct(name, content, reference_time)

    async def _add_episode_direct(
        self,
        name: str,
        content: str,
        reference_time: datetime,
    ):
        try:
            driver = self._get_neo4j_driver()
            if driver is None:
                from neo4j import GraphDatabase

                driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

            with driver.session() as session:
                session.run(
                    'MERGE (e:Episode {name: $name}) '
                    'ON CREATE SET e.content = $content, '
                    'e.reference_time = $ref_time, e.group_id = $group_id, '
                    'e.created_at = datetime(), e.normalizedName = $norm_name '
                    'ON MATCH SET e.content = $content, '
                    'e.reference_time = $ref_time, e.updated_at = datetime()',
                    name=name,
                    content=content,
                    ref_time=reference_time.isoformat(),
                    group_id=self.GROUP_ID,
                    norm_name=self.normalize_entity_name(name),
                )

                try:
                    data = json.loads(content)
                    feature_type = data.get('特征类型', 'unknown')
                    method = data.get('加工方法', '')
                    route = data.get('工艺路线', '')
                    material = data.get('材料', '')
                    machine = data.get('机床', '')

                    norm_ft = self.normalize_entity_name(feature_type)
                    norm_method = self.normalize_entity_name(method)
                    norm_material = self.normalize_entity_name(material)
                    norm_machine = self.normalize_entity_name(machine)

                    session.run(
                        'MERGE (ft:FeatureType {name: $ft}) '
                        'ON CREATE SET ft.normalizedName = $norm_ft, ft.created_at = datetime() '
                        'MERGE (m:Method {name: $method}) '
                        'ON CREATE SET m.normalizedName = $norm_method, m.created_at = datetime() '
                        'MERGE (mt:Material {name: $material}) '
                        'ON CREATE SET mt.normalizedName = $norm_material, mt.created_at = datetime() '
                        'MERGE (mc:MachineTool {name: $machine}) '
                        'ON CREATE SET mc.normalizedName = $norm_machine, mc.created_at = datetime() '
                        'MERGE (ft)-[r1:USES_METHOD]->(m) '
                        'ON CREATE SET r1.created_at = datetime() '
                        'MERGE (m)-[r2:SUITABLE_FOR]->(ft) '
                        'ON CREATE SET r2.created_at = datetime() '
                        'MERGE (m)-[r3:USES_MATERIAL]->(mt) '
                        'ON CREATE SET r3.created_at = datetime() '
                        'MERGE (m)-[r4:USES_MACHINE]->(mc) '
                        'ON CREATE SET r4.created_at = datetime() '
                        'MERGE (e:Episode {name: $name}) '
                        'MERGE (e)-[r5:DESCRIBES]->(ft) '
                        'ON CREATE SET r5.created_at = datetime()',
                        ft=feature_type,
                        norm_ft=norm_ft,
                        method=method,
                        norm_method=norm_method,
                        material=material,
                        norm_material=norm_material,
                        machine=machine,
                        norm_machine=norm_machine,
                        name=name,
                    )

                    if route:
                        norm_route = self.normalize_entity_name(route)
                        session.run(
                            'MERGE (r:ProcessRoute {name: $route}) '
                            'ON CREATE SET r.normalizedName = $norm_route, r.created_at = datetime() '
                            'MERGE (ft:FeatureType {name: $ft}) '
                            'MERGE (ft)-[:HAS_ROUTE]->(r)',
                            route=route,
                            norm_route=norm_route,
                            ft=feature_type,
                        )
                except (json.JSONDecodeError, Exception):
                    pass

            logger.info(f'已通过Neo4j直接写入知识片段: {name}')
            return type('Result', (), {'nodes': [], 'edges': []})()
        except Exception as e2:
            logger.warning(f'Neo4j直接写入也失败: {e2}')
            return None

    async def add_process_case(self, case: dict):
        feature = case.get('feature', {})
        if isinstance(feature, dict):
            feature_type = feature.get('feature_type', 'unknown')
        else:
            feature_type = str(feature)
        name = f"工艺案例_{feature_type}_{case.get('id', '')}"
        content = {
            '特征类型': feature_type,
            '加工方法': case.get('machining_method', ''),
            '工艺路线': case.get('process_route', ''),
            '工艺参数': case.get('parameters', {}),
            '材料': case.get('material', ''),
            '机床': case.get('machine_tool', ''),
        }
        return await self.add_process_episode(
            name=name, content=content, description='工艺案例'
        )

    async def add_process_cases_bulk(self, cases: list[dict]):
        if not self.is_connected:
            logger.warning('知识图谱未连接，跳过批量添加')
            return

        success = 0
        for case in cases:
            try:
                result = await self.add_process_case(case)
                if result is not None:
                    success += 1
            except Exception as e:
                logger.warning(f'批量添加案例失败: {e}')

        logger.info(f'批量添加完成: {success}/{len(cases)} 条成功写入知识图谱')
        return success

    async def search_knowledge(self, query: str, limit: int = 5) -> list[dict]:
        if not self.is_connected:
            return []

        try:
            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = limit
            results = await self.graphiti.search_(
                query=query, config=config, group_ids=[self.GROUP_ID]
            )

            knowledge_items = []
            for edge in results.edges:
                knowledge_items.append({
                    'type': 'edge',
                    'fact': edge.fact,
                    'name': edge.name,
                })
            for node in results.nodes:
                knowledge_items.append({
                    'type': 'node',
                    'name': node.name,
                    'summary': node.summary,
                })
            for ep in results.episodes:
                knowledge_items.append({
                    'type': 'episode',
                    'name': ep.name,
                    'content': ep.content,
                })

            logger.info(
                f'知识图谱检索: {len(results.nodes)}个节点, '
                f'{len(results.edges)}条边, {len(results.episodes)}个episode'
            )
            return knowledge_items[:limit]
        except Exception as e:
            logger.warning(f'知识图谱检索失败: {e}')
            return []

    async def search_with_reranking(
        self, query: str, limit: int = 5, initial_multiplier: int = 3
    ) -> list[dict]:
        initial_limit = limit * initial_multiplier
        items = await self.search_knowledge(query, limit=initial_limit)
        if not items:
            return []

        scored_items = []
        query_lower = query.lower()
        query_keywords = set(re.findall(r'[\u4e00-\u9fff\w]+', query_lower))

        for item in items:
            keyword_score = 0.0
            text_content = ''
            if item.get('type') == 'edge':
                text_content = f"{item.get('fact', '')} {item.get('name', '')}"
            elif item.get('type') == 'node':
                text_content = f"{item.get('name', '')} {item.get('summary', '')}"
            elif item.get('type') == 'episode':
                text_content = f"{item.get('name', '')} {item.get('content', '')}"

            text_lower = text_content.lower()
            text_keywords = set(re.findall(r'[\u4e00-\u9fff\w]+', text_lower))
            overlap = query_keywords & text_keywords
            if query_keywords:
                keyword_score = len(overlap) / len(query_keywords)

            semantic_score = 0.0
            if text_content:
                try:
                    semantic_score = await self._compute_embedding_similarity(query, text_content)
                except Exception:
                    semantic_score = 0.0

            graph_score = 0.0
            if item.get('type') == 'edge':
                graph_score = 1.0
            elif item.get('type') == 'node':
                graph_score = 0.7
            elif item.get('type') == 'episode':
                graph_score = 0.4

            final_score = (
                self.RERANK_WEIGHT_KEYWORD * keyword_score
                + self.RERANK_WEIGHT_SEMANTIC * semantic_score
                + self.RERANK_WEIGHT_GRAPH * graph_score
            )
            scored_items.append((final_score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        reranked = [item for _, item in scored_items[:limit]]

        logger.info(
            f'检索重排序: 查询"{query}", 初始{len(items)}条 -> 重排序后{len(reranked)}条'
        )
        return reranked

    async def search_edges(self, query: str, limit: int = 5) -> list:
        if not self.is_connected:
            return []

        try:
            edges = await self.graphiti.search(
                query=query, num_results=limit, group_ids=[self.GROUP_ID]
            )
            return edges[:limit]
        except Exception as e:
            logger.warning(f'知识图谱边检索失败: {e}')
            return []

    async def search_process_facts(self, query: str, limit: int = 5) -> list[str]:
        if not self.is_connected:
            return []

        facts = []
        try:
            edges = await self.graphiti.search(
                query=query, num_results=limit, group_ids=[self.GROUP_ID]
            )
            facts = [edge.fact for edge in edges]
            logger.info(f'知识图谱事实检索: 查询"{query}", 返回{len(facts)}条事实')
        except Exception as e:
            logger.warning(f'知识图谱事实检索失败，尝试直接Neo4j检索: {e}')

        if not facts:
            facts = await self._search_facts_direct(query, limit)

        return facts

    async def _search_facts_direct(self, query: str, limit: int = 5) -> list[str]:
        try:
            driver = self._get_neo4j_driver()
            if driver is None:
                from neo4j import GraphDatabase

                driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

            with driver.session() as session:
                result = session.run(
                    'MATCH (e:Episode) WHERE e.group_id = $gid '
                    'RETURN e.name AS name, e.content AS content '
                    'LIMIT $limit',
                    gid=self.GROUP_ID,
                    limit=limit,
                )
                facts = []
                for record in result:
                    content = record.get('content', '')
                    name = record.get('name', '')
                    if content:
                        facts.append(f'{name}: {content[:200]}')
            logger.info(f'Neo4j直接检索: 查询"{query}", 返回{len(facts)}条事实')
            return facts[:limit]
        except Exception as e:
            logger.warning(f'Neo4j直接检索也失败: {e}')
            return []

    async def get_graph_statistics(self) -> dict:
        stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'node_labels': {},
            'edge_types': {},
        }

        try:
            driver = self._get_neo4j_driver()
            if driver is None:
                from neo4j import GraphDatabase

                driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

            with driver.session() as session:
                node_count_result = session.run(
                    'MATCH (n) RETURN count(n) AS cnt'
                )
                record = node_count_result.single()
                stats['total_nodes'] = record['cnt'] if record else 0

                edge_count_result = session.run(
                    'MATCH ()-[r]->() RETURN count(r) AS cnt'
                )
                record = edge_count_result.single()
                stats['total_edges'] = record['cnt'] if record else 0

                label_result = session.run(
                    'MATCH (n) RETURN labels(n) AS lbls, count(n) AS cnt'
                )
                for record in label_result:
                    lbls = record.get('lbls', [])
                    cnt = record.get('cnt', 0)
                    for lbl in lbls:
                        stats['node_labels'][lbl] = stats['node_labels'].get(lbl, 0) + cnt

                rel_result = session.run(
                    'MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt'
                )
                for record in rel_result:
                    rel_type = record.get('rel_type', '')
                    cnt = record.get('cnt', 0)
                    stats['edge_types'][rel_type] = cnt

            logger.info(
                f'图谱统计: {stats["total_nodes"]}个节点, '
                f'{stats["total_edges"]}条边, '
                f'标签类型{len(stats["node_labels"])}, '
                f'关系类型{len(stats["edge_types"])}'
            )
        except Exception as e:
            logger.warning(f'获取图谱统计信息失败: {e}')

        return stats

    async def clear_all_data(self) -> bool:
        try:
            driver = self._get_neo4j_driver()
            if driver is None:
                from neo4j import GraphDatabase

                driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

            with driver.session() as session:
                session.run('MATCH (n) DETACH DELETE n')
                result = session.run('MATCH (n) RETURN count(n) AS cnt')
                record = result.single()
                remaining = record['cnt'] if record else -1
            if remaining == 0:
                logger.info('Neo4j数据库已清空（所有节点和关系已删除）')
                return True
            logger.warning(f'Neo4j清空后仍有{remaining}个节点')
            return False
        except Exception as e:
            logger.warning(f'清空Neo4j数据失败: {e}')
            return False

    async def close(self):
        if self.graphiti:
            with contextlib.suppress(Exception):
                await self.graphiti.close()
        if self._neo4j_driver:
            with contextlib.suppress(Exception):
                self._neo4j_driver.close()
            self._neo4j_driver = None
        self._connected = False
        logger.info('知识图谱连接已关闭')
