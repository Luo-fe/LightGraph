import contextlib
import json
import logging
import os
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

    def __init__(self):
        self.graphiti: Graphiti | None = None
        self._connected = False

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
                f'提取实体{len(result.nodes)}个, 关系{len(result.edges)}条'
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
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                session.run(
                    'CREATE (e:Episode {name: $name, content: $content, '
                    'reference_time: $ref_time, group_id: $group_id, '
                    'created_at: datetime()})',
                    name=name,
                    content=content,
                    ref_time=reference_time.isoformat(),
                    group_id=self.GROUP_ID,
                )

                try:
                    data = json.loads(content)
                    feature_type = data.get('特征类型', 'unknown')
                    method = data.get('加工方法', '')
                    route = data.get('工艺路线', '')
                    material = data.get('材料', '')
                    machine = data.get('机床', '')

                    session.run(
                        'MERGE (ft:FeatureType {name: $ft}) '
                        'MERGE (m:Method {name: $method}) '
                        'MERGE (mt:Material {name: $material}) '
                        'MERGE (mc:MachineTool {name: $machine}) '
                        'MERGE (ft)-[:USES_METHOD]->(m) '
                        'MERGE (m)-[:SUITABLE_FOR]->(ft) '
                        'MERGE (m)-[:USES_MATERIAL]->(mt) '
                        'MERGE (m)-[:USES_MACHINE]->(mc) '
                        'MERGE (e:Episode {name: $name}) '
                        'MERGE (e)-[:DESCRIBES]->(ft)',
                        ft=feature_type,
                        method=method,
                        material=material,
                        machine=machine,
                        name=name,
                    )
                except (json.JSONDecodeError, Exception):
                    pass

            driver.close()
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
            driver.close()
            logger.info(f'Neo4j直接检索: 查询"{query}", 返回{len(facts)}条事实')
            return facts[:limit]
        except Exception as e:
            logger.warning(f'Neo4j直接检索也失败: {e}')
            return []

    async def clear_all_data(self) -> bool:
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as session:
                session.run('MATCH (n) DETACH DELETE n')
                result = session.run('MATCH (n) RETURN count(n) AS cnt')
                record = result.single()
                remaining = record['cnt'] if record else -1
            driver.close()
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
        self._connected = False
        logger.info('知识图谱连接已关闭')
