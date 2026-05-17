# LightGraph — 基于知识工程的CAD工艺路线推荐系统

> 输入CAD零件模型 → 自动识别加工特征 → 知识库检索（向量库 + Neo4j知识图谱） → 推荐加工方法和工艺参数 → 输出结构化JSON

## 系统架构

```
CAD模型(.stp)
    ↓
[STEP解析器] ─→ 几何信息(面/曲面/尺寸) + 推断特征 + 置信度 + 材料 + 公差
    ↓
[LLM特征识别] ─→ MachiningFeature列表 (验证+精炼)
    ↓                                    ↓
[知识库检索] ←── Faiss向量库 ──→ BM25+向量混合检索(RRF融合)
    │         ←── Neo4j知识图谱 ──→ 实体/关系/事实检索(重排序)
    ↓
[LLM工艺推荐] ─→ 多候选方案(top-3) + 参数校验 + 方法验证
    ↓
[参数约束] ─→ 范围clamp + 合理性校验 + 修正建议
    ↓
JSON输出 (含知识来源追溯)
```

**核心特色**：三路检索融合 — 向量库（相似案例）+ 知识图谱（关联知识）+ LLM（生成推理），比纯LLM推荐更可靠。

## 大语言模型(LLM)在各步骤中的作用

本系统在以下4个关键步骤中使用大语言模型（智谱AI GLM-4-Flash），每步承担不同角色：

### 步骤1：加工特征识别（LLM作为"特征识别器"）

```
输入: STEP文件解析后的几何摘要文本
输出: MachiningFeature列表（特征类型、尺寸、精度、粗糙度）
```

- **作用**：从STEP解析器提取的几何信息（面数、曲面类型、尺寸估算）中，识别出具体的加工特征
- **Prompt策略**：系统提示定义12种支持的特征类型，要求LLM严格从中选择，并推断合理的尺寸参数
- **v0.2.1增强**：增加特征验证（类型/数值/精度校验）和多轮精炼（低置信度特征再次确认）
- **调用方式**：`extractor.extract_from_text(summary)` → GLM `chat_json`
- **耗时**：3-5秒/零件

### 步骤2：知识图谱实体/关系抽取（LLM作为"知识抽取器"）

```
输入: 工艺案例文本（特征+方法+参数+材料+机床）
输出: 实体列表 + 关系列表（如：通孔-使用->钻-扩-铰）
```

- **作用**：从结构化工艺案例中抽取实体（加工特征、加工方法、刀具、材料、机床）和关系（使用、适用于、加工、包含、对应）
- **Prompt策略**：定义6种实体类型和5种关系类型，要求LLM以JSON格式输出
- **v0.2.1增强**：实体去重（normalizedName归一化 + 语义相似度>0.85聚类合并），参考Graphiti的resolve_extracted_nodes逻辑
- **调用方式**：`knowledge_graph.extract_and_deduplicate_entities(text)` → GLM `chat_json`
- **耗时**：8-15秒/案例（含去重）

### 步骤3：工艺参数推荐（LLM作为"工艺工程师"）

```
输入: 加工特征信息 + 知识库检索上下文（相似案例+知识图谱事实）
输出: 加工方法 + 工艺路线 + 工艺参数（转速/进给/刀具/切深/切宽）
```

- **作用**：基于加工特征和知识库检索结果，生成具体的加工方法、工艺路线和工艺参数
- **Prompt策略**：系统提示定义加工方法选择规则（回转体→车削，孔类→钻削，腔槽→铣削），要求参数为纯数字
- **v0.2.1增强**：多候选方案生成（top-3，尝试不同加工方法）、参数约束校验、评分排序
- **调用方式**：`recommender.recommend_with_full_validation(feature)` → GLM `chat_json`
- **耗时**：5-8秒/特征

### 步骤4：文本向量化（Embedding模型，非生成式LLM）

```
输入: 工艺案例文本
输出: 2048维向量（用于相似度检索）
```

- **作用**：将工艺案例文本转化为向量表示，用于Faiss向量库的相似案例检索
- **模型**：GLM embedding-3（2048维）
- **调用方式**：`glm_embedder.embed(text)` / `embed_batch(texts)`
- **耗时**：0.3-0.5秒/条

### LLM调用统计（15个零件实测）

| 步骤 | 调用次数 | 单次耗时 | 总耗时 | 占比 |
|------|---------|---------|--------|------|
| 特征识别 | ~15次 | 3-5秒 | ~1分钟 | 3% |
| 实体抽取+去重 | ~130次 | 8-15秒 | ~25分钟 | 72% |
| 工艺推荐 | ~47次 | 5-8秒 | ~5分钟 | 14% |
| Embedding | ~130次 | 0.3-0.5秒 | ~1分钟 | 3% |
| **合计** | **~322次** | - | **~32分钟** | **100%** |

> **关键发现**：实体抽取+去重是LLM调用的最大瓶颈（72%），因为每条案例需要2-3次LLM调用。优化方案：改用三元组直写Neo4j（跳过Graphiti的LLM实体抽取），可减少90%的LLM调用。

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/Luo-fe/LightGraph.git
cd LightGraph
pip install -e ".[dev]"
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的配置：

```ini
# 必填：GLM API Key（智谱AI，glm-4-flash免费额度充足）
GLM_API_KEY=你的API密钥

# 可选：Neo4j（不启动则自动降级为LLM+向量库模式）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=你的Neo4j密码

# 可选：TMCAD数据集路径
TMCAD_DATASET_PATH=你的TMCAD数据集路径
```

> **获取GLM API Key**：访问 https://open.bigmodel.cn 注册账号，在API密钥页面创建。

### 3. 启动Neo4j（可选）

**Docker方式（推荐）**：

```bash
docker run -d --name neo4j \
  -p 7474:7687 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5.26.0
```

> Neo4j不是必须的。如果未启动，系统自动降级为LLM+向量库模式，仍可正常工作。

### 4. 运行

```bash
python -m src.pipeline.run
```

该命令自动执行8步流程：

| 步骤 | 内容 | 说明 |
|------|------|------|
| 0 | 清空Neo4j旧数据 | 确保干净状态 |
| 1 | 测试GLM API连接 | 验证API密钥 |
| 2 | 工作1-填充知识库 | 示例数据→清洗评分→结构化标注→三元组→向量库→知识图谱 |
| 3 | 工作1补充-TMCAD构建 | TMCAD数据集→特征识别→向量库→知识图谱 |
| 4 | 工作3-原型验证 | 10个测试用例，计算MDPM/MDMT |
| 5 | 典型零件验证 | 轴类+齿轮两个典型零件验证 |
| 6 | 工作4-端到端 | 单个特征→完整推荐流程 |
| 7 | TMCAD端到端 | 5类零件各取若干运行完整流程 |

## 输出说明

结果自动保存至 `data/output/`，格式如下：

```json
{
  "加工特征": "通孔",
  "特征参数": { "diameter": 10.0, "depth": 25.0, "precision": "IT7", "roughness": 1.6 },
  "加工方法": "钻-扩-铰",
  "加工工艺路线": "1. 钻孔... 2. 扩孔... 3. 铰孔...",
  "加工参数": {
    "主轴转速(rpm)": 3000,
    "进给速度(mm/min)": 50.0,
    "刀具直径(mm)": 9.8,
    "切削深度(mm)": 25.0,
    "切削宽度(mm)": 1.0
  },
  "方法验证": { "valid": true, "recommended": "钻-扩-铰", "valid_methods": ["钻-扩-铰", "钻-镗"] },
  "知识来源": {
    "向量库(相似案例)": ["通孔 → 钻-扩-铰: ..."],
    "知识图谱(Neo4j)": {
      "事实": ["CNC定位直径9.8mm钻头用于在铝合金工件上钻孔。"]
    }
  }
}
```

## 单独使用各模块

### 查询某个零件的工艺路线

```python
import asyncio
from src.pipeline.run import run_end_to_end

result = asyncio.run(run_end_to_end({
    'feature_type': '通孔',
    'diameter': 10, 'depth': 25,
    'precision': 'IT7', 'roughness': 1.6,
}))
```

### 只做特征识别

```python
import asyncio
from src.pipeline.run import run_feature_recognition

features = asyncio.run(run_feature_recognition('该零件包含通孔特征，直径10mm，深度25mm'))
```

### 运行验证

```python
import asyncio
from src.pipeline.run import run_validation

report = asyncio.run(run_validation([
    {
        'feature_type': '通孔', 'diameter': 10, 'depth': 25,
        'precision': 'IT7', 'roughness': 1.6,
        'expected_method': '钻-扩-铰',
        'expected_tool_diameter': 10,
    },
]))
print(f"MDPM={report['mdpm']:.0%}, MDMT={report['mdmt']}")
```

### 在Neo4j Browser中查看知识图谱

启动Neo4j后，在浏览器打开 http://localhost:7474 ，执行：

```cypher
MATCH (n)-[r]->(m) WHERE n.group_id = 'cad_process_knowledge' RETURN n, r, m

MATCH (e:Entity) WHERE e.group_id = 'cad_process_knowledge' RETURN e.name, e.summary

MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.group_id = 'cad_process_knowledge' RETURN a.name, r.fact, b.name
```

## 项目结构

```
LightGraph/
├── src/
│   ├── config/
│   │   └── settings.py              # 全局配置（特征类型、参数范围、路径）
│   ├── data/                        # 工作1：数据处理管线
│   │   ├── collector.py             #   数据收集（JSON/CSV/STEP + 质量校验 + 统计）
│   │   ├── cleaner.py               #   数据清洗（去标签/脱敏/去重/质量评分/字段归一化）
│   │   ├── structurer.py            #   数据结构化（Pydantic模型 + 三元组生成 + 自动标注）
│   │   └── vector_store.py          #   Faiss向量库（GLM Embedding + BM25混合检索 + RRF融合）
│   ├── feature/                     # 工作2：特征识别
│   │   ├── step_parser.py           #   STEP文件解析（几何信息 + 公差 + 粗糙度 + 智能分类 + 材料推断）
│   │   └── extractor.py             #   加工特征提取（LLM驱动 + 验证 + 多轮精炼）
│   ├── recommend/                   # 工作2：工艺参数推荐
│   │   └── recommender.py           #   推荐引擎（多候选方案 + 参数校验 + 评分排序 + 知识来源追溯）
│   ├── graph/                       # 知识图谱基础设施
│   │   ├── knowledge_graph.py       #   知识图谱（实体去重 + 三元组直写 + 检索重排序 + 统计）
│   │   ├── glm_graphiti_client.py   #   GLM兼容的Graphiti LLM客户端
│   │   ├── glm_client.py            #   GLM API客户端（聊天/JSON/实体抽取/工艺推荐）
│   │   └── glm_embedder.py          #   GLM Embedding客户端（文本向量化）
│   ├── pipeline/                    # 工作4：端到端流程编排
│   │   └── run.py                   #   主入口（PipelineState追踪 + 中间结果保存 + 错误恢复）
│   └── validation/                  # 工作3：原型验证
│       └── __init__.py              #   MDPM/MDMT/MDPP指标 + 典型零件验证 + 完整报告
├── data/                            # 运行时数据（gitignore）
│   ├── raw/                         #   原始数据
│   ├── processed/                   #   清洗后数据
│   ├── annotated/                   #   结构化+训练数据+知识三元组
│   ├── faiss_index                  #   Faiss向量索引
│   └── output/                      #   最终JSON输出结果
├── tests/                           # 单元测试（32个用例）
├── .env.example                     # 环境变量模板
├── LightCAD日志.md                   # 项目运行日志（含实测数据）
└── pyproject.toml                   # 项目依赖配置
```

## 支持的加工特征

| 特征类型 | 中文名 | 典型加工方法 |
|----------|--------|-------------|
| rectangular_pocket | 四边形腔 | 粗铣-半精铣 / 粗铣-精铣 |
| square_slot | 方形槽 | 粗铣-精铣 |
| through_hole | 通孔 | 钻-扩-铰 / 钻-镗 |
| blind_hole | 盲孔 | 钻-扩-铰 / 钻-镗 |
| outer_circle | 外圆 | 粗车-精车 / 粗车-半精车 |
| conical_surface | 圆锥面 | 粗车-精车 |
| cylindrical_surface | 圆柱面 | 粗车-精车 |
| circular_curve | 圆曲线 | 粗车-精车 / 粗铣-精铣 |
| thread | 螺纹 | 车螺纹 / 铣螺纹 |
| gear_tooth | 齿形 | 滚齿-剃齿 / 铣齿-磨齿 |

## 评价指标

| 指标 | 全称 | 说明 |
|------|------|------|
| MDPM | Machining Method Prediction Match | 加工方法推荐准确率 |
| MDMT | Machining Tool Match | 刀具直径匹配度(误差<20%) |
| MDPP | Machining Parameter Prediction Precision | 工艺参数合理性 |
| FRR | Feature Recognition Rate | 特征识别成功率 |
| KGR | Knowledge Graph Recall | 知识图谱检索命中率 |

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| LLM | 智谱AI GLM-4-Flash | 特征识别、工艺推荐、实体抽取 |
| Embedding | GLM embedding-3 (2048维) | 文本向量化 |
| 向量数据库 | Faiss (IndexFlatIP) | 相似案例检索 |
| 文本检索 | BM25 (jieba分词) | 全文检索 + RRF融合 |
| 图数据库 | Neo4j + Graphiti | 工艺知识图谱存储与检索 |
| 数据模型 | Pydantic v2 | 结构化数据验证 |
| CAD解析 | 正则表达式 | STEP文件几何信息提取 |

## 常见问题

**GLM API连接失败**：检查 `.env` 中 `GLM_API_KEY` 是否正确，确认网络可访问 `https://open.bigmodel.cn`。

**Neo4j连接失败**：确认Neo4j已启动，检查端口和密码。Neo4j不是必须的，连接失败时系统自动降级。

**向量库为空**：首次运行需先执行完整流程 `python -m src.pipeline.run`，会自动填充向量库。

**TMCAD数据集路径**：在 `.env` 中设置 `TMCAD_DATASET_PATH` 指向本地数据集路径。数据集不包含在仓库中。

## 运行测试

```bash
pytest tests/ -v
```

## License

MIT
