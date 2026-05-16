# LightGraph — 基于知识图谱的CAD工艺路线推荐系统

> 输入CAD零件模型 → 自动识别加工特征 → 知识库检索（向量库 + Neo4j知识图谱） → 推荐加工方法和工艺参数 → 输出结构化JSON

## 系统架构

```
CAD模型(.stp) ──→ STEP解析 ──→ 特征识别(LLM) ──→ 知识库检索 ──→ 参数生成(LLM) ──→ JSON输出
                                                    │
                                          ┌─────────┴─────────┐
                                          │                   │
                                     Faiss向量库         Neo4j知识图谱
                                    (相似案例检索)     (实体/关系/事实检索)
```

**核心特色**：三路检索融合 — 向量库（相似案例）+ 知识图谱（关联知识）+ LLM（生成推理），比纯LLM推荐更可靠。

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
```

> **获取GLM API Key**：访问 https://open.bigmodel.cn 注册账号，在API密钥页面创建。

### 3. 启动Neo4j（可选）

**Docker方式（推荐）**：

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5.26.0
```

**Neo4j Desktop方式**：下载安装 https://neo4j.com/download/ ，创建数据库并启动。

> Neo4j不是必须的。如果未启动，系统自动降级为LLM+向量库模式，仍可正常工作。

### 4. 运行

```bash
python -m src.pipeline.run
```

该命令自动执行5步流程：

| 步骤 | 内容 | 说明 |
|------|------|------|
| 1 | 测试GLM API连接 | 验证API密钥 |
| 2 | 填充知识库 | 示例数据 → 清洗 → 结构化 → 向量库 + Neo4j知识图谱 |
| 3 | 原型验证 | 2个典型零件（通孔 + 四边形腔） |
| 4 | 示例端到端 | 四边形腔特征 → 完整推荐流程 |
| 5 | TMCAD端到端 | 5类零件各取1个运行完整流程 |

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
-- 查看完整知识图谱
MATCH (n)-[r]->(m) WHERE n.group_id = 'cad_process_knowledge' RETURN n, r, m

-- 查看所有实体
MATCH (e:Entity) WHERE e.group_id = 'cad_process_knowledge' RETURN e.name, e.summary

-- 查看实体间关系
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity) WHERE a.group_id = 'cad_process_knowledge' RETURN a.name, r.fact, b.name
```

## 项目结构

```
LightGraph/
├── src/
│   ├── config/
│   │   └── settings.py              # 全局配置（特征类型、参数范围、路径）
│   ├── data/                        # 工作1：数据处理管线
│   │   ├── collector.py             #   数据收集（从目录读取JSON/CSV）
│   │   ├── cleaner.py               #   数据清洗（去标签/脱敏/去重）
│   │   ├── structurer.py            #   数据结构化（Pydantic模型 + 训练数据生成）
│   │   └── vector_store.py          #   Faiss向量库（GLM Embedding + 内积检索）
│   ├── feature/                     # 工作2：特征识别
│   │   ├── step_parser.py           #   STEP文件解析（正则提取几何信息）
│   │   └── extractor.py             #   加工特征提取（LLM驱动）
│   ├── recommend/                   # 工作2：工艺参数推荐
│   │   └── recommender.py           #   推荐引擎（向量库+知识图谱+LLM三路融合）
│   ├── graph/                       # 知识图谱基础设施
│   │   ├── knowledge_graph.py       #   Graphiti知识图谱连接（Neo4j后端）
│   │   ├── glm_graphiti_client.py   #   GLM兼容的Graphiti LLM客户端
│   │   ├── glm_client.py            #   GLM API客户端（聊天/JSON/实体抽取/工艺推荐）
│   │   └── glm_embedder.py          #   GLM Embedding客户端（文本向量化）
│   ├── pipeline/                    # 工作4：端到端流程编排
│   │   └── run.py                   #   主入口：5步流程串联
│   └── validation/                  # 工作3：原型验证
│       └── __init__.py              #   MDPM/MDMT指标计算
├── data/                            # 运行时数据（gitignore）
│   ├── raw/                         #   原始数据
│   ├── processed/                   #   清洗后数据
│   ├── annotated/                   #   结构化+训练数据
│   ├── faiss_index                  #   Faiss向量索引
│   └── output/                      #   最终JSON输出结果
├── tests/                           # 单元测试
├── .env.example                     # 环境变量模板
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
| thread | 螺纹 | 车-精车 |
| gear_tooth | 齿形 | 粗铣-精铣 |

## 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| LLM | 智谱AI GLM-4-Flash | 特征识别、工艺推荐、实体抽取 |
| Embedding | GLM embedding-3 (2048维) | 文本向量化 |
| 向量数据库 | Faiss (IndexFlatIP) | 相似案例检索 |
| 图数据库 | Neo4j + Graphiti | 工艺知识图谱存储与检索 |
| 数据模型 | Pydantic v2 | 结构化数据验证 |
| CAD解析 | 正则表达式 | STEP文件几何信息提取 |

## 常见问题

**GLM API连接失败**：检查 `.env` 中 `GLM_API_KEY` 是否正确，确认网络可访问 `https://open.bigmodel.cn`。

**Neo4j连接失败**：确认Neo4j已启动，检查端口和密码。Neo4j不是必须的，连接失败时系统自动降级。

**向量库为空**：首次运行需先执行完整流程 `python -m src.pipeline.run`，会自动填充向量库。

**在Neo4j Desktop中看不到数据**：确认你查看的数据库端口与 `.env` 中 `NEO4J_URI` 一致。也可直接在浏览器访问 http://localhost:7474 连接。

## 运行测试

```bash
pytest tests/ -v
```

## License

MIT
