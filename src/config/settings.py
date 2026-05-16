import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

GLM_API_KEY = os.getenv('GLM_API_KEY', '')
GLM_BASE_URL = os.getenv('GLM_BASE_URL', 'https://open.bigmodel.cn/api/paas/v4')
GLM_MODEL = os.getenv('GLM_MODEL', 'glm-4-flash')
GLM_EMBEDDING_MODEL = os.getenv('GLM_EMBEDDING_MODEL', 'embedding-3')

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '')

FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', str(BASE_DIR / 'data' / 'faiss_index'))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '2048'))

DATA_DIR = Path(os.getenv('DATA_DIR', str(BASE_DIR / 'data')))
RAW_DATA_DIR = Path(os.getenv('RAW_DATA_DIR', str(DATA_DIR / 'raw')))
PROCESSED_DATA_DIR = Path(os.getenv('PROCESSED_DATA_DIR', str(DATA_DIR / 'processed')))
ANNOTATED_DATA_DIR = Path(os.getenv('ANNOTATED_DATA_DIR', str(DATA_DIR / 'annotated')))

TMCAD_DATASET_PATH = Path(os.getenv('TMCAD_DATASET_PATH', str(BASE_DIR / 'TMCAD_dataset_v2' / 'mechcad' / 'mechcad')))

PART_CATEGORY_MAP = {
    'bolt': '螺栓',
    'gear': '齿轮',
    'nut': '螺母',
    'shaft': '轴',
    'flange': '法兰',
}

MACHINING_FEATURES = [
    'rectangular_pocket',
    'square_slot',
    'circular_boss',
    'square_boss',
    'through_hole',
    'blind_hole',
    'outer_circle',
    'conical_surface',
    'cylindrical_surface',
    'circular_curve',
    'thread',
    'gear_tooth',
]

FEATURE_NAME_MAP = {
    'rectangular_pocket': '四边形腔',
    'square_slot': '方形槽',
    'circular_boss': '圆形凸台',
    'square_boss': '方形凸台',
    'through_hole': '通孔',
    'blind_hole': '盲孔',
    'outer_circle': '外圆',
    'conical_surface': '圆锥面',
    'cylindrical_surface': '圆柱面',
    'circular_curve': '圆曲线',
    'thread': '螺纹',
    'gear_tooth': '齿形',
}

MACHINING_METHODS = {
    'rectangular_pocket': ['粗铣-半精铣', '粗铣-精铣'],
    'square_slot': ['粗铣-半精铣', '粗铣-精铣'],
    'circular_boss': ['粗车-半精车', '粗车-精车'],
    'square_boss': ['粗铣-半精铣', '粗铣-精铣'],
    'through_hole': ['钻-扩-铰', '钻-镗'],
    'blind_hole': ['钻-扩-铰', '钻-镗'],
    'outer_circle': ['粗车-半精车', '粗车-精车'],
    'conical_surface': ['粗车-精车', '粗车-半精车-精车'],
    'cylindrical_surface': ['粗车-精车', '粗车-半精车-精车'],
    'circular_curve': ['粗车-精车', '粗铣-精铣'],
    'thread': ['车螺纹', '铣螺纹'],
    'gear_tooth': ['滚齿-剃齿', '铣齿-磨齿'],
}

PROCESS_PARAMETERS = [
    'spindle_speed',
    'feed_rate',
    'tool_diameter',
    'cutting_depth',
    'cutting_width',
]

PARAMETER_NAME_MAP = {
    'spindle_speed': '主轴转速',
    'feed_rate': '进给速度',
    'tool_diameter': '刀具直径',
    'cutting_depth': '切削深度',
    'cutting_width': '切削宽度',
}

FEED_RATE_RANGE = {
    'turning': (50, 500),
    'milling': (50, 800),
    'drilling': (30, 300),
    'reaming': (20, 200),
}

SPINDLE_SPEED_RANGE = {
    'turning': (200, 3000),
    'milling': (500, 8000),
    'drilling': (200, 3000),
    'reaming': (100, 2000),
}
