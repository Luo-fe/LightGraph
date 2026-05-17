import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src.config.settings import (
    ANNOTATED_DATA_DIR,
    FEATURE_NAME_MAP,
    FEED_RATE_RANGE,
    MACHINING_METHODS,
    SPINDLE_SPEED_RANGE,
)

logger = logging.getLogger(__name__)


class MachiningFeature(BaseModel):
    feature_type: str = Field(description='加工特征类型')
    length: float | None = Field(default=None, description='长度(mm)')
    width: float | None = Field(default=None, description='宽度(mm)')
    diameter: float | None = Field(default=None, description='直径(mm)')
    depth: float | None = Field(default=None, description='深度(mm)')
    precision: str | None = Field(default=None, description='精度等级')
    roughness: float | None = Field(default=None, description='粗糙度Ra(μm)')


class ProcessParameter(BaseModel):
    spindle_speed: float | None = Field(default=None, description='主轴转速(rpm)')
    feed_rate: float | None = Field(default=None, description='进给速度(mm/min)')
    tool_diameter: float | None = Field(default=None, description='刀具直径(mm)')
    cutting_depth: float | None = Field(default=None, description='切削深度(mm)')
    cutting_width: float | None = Field(default=None, description='切削宽度(mm)')


class ProcessCase(BaseModel):
    id: str = Field(description='案例唯一标识')
    feature: MachiningFeature = Field(description='加工特征')
    machining_method: str = Field(description='加工方法')
    process_route: str = Field(description='加工工艺路线')
    parameters: ProcessParameter = Field(default_factory=ProcessParameter, description='工艺参数')
    material: str | None = Field(default=None, description='材料')
    machine_tool: str | None = Field(default=None, description='机床')
    tool: str | None = Field(default=None, description='刀具')


class TrainingDataItem(BaseModel):
    instruction: str = Field(description='指令')
    input_text: str = Field(description='输入')
    output: str = Field(description='输出')


class KnowledgeTriple(BaseModel):
    source_entity: str = Field(description='源实体')
    relation: str = Field(description='关系')
    target_entity: str = Field(description='目标实体')
    attributes: dict = Field(default_factory=dict, description='属性')


class DataStructurer:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or ANNOTATED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_feature_type(self, feature_type: str) -> str:
        for eng, chn in FEATURE_NAME_MAP.items():
            if chn in feature_type or eng in feature_type.lower():
                return eng
        return feature_type

    def _determine_process_type(self, method: str) -> str:
        if any(k in method for k in ['铣', '腔', '槽']):
            return 'milling'
        if any(k in method for k in ['钻', '扩', '铰']):
            return 'drilling'
        if any(k in method for k in ['车', '圆']):
            return 'turning'
        if any(k in method for k in ['滚', '剃', '磨齿']):
            return 'milling'
        return 'turning'

    def structure_process_case(self, raw_data: dict) -> ProcessCase | None:
        try:
            raw_feature_type = raw_data.get('feature_type', raw_data.get('特征类型', ''))
            feature_type = self._normalize_feature_type(raw_feature_type)
            feature = MachiningFeature(
                feature_type=feature_type,
                length=raw_data.get('length', raw_data.get('长度')),
                width=raw_data.get('width', raw_data.get('宽度')),
                diameter=raw_data.get('diameter', raw_data.get('直径')),
                depth=raw_data.get('depth', raw_data.get('深度')),
                precision=raw_data.get('precision', raw_data.get('精度')),
                roughness=raw_data.get('roughness', raw_data.get('粗糙度')),
            )
            parameters = ProcessParameter(
                spindle_speed=raw_data.get('spindle_speed', raw_data.get('主轴转速')),
                feed_rate=raw_data.get('feed_rate', raw_data.get('进给速度')),
                tool_diameter=raw_data.get('tool_diameter', raw_data.get('刀具直径')),
                cutting_depth=raw_data.get('cutting_depth', raw_data.get('切削深度')),
                cutting_width=raw_data.get('cutting_width', raw_data.get('切削宽度')),
            )
            case = ProcessCase(
                id=raw_data.get('id', raw_data.get('编号', '')),
                feature=feature,
                machining_method=raw_data.get('machining_method', raw_data.get('加工方法', '')),
                process_route=raw_data.get('process_route', raw_data.get('工艺路线', '')),
                parameters=parameters,
                material=raw_data.get('material', raw_data.get('材料')),
                machine_tool=raw_data.get('machine_tool', raw_data.get('机床')),
                tool=raw_data.get('tool', raw_data.get('刀具')),
            )
            return case
        except Exception as e:
            logger.warning(f'结构化失败: {e}')
            return None

    def convert_to_training_format(self, case: ProcessCase) -> TrainingDataItem:
        feature = case.feature
        input_parts = [f'加工特征：{feature.feature_type}']
        if feature.length is not None:
            input_parts.append(f'长度{feature.length}')
        if feature.width is not None:
            input_parts.append(f'宽度{feature.width}')
        if feature.diameter is not None:
            input_parts.append(f'直径{feature.diameter}')
        if feature.depth is not None:
            input_parts.append(f'深度{feature.depth}')
        if feature.precision is not None:
            input_parts.append(f'精度{feature.precision}')
        if feature.roughness is not None:
            input_parts.append(f'粗糙度{feature.roughness}')

        input_text = '。'.join(input_parts)
        output_parts = [f'加工方法：{case.machining_method}', f'加工工艺路线：{case.process_route}']
        output = '。'.join(output_parts)

        return TrainingDataItem(
            instruction='作为一名资深工艺加工工程师回答以下问题，通过给定加工特征与加工参数生成相应加工方法与工艺路线',
            input_text=input_text,
            output=output,
        )

    def structure_dataset(self, raw_data_list: list[dict]) -> list[ProcessCase]:
        cases = []
        for raw in raw_data_list:
            case = self.structure_process_case(raw)
            if case:
                cases.append(case)
        logger.info(f'结构化完成: {len(raw_data_list)} -> {len(cases)} 条')
        return cases

    def generate_training_data(self, cases: list[ProcessCase]) -> list[dict]:
        items = []
        for case in cases:
            item = self.convert_to_training_format(case)
            items.append({
                'instruction': item.instruction,
                'input': item.input_text,
                'output': item.output,
            })
        logger.info(f'训练数据生成完成: {len(items)} 条')
        return items

    def save_structured_data(self, cases: list[ProcessCase], filename: str = 'structured_cases.json'):
        filepath = self.output_dir / filename
        data = [case.model_dump() for case in cases]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f'结构化数据已保存至 {filepath}')
        return filepath

    def save_training_data(self, items: list[dict], filename: str = 'training_data.json'):
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.info(f'训练数据已保存至 {filepath}')
        return filepath

    def generate_knowledge_triples(self, case: ProcessCase) -> list[KnowledgeTriple]:
        triples = []
        feature = case.feature
        feature_name = FEATURE_NAME_MAP.get(feature.feature_type, feature.feature_type)
        method = case.machining_method

        feature_attrs = {}
        if feature.length is not None:
            feature_attrs['长度'] = feature.length
        if feature.width is not None:
            feature_attrs['宽度'] = feature.width
        if feature.diameter is not None:
            feature_attrs['直径'] = feature.diameter
        if feature.depth is not None:
            feature_attrs['深度'] = feature.depth
        if feature.precision is not None:
            feature_attrs['精度'] = feature.precision
        if feature.roughness is not None:
            feature_attrs['粗糙度'] = feature.roughness

        triples.append(KnowledgeTriple(
            source_entity=feature_name,
            relation='使用',
            target_entity=method,
            attributes=feature_attrs,
        ))

        if case.tool:
            tool_attrs = {}
            if case.parameters.tool_diameter is not None:
                tool_attrs['刀具直径'] = case.parameters.tool_diameter
            triples.append(KnowledgeTriple(
                source_entity=method,
                relation='需要',
                target_entity=case.tool,
                attributes=tool_attrs,
            ))

        if case.material:
            triples.append(KnowledgeTriple(
                source_entity=method,
                relation='适用于',
                target_entity=case.material,
                attributes={},
            ))

        if case.machine_tool:
            triples.append(KnowledgeTriple(
                source_entity=method,
                relation='在机床上执行',
                target_entity=case.machine_tool,
                attributes={},
            ))

        param_attrs = {}
        if case.parameters.spindle_speed is not None:
            param_attrs['主轴转速'] = case.parameters.spindle_speed
        if case.parameters.feed_rate is not None:
            param_attrs['进给速度'] = case.parameters.feed_rate
        if case.parameters.cutting_depth is not None:
            param_attrs['切削深度'] = case.parameters.cutting_depth
        if case.parameters.cutting_width is not None:
            param_attrs['切削宽度'] = case.parameters.cutting_width
        if param_attrs:
            triples.append(KnowledgeTriple(
                source_entity=method,
                relation='具有参数',
                target_entity=f'{method}工艺参数',
                attributes=param_attrs,
            ))

        return triples

    def auto_annotate(self, case: ProcessCase) -> ProcessCase:
        feature_type = case.feature.feature_type
        update_data = case.model_dump()

        if not case.machining_method or case.machining_method.strip() == '':
            methods = MACHINING_METHODS.get(feature_type, [])
            if methods:
                update_data['machining_method'] = methods[0]

        if not case.process_route or case.process_route.strip() == '':
            methods = MACHINING_METHODS.get(feature_type, [])
            if methods:
                update_data['process_route'] = methods[0]

        method = update_data.get('machining_method', case.machining_method)
        process_type = self._determine_process_type(method)

        speed_range = SPINDLE_SPEED_RANGE.get(process_type)
        feed_range = FEED_RATE_RANGE.get(process_type)

        if case.parameters.spindle_speed is None and speed_range:
            update_data['parameters']['spindle_speed'] = (speed_range[0] + speed_range[1]) / 2

        if case.parameters.feed_rate is None and feed_range:
            update_data['parameters']['feed_rate'] = (feed_range[0] + feed_range[1]) / 2

        if case.parameters.tool_diameter is None and case.feature.diameter is not None:
            update_data['parameters']['tool_diameter'] = case.feature.diameter

        if case.parameters.cutting_depth is None and case.feature.depth is not None:
            update_data['parameters']['cutting_depth'] = case.feature.depth * 0.5

        if case.parameters.cutting_width is None and case.feature.width is not None:
            update_data['parameters']['cutting_width'] = case.feature.width * 0.5

        if case.tool is None or case.tool.strip() == '':
            if case.parameters.tool_diameter is not None:
                update_data['tool'] = f'φ{case.parameters.tool_diameter}铣刀' if process_type == 'milling' else f'φ{case.parameters.tool_diameter}钻头'
            elif case.feature.diameter is not None:
                update_data['tool'] = f'φ{case.feature.diameter}铣刀' if process_type == 'milling' else f'φ{case.feature.diameter}钻头'

        return ProcessCase(**update_data)

    def structure_and_annotate(self, raw_data: dict) -> ProcessCase | None:
        case = self.structure_process_case(raw_data)
        if case is None:
            return None
        annotated = self.auto_annotate(case)
        return annotated

    def save_knowledge_triples(self, triples: list[KnowledgeTriple], filename: str = 'knowledge_triples.json'):
        filepath = self.output_dir / filename
        data = [triple.model_dump() for triple in triples]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f'知识三元组已保存至 {filepath}，共 {len(triples)} 条')
        return filepath
