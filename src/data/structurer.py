import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src.config.settings import (
    ANNOTATED_DATA_DIR,
    FEATURE_NAME_MAP,
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


class DataStructurer:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or ANNOTATED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_feature_type(self, feature_type: str) -> str:
        for eng, chn in FEATURE_NAME_MAP.items():
            if chn in feature_type or eng in feature_type.lower():
                return eng
        return feature_type

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
