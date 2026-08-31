from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TextSample:
    """表示一条可在两个模型之间稳定对齐的测试文本。

    属性:
        sample_id: 样本唯一编号，同时用于WAV和图像文件名。
        text: 送入ESPnet文本前端的英文原文。
    """

    sample_id: str
    text: str


def load_text_samples(path: Path) -> list[TextSample]:
    """读取并校验``ID|文本``格式的TTS测试集。

    参数:
        path: UTF-8编码测试文本文件路径，每个非空行应为``样本ID|英文文本``。

    返回:
        按文件原始顺序排列的TextSample列表。

    处理步骤:
        1. 逐行读取文件并跳过空行。
        2. 使用第一个竖线拆分样本ID和文本。
        3. 检查分隔符、ID和文本是否完整。
        4. 检查样本ID是否重复，保证跨模型结果可以一一对应。
        5. 将有效记录转换为TextSample；若没有样本则抛出异常。
    """
    samples: list[TextSample] = []
    seen_ids: set[str] = set()

    # 稳定的样本ID用于对齐两个模型的音频、图像和人工评分。
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            sample_id, separator, text = line.partition("|")
            sample_id = sample_id.strip()
            text = text.strip()
            if not separator or not sample_id or not text:
                raise ValueError(f"{path}:{line_number} 必须使用 ID|文本 格式")
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number} 包含重复 ID: {sample_id}")
            seen_ids.add(sample_id)
            samples.append(TextSample(sample_id=sample_id, text=text))

    if not samples:
        raise ValueError(f"测试文本为空: {path}")
    return samples


def count_parameters(module) -> int:
    """统计一个PyTorch模块包含的参数元素总数。

    参数:
        module: 提供``parameters()``方法的PyTorch模型模块。

    返回:
        所有参数张量元素数量之和，不区分参数当前是否参与梯度更新。

    处理步骤:
        1. 遍历模型的全部参数张量。
        2. 使用numel取得每个张量的元素数。
        3. 累加后返回模型参数总量。
    """
    return sum(parameter.numel() for parameter in module.parameters())


def file_size_bytes(path: Path) -> int:
    """取得指定模型检查点在磁盘上的文件大小。

    参数:
        path: 已存在的模型检查点路径。

    返回:
        文件占用的字节数。

    处理步骤:
        1. 读取文件系统元数据。
        2. 返回其中的st_size字段。
    """
    return path.stat().st_size


def summarize_records(records: Iterable[dict]) -> dict[str, dict]:
    """把逐句推理记录汇总为按模型划分的实验指标。

    参数:
        records: 可迭代的逐句记录，每条记录必须包含模型名、推理时间、
            音频时长、RTF、参数量、检查点大小和采样率。

    返回:
        以模型名为键的汇总字典，包含句子数、总推理时间、总音频时长、
        总体RTF、逐句平均RTF、参数量、检查点大小和采样率。

    处理步骤:
        1. 按``model``字段对逐句记录分组。
        2. 分别累加每组的推理时间和音频时长。
        3. 计算按总音频时长加权的总体RTF。
        4. 计算每句话等权重的平均RTF。
        5. 从同组首条记录提取不会逐句变化的模型规模信息。
    """
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record)

    summary: dict[str, dict] = {}
    for model_name, model_records in grouped.items():
        total_inference = sum(item["inference_seconds"] for item in model_records)
        total_audio = sum(item["audio_seconds"] for item in model_records)
        # 总体RTF按音频时长加权，逐句平均RTF则让每个句子权重相同。
        summary[model_name] = {
            "utterances": len(model_records),
            "total_inference_seconds": total_inference,
            "total_audio_seconds": total_audio,
            "mean_rtf": total_inference / total_audio if total_audio else None,
            "mean_utterance_rtf": sum(item["rtf"] for item in model_records)
            / len(model_records),
            "parameter_count": model_records[0]["parameter_count"],
            "checkpoint_size_bytes": model_records[0]["checkpoint_size_bytes"],
            "sample_rate": model_records[0]["sample_rate"],
        }
    return summary


def write_mos_template(path: Path, samples: list[TextSample], model_names: list[str]) -> None:
    """为人工MOS自然度评价创建待填写CSV模板。

    参数:
        path: MOS CSV输出路径。
        samples: 需要参加主观评价的测试样本。
        model_names: 需要评价的模型名称列表。

    返回:
        无返回值；模板直接写入指定路径。

    处理步骤:
        1. 检查目标文件是否已有内容，避免覆盖人工评分。
        2. 创建指标输出目录。
        3. 写入听众、样本、模型、评分和备注字段。
        4. 为每个样本与每个模型生成一行空白评分记录。
    """
    # 已填写的主观评分不可被重复运行实验时覆盖。
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["listener_id", "sample_id", "model", "score", "notes"])
        for sample in samples:
            for model_name in model_names:
                writer.writerow(["", sample.sample_id, model_name, "", ""])
