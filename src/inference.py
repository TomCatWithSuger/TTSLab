from __future__ import annotations

import time
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nltk
import scipy.signal
import soundfile as sf
import torch

# g2p-en导入时会重复联网检查资源，镜像内已预置所需NLTK数据。
_nltk_download = nltk.download
nltk.download = lambda *args, **kwargs: True
try:
    from espnet2.bin.tts_inference import Text2Speech
    from espnet_model_zoo.downloader import ModelDownloader
finally:
    nltk.download = _nltk_download

from benchmark import TextSample, count_parameters, file_size_bytes


# Parallel WaveGAN 0.6.1仍从旧位置导入kaiser窗函数。
if not hasattr(scipy.signal, "kaiser"):
    setattr(scipy.signal, "kaiser", scipy.signal.windows.kaiser)


@dataclass(frozen=True)
class ModelSpec:
    """描述一个可运行的TTS模型及其专用推理参数。

    属性:
        name: 实验结果中使用的简短模型名称，同时作为输出目录名称。
        model_tag: ESPnet Model Zoo用于定位预训练模型的唯一标识。
        inference_options: 传递给ESPnet Text2Speech的模型专用解码参数。
    """

    name: str
    model_tag: str
    inference_options: dict[str, float]


# 两个模型共享测试文本和声码器，仅保留各自必要的解码参数。
MODEL_SPECS = {
    "fastspeech2": ModelSpec(
        name="fastspeech2",
        model_tag="kan-bayashi/ljspeech_conformer_fastspeech2",
        inference_options={"speed_control_alpha": 1.0},
    ),
    "tacotron2": ModelSpec(
        name="tacotron2",
        model_tag="kan-bayashi/ljspeech_tacotron2",
        inference_options={"threshold": 0.5, "minlenratio": 0.0, "maxlenratio": 10.0},
    ),
}

VOCODER_TAG = "ljspeech_hifigan.v1"
VOCODER_FILE_ID = "1i6-hR_ksEssCYNlNII86v3AoeA1JcuWD"


@dataclass
class LoadedModel:
    """保存已加载的ESPnet推理对象及模型规模信息。

    属性:
        spec: 当前模型对应的静态配置。
        synthesizer: 可直接接收文本并返回波形的ESPnet Text2Speech对象。
        checkpoint_path: 声学模型检查点在缓存目录中的实际路径。
        parameter_count: 声学模型的可训练参数总数。
        checkpoint_size_bytes: 声学模型检查点占用的字节数。
    """

    spec: ModelSpec
    synthesizer: Text2Speech
    checkpoint_path: Path
    parameter_count: int
    checkpoint_size_bytes: int


def _resolve_model_paths(downloaded: dict) -> tuple[Path, Path]:
    """从ESPnet下载结果中取得训练配置和模型检查点路径。

    参数:
        downloaded: ModelDownloader返回的模型文件映射，必须包含
            ``train_config``和``model_file``两个键。

    返回:
        二元组，第一个元素是训练配置路径，第二个元素是检查点路径。

    处理步骤:
        1. 从下载结果中读取两个必要字段。
        2. 检查模型包内容是否完整。
        3. 将字符串路径转换为Path对象后返回。
    """
    config = downloaded.get("train_config")
    checkpoint = downloaded.get("model_file")
    if config is None or checkpoint is None:
        raise RuntimeError(f"模型包缺少 train_config 或 model_file: {downloaded}")
    return Path(config), Path(checkpoint)


def _find_vocoder_files(cache_dir: Path) -> tuple[Path, Path] | None:
    """在声码器缓存目录中递归查找配置和检查点。

    参数:
        cache_dir: Parallel WaveGAN声码器的缓存根目录。

    返回:
        找到时返回``(config.yml路径, .pkl检查点路径)``；任一文件缺失时返回None。

    处理步骤:
        1. 递归收集所有config.yml文件。
        2. 递归收集所有.pkl检查点。
        3. 两类文件都存在时返回稳定排序后的匹配结果。
    """
    # 声码器解压目录名称由上游模型包决定，因此递归查找配置和检查点。
    configs = sorted(cache_dir.rglob("config.yml"))
    checkpoints = sorted(cache_dir.rglob("*.pkl"))
    if not configs or not checkpoints:
        return None
    return configs[0], checkpoints[-1]


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """验证并解压声码器tar归档，防止归档内容越过目标目录。

    参数:
        archive_path: 待解压的HiFi-GAN tar.gz文件。
        destination: 允许写入的目标缓存目录。

    返回:
        无返回值；验证失败或归档损坏时直接抛出异常。

    处理步骤:
        1. 计算目标目录的规范化绝对路径。
        2. 检查每个归档成员解压后是否仍位于目标目录内。
        3. 拒绝符号链接和硬链接，避免链接绕过路径检查。
        4. 全部成员通过验证后一次性解压。
    """
    # 下载文件属于外部输入，解压前拒绝目录穿越和链接成员。
    destination_root = destination.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if destination_root not in member_path.parents and member_path != destination_root:
                raise RuntimeError(f"声码器压缩包包含非法路径: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"声码器压缩包包含不允许的链接: {member.name}")
        archive.extractall(destination)


def _download_vocoder(cache_dir: Path) -> tuple[Path, Path]:
    """获取可用的HiFi-GAN配置和检查点，支持缓存、恢复和断点续传。

    参数:
        cache_dir: 声码器文件的持久化缓存目录。

    返回:
        二元组，第一个元素是声码器配置路径，第二个元素是检查点路径。

    处理步骤:
        1. 检查缓存中是否已经存在可用声码器，存在则直接返回。
        2. 如果存在完整归档，尝试安全解压；若归档不完整则改为.part续传文件。
        3. 使用gdown从固定Google Drive文件ID下载，最多重试三次。
        4. 下载完成后安全解压并删除压缩包，减少磁盘占用。
        5. 再次验证配置和检查点是否存在，然后返回实际路径。
    """
    existing = _find_vocoder_files(cache_dir)
    if existing is not None:
        return existing

    archive_path = cache_dir / f"{VOCODER_TAG}.tar.gz"
    if archive_path.exists():
        # 上次下载可能在完成后、解压前中断，优先尝试直接恢复。
        try:
            _safe_extract_tar(archive_path, cache_dir)
        except (tarfile.TarError, EOFError):
            resume_path = cache_dir / f"{archive_path.name}.resume.part"
            archive_path.replace(resume_path)
        else:
            archive_path.unlink()
            downloaded = _find_vocoder_files(cache_dir)
            if downloaded is None:
                raise RuntimeError("HiFi-GAN 压缩包中未找到 config.yml 或检查点")
            return downloaded

    url = f"https://drive.google.com/uc?id={VOCODER_FILE_ID}"
    last_error: subprocess.CalledProcessError | None = None
    # HiFi-GAN归档较大，保留.part文件并允许最多三次断点续传。
    for attempt in range(1, 4):
        try:
            subprocess.run(
                [sys.executable, "-m", "gdown", "--continue", url, "-O", str(archive_path)],
                check=True,
            )
            break
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt < 3:
                print(f"HiFi-GAN 下载中断，准备第 {attempt + 1} 次尝试", flush=True)
    else:
        raise RuntimeError("HiFi-GAN 下载连续失败") from last_error

    _safe_extract_tar(archive_path, cache_dir)
    archive_path.unlink()
    downloaded = _find_vocoder_files(cache_dir)
    if downloaded is None:
        raise RuntimeError("HiFi-GAN 压缩包中未找到 config.yml 或检查点")
    return downloaded


def load_model(spec: ModelSpec, cache_dir: Path) -> LoadedModel:
    """下载并加载一个ESPnet声学模型和共享HiFi-GAN声码器。

    参数:
        spec: 模型标识及对应推理参数。
        cache_dir: ESPnet模型和声码器共同使用的持久化缓存根目录。

    返回:
        LoadedModel对象，包含可调用的合成器、检查点路径、参数量和文件大小。

    处理步骤:
        1. 创建ESPnet与Parallel WaveGAN各自的缓存目录。
        2. 通过ModelDownloader下载或读取缓存中的ESPnet模型包。
        3. 获取共享HiFi-GAN配置和检查点。
        4. 使用训练配置、声学模型和声码器构建CPU版Text2Speech。
        5. 统计声学模型参数量和检查点大小并封装返回。
    """
    espnet_cache = cache_dir / "espnet"
    vocoder_cache = cache_dir / "parallel_wavegan"
    espnet_cache.mkdir(parents=True, exist_ok=True)
    vocoder_cache.mkdir(parents=True, exist_ok=True)

    downloader = ModelDownloader(str(espnet_cache))
    train_config, checkpoint = _resolve_model_paths(downloader.download_and_unpack(spec.model_tag))
    vocoder_config, vocoder_file = _download_vocoder(vocoder_cache)

    # Text2Speech根据训练配置恢复文本前端、声学模型和外部声码器。
    synthesizer = Text2Speech(
        train_config=train_config,
        model_file=checkpoint,
        vocoder_config=vocoder_config,
        vocoder_file=vocoder_file,
        device="cpu",
        dtype="float32",
        always_fix_seed=True,
        seed=777,
        **spec.inference_options,
    )
    return LoadedModel(
        spec=spec,
        synthesizer=synthesizer,
        checkpoint_path=checkpoint,
        parameter_count=count_parameters(synthesizer.tts),
        checkpoint_size_bytes=file_size_bytes(checkpoint),
    )


def _extract_waveform(result) -> np.ndarray:
    """把不同ESPnet版本的推理返回值统一转换为一维NumPy波形。

    参数:
        result: Text2Speech的返回结果，可能是包含``wav``的字典，也可能是元组；
            波形本身可能是Tensor、列表或元组。

    返回:
        dtype为float32的一维NumPy数组。

    处理步骤:
        1. 根据返回类型提取波形字段。
        2. 兼容新版批量接口返回的单元素列表或元组。
        3. 将PyTorch Tensor移到CPU并转换为NumPy数组。
        4. 统一数据类型和形状。
    """
    waveform = result["wav"] if isinstance(result, dict) else result[0]
    if isinstance(waveform, (list, tuple)):
        if len(waveform) != 1:
            raise RuntimeError(f"期望单条音频，实际得到 {len(waveform)} 条")
        waveform = waveform[0]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    return np.asarray(waveform, dtype=np.float32).reshape(-1)


def synthesize_samples(
    model: LoadedModel,
    samples: list[TextSample],
    output_dir: Path,
    warmup: bool,
) -> list[dict]:
    """使用一个已加载模型合成全部文本，并记录逐句性能指标。

    参数:
        model: 已加载的ESPnet模型及其规模信息。
        samples: 按固定ID组织的测试文本列表。
        output_dir: 当前模型WAV文件的输出目录。
        warmup: 是否先执行一次不计时推理，以排除首次初始化开销。

    返回:
        逐句实验记录列表。每条记录包含文本、推理时间、音频时长、RTF、
        采样率、模型参数量、检查点大小及输出路径。

    处理步骤:
        1. 创建输出目录并读取模型采样率。
        2. 根据配置执行一次不计入结果的预热推理。
        3. 对每条文本在torch.inference_mode下执行文本到波形推理。
        4. 根据样本数和采样率计算音频时长及RTF。
        5. 将波形保存为16位PCM WAV。
        6. 组装逐句指标、打印进度并返回全部记录。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    synthesizer = model.synthesizer
    if synthesizer.fs is None:
        raise RuntimeError(f"{model.spec.name} 未提供采样率")
    sample_rate = int(synthesizer.fs)

    # 预热用于排除首次调用的初始化开销，使两个模型的RTF更可比。
    if warmup:
        with torch.inference_mode():
            synthesizer(samples[0].text)

    records: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        # 计时范围只包含文本到波形的单次推理，不包含模型加载和文件写入。
        started = time.perf_counter()
        with torch.inference_mode():
            result = synthesizer(sample.text)
        elapsed = time.perf_counter() - started

        waveform = _extract_waveform(result)
        audio_seconds = len(waveform) / sample_rate
        if audio_seconds <= 0:
            raise RuntimeError(f"{model.spec.name}/{sample.sample_id} 生成了空音频")

        output_path = output_dir / f"{sample.sample_id}.wav"
        sf.write(output_path, waveform, sample_rate, subtype="PCM_16")
        record = {
            "model": model.spec.name,
            "model_tag": model.spec.model_tag,
            "vocoder_tag": f"parallel_wavegan/{VOCODER_TAG}",
            "sample_id": sample.sample_id,
            "text": sample.text,
            "text_characters": len(sample.text),
            "inference_seconds": elapsed,
            "audio_seconds": audio_seconds,
            "rtf": elapsed / audio_seconds,
            "sample_rate": sample_rate,
            "parameter_count": model.parameter_count,
            "checkpoint_size_bytes": model.checkpoint_size_bytes,
            "checkpoint_path": str(model.checkpoint_path),
            "audio_path": str(output_path),
        }
        records.append(record)
        print(
            f"[{model.spec.name}] {index}/{len(samples)} {sample.sample_id}: "
            f"{elapsed:.3f}s, {audio_seconds:.3f}s audio, RTF={record['rtf']:.3f}",
            flush=True,
        )
    return records
