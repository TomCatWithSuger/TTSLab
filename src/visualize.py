from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """读取单声道音频并保留文件原始采样率。

    参数:
        path: 待分析WAV文件路径。

    返回:
        二元组，第一个元素是一维浮点波形数组，第二个元素是原始采样率。

    处理步骤:
        1. 使用librosa读取音频文件。
        2. 设置sr=None避免自动重采样。
        3. 设置mono=True将多声道输入统一为单声道。
        4. 返回波形和采样率供后续绘图使用。
    """
    # 保留模型原始采样率，避免重采样影响波形和频谱比较。
    waveform, sample_rate = librosa.load(path, sr=None, mono=True)
    return waveform, sample_rate


def plot_waveform_comparison(audio_paths: dict[str, Path], output_path: Path, title: str) -> None:
    """把同一句文本的多个模型波形绘制到一张纵向对比图中。

    参数:
        audio_paths: 模型名称到WAV路径的映射，映射顺序决定子图顺序。
        output_path: 波形对比图PNG输出路径。
        title: 当前样本标识，会与模型名称共同显示在子图标题中。

    返回:
        无返回值；图像直接保存到output_path。

    处理步骤:
        1. 按模型数量创建纵向排列的子图。
        2. 逐个读取模型生成的波形和采样率。
        3. 使用真实时间刻度绘制时域振幅。
        4. 设置模型标题、时间轴和振幅轴标签。
        5. 调整布局、创建输出目录、保存PNG并释放图形资源。
    """
    # 每个模型独占一行，便于比较停顿位置和整体时长。
    figure, axes = plt.subplots(len(audio_paths), 1, figsize=(12, 3 * len(audio_paths)), squeeze=False)
    for axis, (model_name, audio_path) in zip(axes[:, 0], audio_paths.items()):
        waveform, sample_rate = _load_audio(audio_path)
        librosa.display.waveshow(waveform, sr=sample_rate, ax=axis)
        axis.set_title(f"{model_name} - {title}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Amplitude")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_spectrogram_comparison(audio_paths: dict[str, Path], output_path: Path, title: str) -> None:
    """把同一句文本的多个模型语谱图绘制到一张纵向对比图中。

    参数:
        audio_paths: 模型名称到WAV路径的映射，所有模型使用相同STFT参数。
        output_path: 语谱图PNG输出路径。
        title: 当前样本标识，用于区分不同测试句子。

    返回:
        无返回值；生成的语谱图直接写入output_path。

    处理步骤:
        1. 按模型数量创建纵向子图。
        2. 读取每个模型的原始采样率波形。
        3. 使用n_fft=1024、hop_length=256计算短时傅里叶变换。
        4. 将线性幅度转换为以最大幅度为参考的dB值。
        5. 绘制时间—频率图，并为所有子图添加统一颜色刻度。
        6. 调整边距、保存PNG并关闭Matplotlib图形。
    """
    figure, axes = plt.subplots(len(audio_paths), 1, figsize=(12, 4 * len(audio_paths)), squeeze=False)
    image = None
    for axis, (model_name, audio_path) in zip(axes[:, 0], audio_paths.items()):
        waveform, sample_rate = _load_audio(audio_path)
        # 两个模型固定使用相同STFT参数，保证频谱分辨率一致。
        spectrum = librosa.stft(waveform, n_fft=1024, hop_length=256)
        decibels = librosa.amplitude_to_db(np.abs(spectrum), ref=np.max)
        image = librosa.display.specshow(
            decibels,
            sr=sample_rate,
            hop_length=256,
            x_axis="time",
            y_axis="hz",
            ax=axis,
        )
        axis.set_title(f"{model_name} - {title}")
    if image is not None:
        figure.colorbar(image, ax=axes[:, 0].tolist(), format="%+2.0f dB")
    figure.subplots_adjust(left=0.08, right=0.9, top=0.94, bottom=0.08, hspace=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
