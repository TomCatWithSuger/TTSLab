from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import espnet
import numpy as np
import torch

from benchmark import load_text_samples, summarize_records, write_mos_template
from inference import MODEL_SPECS, VOCODER_TAG, load_model, synthesize_samples
from visualize import plot_spectrogram_comparison, plot_waveform_comparison


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """定义并解析完整实验支持的命令行参数。

    参数:
        无显式参数；ArgumentParser从当前进程的命令行读取选项。

    返回:
        argparse.Namespace对象，包含文本路径、缓存路径、输出路径、模型列表、
        文本数量限制、CPU线程数、预热开关和绘图开关。

    处理步骤:
        1. 创建实验命令行解析器。
        2. 设置项目内文本、模型缓存和输出目录的默认路径。
        3. 限制模型参数只能选择MODEL_SPECS中已配置的模型。
        4. 添加快速测试、CPU线程、预热和绘图控制选项。
        5. 解析当前命令行并返回结果。
    """
    parser = argparse.ArgumentParser(description="ESPnet CPU TTS comparison experiment")
    # 默认路径均以项目根目录为基准，容器内外使用同一套目录结构。
    parser.add_argument("--text-file", type=Path, default=ROOT / "data/texts/test.txt")
    parser.add_argument("--model-cache", type=Path, default=ROOT / "models/pretrained")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """在模型加载前校验用户提供的实验参数。

    参数:
        args: parse_args返回的命令行参数对象。

    返回:
        无返回值；参数有效时正常结束，无效时抛出带原因的异常。

    处理步骤:
        1. 检查文本数量限制是否大于零。
        2. 检查CPU线程数是否大于零。
        3. 检查测试文本文件是否真实存在且为普通文件。
    """
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须大于 0")
    if args.threads <= 0:
        raise ValueError("--threads 必须大于 0")
    if not args.text_file.is_file():
        raise FileNotFoundError(f"找不到测试文本: {args.text_file}")


def create_figures(samples, model_names: list[str], output_dir: Path) -> None:
    """为每条测试文本生成跨模型波形图和语谱图。

    参数:
        samples: 本次实验实际处理的TextSample列表。
        model_names: 本次参与实验的模型名称，决定需要对齐哪些WAV文件。
        output_dir: 实验输出根目录，其下应包含audio和figures目录。

    返回:
        无返回值；每条有效样本会在figures目录生成两张PNG。

    处理步骤:
        1. 根据样本ID拼接每个模型应有的WAV路径。
        2. 检查当前样本的所有模型音频是否齐全。
        3. 音频不完整时跳过，避免生成不完整的对比图。
        4. 调用波形绘图方法生成时域对比图。
        5. 调用语谱图方法生成频域对比图。
    """
    for sample in samples:
        # 只有参与本次实验的模型音频全部存在时才生成对比图。
        audio_paths = {
            model_name: output_dir / "audio" / model_name / f"{sample.sample_id}.wav"
            for model_name in model_names
        }
        if not all(path.is_file() for path in audio_paths.values()):
            continue
        plot_waveform_comparison(
            audio_paths,
            output_dir / "figures" / f"waveform_{sample.sample_id}.png",
            sample.sample_id,
        )
        plot_spectrogram_comparison(
            audio_paths,
            output_dir / "figures" / f"spectrogram_{sample.sample_id}.png",
            sample.sample_id,
        )


def main() -> None:
    """执行从文本读取到指标保存的完整ESPnet TTS对比实验。

    参数:
        无显式参数；实验配置由命令行参数和模块内模型配置共同决定。

    返回:
        无返回值；音频、图像和指标写入输出目录，摘要打印到标准输出。

    处理步骤:
        1. 解析并校验命令行参数。
        2. 固定CPU线程和随机种子，建立可复现实验条件。
        3. 读取测试文本，并按--limit截取快速测试子集。
        4. 依次加载每个模型、执行预热和逐句合成，再释放模型内存。
        5. 根据开关为全部样本生成波形图和语谱图。
        6. 汇总RTF、模型规模、运行平台和软件版本信息。
        7. 写入benchmark.json并创建不会覆盖已有评分的MOS模板。
        8. 在终端打印两个模型的核心实验结果和指标文件位置。
    """
    args = parse_args()
    validate_args(args)
    # 固定线程数和随机种子，降低重复实验之间的非模型差异。
    torch.set_num_threads(args.threads)
    np.random.seed(777)
    torch.manual_seed(777)

    samples = load_text_samples(args.text_file)
    if args.limit is not None:
        samples = samples[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    # 模型按顺序加载并释放，避免两个大型模型同时占用内存。
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        print(f"正在加载 {model_name}: {spec.model_tag}", flush=True)
        loaded_model = load_model(spec, args.model_cache)
        all_records.extend(
            synthesize_samples(
                loaded_model,
                samples,
                args.output_dir / "audio" / model_name,
                warmup=not args.no_warmup,
            )
        )
        del loaded_model

    if not args.skip_figures:
        create_figures(samples, args.models, args.output_dir)

    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # 环境版本与平台信息用于解释不同设备上的性能差异。
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "cpu_threads": args.threads,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "espnet_version": espnet.__version__,
        "vocoder_tag": f"parallel_wavegan/{VOCODER_TAG}",
        "text_file": str(args.text_file),
        "warmup_excluded_from_timing": not args.no_warmup,
        "summary": summarize_records(all_records),
        "utterances": all_records,
    }
    benchmark_path = metrics_dir / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_mos_template(metrics_dir / "mos.csv", samples, args.models)

    print("\n实验完成：", flush=True)
    for model_name, summary in report["summary"].items():
        size_mb = summary["checkpoint_size_bytes"] / 1024**2
        params_m = summary["parameter_count"] / 1_000_000
        print(
            f"- {model_name}: RTF={summary['mean_rtf']:.3f}, "
            f"参数={params_m:.2f}M, 检查点={size_mb:.2f}MiB",
            flush=True,
        )
    print(f"指标文件: {benchmark_path}", flush=True)


if __name__ == "__main__":
    main()