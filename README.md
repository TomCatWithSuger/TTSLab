# TTSLab

基于 **ESPnet-TTS** 的英文语音合成对比实验项目。项目使用FastSpeech 2和Tacotron 2预训练模型合成语音，并生成WAV、波形图、语谱图以及RTF、模型规模和MOS评价结果。

## 当前配置

| 项目       | 配置                       |
| ---------- | -------------------------- |
| 预训练模型 | FastSpeech 2、Tacotron 2   |
| 声码器     | HiFi-GAN                   |
| 输入       | 英文文本                   |
| 输出采样率 | 22.05 kHz                  |
| 推理设备   | CPU                        |
| 评价指标   | RTF、参数量、模型大小、MOS |
| 运行环境   | Docker Compose             |
| 默认文本数 | 10条                       |

## 主流程

```text
读取测试文本
      ↓
检查/下载预训练模型和声码器
      ↓
FastSpeech 2与Tacotron 2语音合成
      ↓
生成波形图、语谱图和评价结果
```

模型已有完整缓存时会自动跳过下载。

## 项目结构

```text
TTSLab/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── run_experiment.py    # 主流程入口
│   ├── inference.py         # ESPnet模型加载与语音合成
│   ├── benchmark.py         # RTF与模型规模统计
│   └── visualize.py         # 波形图与语谱图
├── data/texts/              # 英文测试文本
├── models/pretrained/       # 预训练模型缓存
├── outputs/                 # 音频、图像和指标结果
└── docs/语音合成实验报告.md  # 实验步骤
```

## 快速运行

宿主机只需要安装并启动Docker，然后在项目根目录执行：

```bash
docker compose build
docker compose run --rm app
```

使用一条文本快速测试：

```bash
docker compose run --rm app \
    python src/run_experiment.py --limit 1
```

只运行一个模型：

```bash
docker compose run --rm app \
    python src/run_experiment.py --models fastspeech2
```

## 输出结果

```text
outputs/
├── audio/
│   ├── fastspeech2/         # FastSpeech 2生成的WAV
│   └── tacotron2/           # Tacotron 2生成的WAV
├── figures/                 # 波形图与语谱图
└── metrics/
    ├── benchmark.json       # RTF与模型规模结果
    └── mos.csv              # MOS人工评分表
```

详细环境搭建和实验说明见[语音合成实验报告](docs/语音合成实验报告.md)。
