---
name: natural-voice-clone
description: >-
  Use this skill when the user asks to clone their own voice, generate natural
  human-like Chinese long-form narration (such as Mediastorm / 影视飓风 tech-creator
  style), remove AI/metallic/vocoder artifacts from TTS or cloned audio, or play
  generated audio reliably on Windows speakers.
---

# Natural Voice Clone（本人音色自然配音与去 AI 味工作流）

将短参考录音克隆为**零电音、自然松弛、耳感舒适**的中文长文本口播，默认对标「影视飓风」面对面聊天式科技讲解风格，并在 Windows 端提供可靠的扬声器直连播放与全篇回听验收闭环。

## 不可违背的铁律

1. **参考音频零破坏**：直接从最原始音频无损解码准备参考段，**严禁对参考音频做 `atempo` 变速拉伸、动态压缩、激进 EQ 染色或二次有损重编**（WSOLA 相位毛刺会被克隆模型放大成全篇电音）。仅允许去直流、70 Hz 高通，以及用原生底噪无缝填补 `-240 dB` 静音断层。
2. **严禁用固定公共 TTS 冒充本人音色**：只要用户要求“结合本人音色”或“更像真人”，必须使用零样本声纹克隆，不要拿固定商业/公共发音人（如 `edge-tts` 云希）替代。
3. **优先 44.1 kHz 连续扩散模型**：首选 **VoxCPM2**（原生 `44.1 kHz` 连续扩散架构，天生无 12 Hz 离散码本金属电音）；参考段裁为 `44.1 kHz`，开头留 `0.08s`，句尾仅留 `0.05s` 并做短余弦淡出。
4. **全篇统一风格指令**：当参考音频较短或原本偏朗读腔时，使用仅提取声纹 + 风格控制模式（`use_prompt_text=False`），且**所有分段必须共用完全一致的 `control_instruction`**，避免段间音色或话筒距离跳变。
5. **多音字规避 + 必须全篇亲耳回听**：发送给模型的文本须提前替换易错多音字（如「校准」→「较准」）；生成混音后必须用音频读取工具从头到尾试听验收，发现错音或突兀段落只用 `--only` 单段重配再重新混音。
6. **耳感舒适母带（不加混响、不硬提 2–4 kHz）**：全程保持原生 `44.1 kHz`，输出 `256 kbps` MP3；**禁止生硬拉高 2–4 kHz**（笔记本外放会放大电音与刺耳感），不加混响；仅做轻去闷、柔和压齿音、`1.35:1` 柔压缩与 `-62 dBFS` 连续室音托底，目标响度 `-15.5 LUFS / -1.8 dBTP`。

## 标准四步流程

### 第 1 步：改写口语化长文稿
- 按自然语义拆分为 **10–17 秒/段（约 40–70 字）**，兼顾语调连贯与生成稳定。
- 推荐「破误解 → 抛轻反问（对吧？）→ 坦诚踩坑 → 画面对比 → 通透收尾」的口播结构；两个标点间不超过 22 字，以口语双字词为主并保留少量单字，避免「不是……而是……」「综上所述」「总之」。
- 将英文缩写、数字与易错多音字写成念法（`scripts/voice_pipeline.py` 内置常见科技多音字替换，也支持在 JSON 中通过 `say` 字段显式指定念法）。

### 第 2 步：环境自检与一键生成
优先调用 [scripts/voice_pipeline.py](./scripts/voice_pipeline.py)：

```bash
# 1. 环境自检（FFmpeg、依赖包、扬声器设备、云端引擎连通性）
python scripts/voice_pipeline.py doctor

# 2. 一键端到端：零拉伸参考准备 -> VoxCPM2 分段克隆 -> 舒适母带 -> 导出 MP3
python scripts/voice_pipeline.py run ref_raw.mp3 script.json output.mp3 --style mediastorm --report
```

或分步精修（便于单段重配与二次调音）：

```bash
# 准备 44.1 kHz 纯净零拉伸参考段
python scripts/voice_pipeline.py prep-ref ref_raw.mp3 work/ref_pure.wav --engine voxcpm2

# 分段克隆（支持 --only 5,6 仅重配个别出错段落）
python scripts/voice_pipeline.py clone work/ref_pure.wav script.json work --style mediastorm

# 舒适无电音母带混音 + 声学质检报告
python scripts/voice_pipeline.py master work output.mp3 --report
```

> 若 VoxCPM2 临时不可用，可加 `--engine qwen3 --ref-text "<参考逐字稿>"` 切换至独立 GPU 上的 `Qwen3-TTS 1.7B`（`prep-ref --engine qwen3` 会自动按 `24 kHz`、头 `0.10s`、尾 `0.20s` 裁剪参考段）。

### 第 3 步：全篇试听与单段返修
- 用多模态音频读取工具完整听一遍 `output.mp3`，逐项核对：
  1. 是否零错字、零多音字误读；
  2. 是否零电音/金属颗粒感、外放耳朵舒适；
  3. 段间音量、底噪与话筒距离是否连贯，句尾是否自然不吞字。
- 如仅第 N 段有瑕疵，调整该段 `say` 字段或直接执行：
  ```bash
  python scripts/voice_pipeline.py clone work/ref_pure.wav script.json work --only N
  python scripts/voice_pipeline.py master work output.mp3 --report
  ```

### 第 4 步：Windows 本机可靠播放
当用户要求在电脑上直接播放试听时：
- **禁止** `Start-Process xxx.mp3`（Windows 无默认关联会弹出 `OpenWith` 选择框）；
- **禁止**直接把含中文路径的参数交给 WinGet `ffplay`（容易编码闪退）；
- 直接运行：
  ```bash
  python scripts/voice_pipeline.py play output.mp3
  ```
  脚本会自动清理旧播放进程、转码为纯 ASCII 临时 WAV，再通过 `sounddevice` 直连系统默认扬声器后台播放并验证进程存活。

## 延伸参考
- 详细声学参数依据与踩坑复盘见 [docs/workflow-and-pitfalls.md](./docs/workflow-and-pitfalls.md)。
- 示范分段脚本格式见 [examples/sample_script.json](./examples/sample_script.json)。
