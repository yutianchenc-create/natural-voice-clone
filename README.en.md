<div align="center">

# Natural Voice Clone (`natural-voice-clone`)

**Zero-Metallic, Ear-Friendly Personal Voice Cloning Workflow & AI Agent Skill**

[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Python >= 3.10](https://img.shields.io/badge/Python-%3E%3D3.10-3B82F6?style=flat-square)](requirements.txt)
[![Audio: 44.1kHz Diffusion](https://img.shields.io/badge/Audio-44.1kHz%20VoxCPM2-8B5CF6?style=flat-square)](#architecture)
[![Agent Skill Compatible](https://img.shields.io/badge/Agent%20Skill-SKILL.md-F59E0B?style=flat-square)](SKILL.md)

[简体中文](README.md) · English · [Workflow & Pitfalls (CN)](docs/workflow-and-pitfalls.md)

</div>

---

`natural-voice-clone` packages a battle-tested, zero-metallic personal voice cloning and restrained studio mastering workflow into both a standalone **Python CLI** and a plug-and-play **AI Agent Skill (`SKILL.md`)**.

### Key Engineering Principles
1. **Zero-Destruction Reference Prep**: Never feed `atempo`-stretched, re-compressed, or heavily EQ'd audio into a voice cloning model—WSOLA phase artifacts get cloned into every single word. Decode losslessly, apply only DC removal + 70 Hz high-pass, and bridge artificial `-240 dB` digital-zero gaps with matched native room tone.
2. **44.1 kHz Continuous Diffusion First**: Uses **VoxCPM2** (`44.1 kHz` continuous diffusion) by default to eliminate 12 Hz discrete-codec metallic artifacts, with dedicated-GPU **Qwen3-TTS 1.7B** as a built-in fallback.
3. **Unified Style Control**: Extracts timbre while applying one consistent natural-language style instruction across all segments (`mediastorm`, `podcast`, `defense`, or custom) so timbre and mic distance never jump between paragraphs.
4. **Ear-Friendly Comfort Mastering**: Native `44.1 kHz / 256 kbps` output, `-46 dB` breath-tail preservation, per-segment `-18 LUFS` leveling, semantic pause spacing, `-62 dBFS` ducked continuous room tone, **no reverb**, and **no harsh 2–4 kHz boost**, mastered to `-15.5 LUFS / -1.8 dBTP`.
5. **Single-Segment Retakes & Reliable Windows Playback**: Built-in Chinese polyphone substitution (`say` override), `--only` partial regeneration, and ASCII temp WAV + `sounddevice` speaker playback on Windows.

### Quick Start

```bash
pip install -r requirements.txt
python scripts/voice_pipeline.py doctor
python scripts/voice_pipeline.py run my_voice.mp3 examples/sample_script.json out.mp3 --style mediastorm --report --play
```

### Install as an AI Agent Skill

```bash
git clone https://github.com/yutianchenc-create/natural-voice-clone.git ~/.gemini/config/skills/natural-voice-clone
```

### License & Responsible Use
MIT License. Only clone your own voice or voices with explicit permission.
