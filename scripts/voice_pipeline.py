#!/usr/bin/env python3
"""Natural Voice Clone Pipeline CLI.

End-to-end, zero-metallic Chinese voice cloning & comfort mastering workflow:
  - prep-ref : Prepare lossless, zero-atempo reference WAV (bridges -240 dB digital-zero gaps)
  - clone    : Segment-by-segment voice cloning via VoxCPM2 (44.1 kHz diffusion) or Qwen3-TTS 1.7B
  - master   : Restrained, ear-friendly studio mastering (-46 dB tail preservation, per-segment LUFS
               leveling, semantic pauses, continuous warm room-tone bed, no harsh 2-4 kHz boost)
  - play     : Reliable Windows speaker playback via ASCII temp WAV + sounddevice (no OpenWith/ffplay bugs)
  - run      : Full pipeline (prep-ref -> clone -> master [-> play])
  - doctor   : Check FFmpeg, Python deps, speaker output device, and remote cloning endpoints
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, welch

VOXCPM2_BASE = os.environ.get("NVC_VOXCPM2_URL", "https://openbmb-voxcpm-demo.hf.space")
QWEN3_BASE = os.environ.get("NVC_QWEN3_URL", "https://wordercom-qwen3-tts.hf.space")

STYLE_PRESETS: dict[str, str] = {
    "mediastorm": (
        "声音自然温暖舒适，像科技视频博主面对面轻松聊天分享一样，"
        "语速流畅、有真实呼吸感，完全没有播音腔和机械感"
    ),
    "podcast": (
        "声音温润松弛、低沉舒适，像深夜科技播客主播在安静录音棚里娓娓道来，"
        "节奏舒缓自然、有真实呼吸与停顿，无播音腔"
    ),
    "defense": (
        "声音清晰沉稳、自然真诚，像高校项目答辩人在现场从容讲解技术方案，"
        "咬字清楚但不生硬，语速适中、有真实换气感，无机械朗读腔"
    ),
}

# Common technical Chinese polyphones prone to mispronunciation by neural TTS
DEFAULT_POLYPHONE_MAP: list[tuple[str, str]] = [
    ("自校准", "自较准"),
    ("盲校准", "盲较准"),
    ("校准源", "较准源"),
    ("校准", "较准"),
    ("重采样", "崇采样"),
    ("调谐", "条谐"),
]

DEFAULT_PAUSES = [0.44, 0.48, 0.46, 0.48, 0.50, 0.44]


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr}")
    return proc


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    return run_cmd(["ffmpeg", "-hide_banner", "-v", "error", "-y", *args])


def biquad_peaking(f0: float, gain_db: float, q: float, fs: int) -> np.ndarray:
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * amp
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * amp
    a0 = 1.0 + alpha / amp
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / amp
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def apply_polyphone_fixes(text: str, extra_map: dict[str, str] | None = None) -> str:
    out = text
    for src, dst in DEFAULT_POLYPHONE_MAP:
        out = out.replace(src, dst)
    if extra_map:
        for src, dst in extra_map.items():
            out = out.replace(src, dst)
    return out


def frame_rms_db(y: np.ndarray, sr: int, hop_sec: float = 0.01) -> np.ndarray:
    hop = max(1, int(round(hop_sec * sr)))
    n = len(y) // hop
    if n <= 0:
        return np.array([-120.0], dtype=np.float64)
    rms = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-14)
    return 20.0 * np.log10(rms)


def synth_warm_room_tone(length: int, sr: int, target_dbfs: float = -62.0, seed: int = 2026) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(length)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, 1.0 / sr)
    shape = 1.0 / np.maximum(freqs, 40.0) ** 0.70
    shape[0] = 0.0
    shaped = np.fft.irfft(spec * shape, n=length)
    shaped = sosfilt(butter(2, 70.0, "highpass", fs=sr, output="sos"), shaped)
    shaped = sosfilt(butter(2, min(3800.0, sr * 0.42), "lowpass", fs=sr, output="sos"), shaped)
    rms = np.sqrt(np.mean(shaped**2) + 1e-14)
    return shaped * ((10.0 ** (target_dbfs / 20.0)) / rms)


def extract_or_synth_native_bed(y: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    """Extract real room tone from quiet non-zero frames (-75..-45 dBFS), else synthesize matched bed."""
    hop = int(0.01 * sr)
    db = frame_rms_db(y, sr, 0.01)
    # Prefer quiet natural frames that are NOT digital zero
    candidates = np.nonzero((db > -74.0) & (db < -46.0))[0]
    if len(candidates) >= 8:
        chunks = [y[i * hop : (i + 1) * hop] for i in candidates[:40]]
        sample = np.concatenate(chunks)
        # Smoothly tile with Hann crossfades or spectral match
        n_fft = min(2048, len(sample))
        win = np.hanning(n_fft)
        mag = np.abs(np.fft.rfft(sample[:n_fft] * win, n=n_fft))
        mag = np.convolve(mag, np.ones(9) / 9.0, mode="same")
        rng = np.random.default_rng(2026)
        w = np.fft.rfft(rng.standard_normal(target_len))
        f_w = np.fft.rfftfreq(target_len, 1.0 / sr)
        f_s = np.fft.rfftfreq(n_fft, 1.0 / sr)
        shaped = np.fft.irfft(w * np.interp(f_w, f_s, mag), n=target_len)
        target_rms = np.sqrt(np.mean(sample**2) + 1e-14) * 0.80
        cur_rms = np.sqrt(np.mean(shaped**2) + 1e-14)
        return shaped * (target_rms / cur_rms)
    return synth_warm_room_tone(target_len, sr, target_dbfs=-60.0)


def bridge_digital_zero_gaps(y: np.ndarray, sr: int, zero_thresh_db: float = -82.0) -> tuple[np.ndarray, int]:
    """Replace artificial -240 dB / <-82 dB digital-zero silences inside speech with matched room tone."""
    hop = int(0.01 * sr)
    db = frame_rms_db(y, sr, 0.01)
    is_zero = db < zero_thresh_db
    out = y.copy()
    bed = extract_or_synth_native_bed(y, sr, len(y))
    n_bridged = 0
    i = 0
    xf = int(0.015 * sr)
    while i < len(is_zero):
        if not is_zero[i]:
            i += 1
            continue
        j = i
        while j < len(is_zero) and is_zero[j]:
            j += 1
        # Only bridge internal gaps (>= 40ms) so outer lead/tail are handled cleanly by trim
        if (j - i) >= 4 and i > 2 and j < len(is_zero) - 2:
            s0, s1 = i * hop, min(len(out), j * hop)
            seg_len = s1 - s0
            f = min(xf, seg_len // 3)
            if f > 2:
                w = np.linspace(0.0, 1.0, f)
                out[s0 : s0 + f] = out[s0 : s0 + f] * (1.0 - w) + bed[s0 : s0 + f] * w
                out[s0 + f : s1 - f] = bed[s0 + f : s1 - f]
                out[s1 - f : s1] = bed[s1 - f : s1] * (1.0 - w) + out[s1 - f : s1] * w
            else:
                out[s0:s1] = bed[s0:s1]
            n_bridged += 1
        i = j
    return out, n_bridged


def prepare_reference_audio(src_path: Path, dst_wav: Path, engine: str = "voxcpm2") -> dict[str, Any]:
    """Zero-atempo, zero-compression reference prep tailored for VoxCPM2 (44.1kHz) or Qwen3-TTS (24kHz)."""
    if engine == "voxcpm2":
        target_sr, lead_s, tail_s = 44100, 0.08, 0.05
    elif engine == "qwen3":
        target_sr, lead_s, tail_s = 24000, 0.10, 0.20
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    with tempfile.TemporaryDirectory() as td:
        tmp_raw = Path(td) / "raw.wav"
        run_ffmpeg(["-i", str(src_path), "-ar", str(target_sr), "-ac", "1", "-c:a", "pcm_f32le", str(tmp_raw)])
        sr, y = wavfile.read(tmp_raw)
        y = y.astype(np.float64)

    # 1. Remove DC + 70 Hz 2nd-order high-pass (no atempo, no EQ coloration, no compression!)
    y = y - np.mean(y)
    y = sosfilt(butter(2, 70.0, "highpass", fs=sr, output="sos"), y)

    # 2. Bridge any internal digital-zero silence holes so the model never clones noise-gate cliffs
    y, bridged = bridge_digital_zero_gaps(y, sr)

    # 3. Trim to clean speech boundaries + model-specific lead/tail room tone
    db = frame_rms_db(y, sr, 0.01)
    p95 = float(np.percentile(db, 95))
    rel = db - p95
    active_on = np.nonzero(np.convolve((rel > -38.0).astype(float), np.ones(3), mode="same") >= 3)[0]
    active_off = np.nonzero(rel > -44.0)[0]
    s_on = max(0, int((active_on[0] - 1) * 0.01 * sr)) if len(active_on) else 0
    s_off = min(len(y), int((active_off[-1] + 2) * 0.01 * sr)) if len(active_off) else len(y)
    body = y[s_on:s_off].copy()

    # Softly blend body edges
    fi = min(int(0.015 * sr), len(body) // 6)
    fo = min(int(0.025 * sr), len(body) // 6)
    if fi > 1:
        body[:fi] *= np.linspace(0.0, 1.0, fi)
    if fo > 1:
        body[-fo:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fo)))

    lead = extract_or_synth_native_bed(y, sr, int(lead_s * sr)) * 0.85
    tail = extract_or_synth_native_bed(y, sr, int(tail_s * sr)) * 0.85
    if len(lead) > int(0.015 * sr):
        lead[: int(0.015 * sr)] *= np.linspace(0.0, 1.0, int(0.015 * sr))
    tail_fade = min(len(tail), int(0.03 * sr))
    if tail_fade > 1:
        tail[-tail_fade:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, tail_fade)))

    y_ref = np.concatenate([lead, body, tail])
    peak = float(np.max(np.abs(y_ref)) + 1e-12)
    y_ref = y_ref / peak * 0.75  # ~ -2.5 dBFS headroom

    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(dst_wav, sr, (np.clip(y_ref, -1.0, 1.0) * 32767.0).astype(np.int16))
    return {
        "output": str(dst_wav),
        "sample_rate": sr,
        "duration_sec": round(len(y_ref) / sr, 3),
        "bridged_zero_gaps": bridged,
        "engine": engine,
    }


def load_script(script_path: Path, polyphone_fix: bool = True) -> dict[str, Any]:
    """Load JSON script or plain UTF-8 text file (auto-chunking plain text into 40-75 char natural segments)."""
    raw = script_path.read_text(encoding="utf-8").strip()
    if script_path.suffix.lower() == ".json":
        data = json.loads(raw)
        segs = data["segments"] if isinstance(data, dict) else data
        meta = data if isinstance(data, dict) else {}
    else:
        meta = {}
        # Split on blank lines or lines; group short sentences into 40-75 char natural speaking chunks
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        segs = []
        buf = ""
        for ln in lines:
            if buf and len(buf) + len(ln) > 75:
                segs.append({"text": buf})
                buf = ln
            else:
                buf = (buf + ln) if not buf else (buf + ln)
        if buf:
            segs.append({"text": buf})

    normalized: list[dict[str, Any]] = []
    extra_map = meta.get("polyphone_map") if isinstance(meta, dict) else None
    for idx, item in enumerate(segs, 1):
        if isinstance(item, str):
            item = {"text": item}
        text = item["text"].strip()
        say = item.get("say", text).strip()
        if polyphone_fix and "say" not in item:
            say = apply_polyphone_fixes(say, extra_map)
        pause = float(item.get("pause", DEFAULT_PAUSES[(idx - 1) % len(DEFAULT_PAUSES)]))
        if idx == len(segs) and "pause" not in item:
            pause = 0.32
        normalized.append({"id": int(item.get("id", idx)), "text": text, "say": say, "pause": pause})

    return {
        "title": meta.get("title", script_path.stem),
        "artist": meta.get("artist", "natural-voice-clone"),
        "ref_text": meta.get("ref_text", ""),
        "instruction": meta.get("instruction", ""),
        "segments": normalized,
    }


def gradio_upload(base_url: str, filepath: Path) -> str:
    boundary = "----NVCFormBoundary" + uuid.uuid4().hex
    fdata = filepath.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{filepath.name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + fdata + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/gradio_api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "User-Agent": "Mozilla/5.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    return res[0]


def gradio_call_audio(base_url: str, endpoint: str, data: list[Any], out_path: Path, retries: int = 7) -> float:
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{base_url}/gradio_api/call{endpoint}",
                data=json.dumps({"data": data}).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                ev_id = json.loads(resp.read().decode("utf-8"))["event_id"]

            sse_req = urllib.request.Request(
                f"{base_url}/gradio_api/call{endpoint}/{ev_id}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(sse_req, timeout=240) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")

            low = raw.lower()
            if "busy" in low or "queue" in low:
                wait_s = min(3 * attempt, 12)
                print(f"  [retry {attempt}/{retries}] backend busy, waiting {wait_s}s...", flush=True)
                time.sleep(wait_s)
                continue

            for line in raw.splitlines():
                if line.startswith("data: "):
                    parsed = json.loads(line[6:])
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and parsed[0].get("url"):
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        urllib.request.urlretrieve(parsed[0]["url"], out_path)
                        return time.time() - t0
            raise RuntimeError(f"No audio URL returned from {base_url}{endpoint}: {raw[:300]}")
        except Exception as exc:
            if attempt == retries:
                raise
            wait_s = min(3 * attempt, 12)
            print(f"  [retry {attempt}/{retries}] {exc}; waiting {wait_s}s...", flush=True)
            time.sleep(wait_s)
    raise RuntimeError("Exhausted retries")


def clone_segments(
    ref_wav: Path,
    script_path: Path,
    work_dir: Path,
    engine: str = "voxcpm2",
    style: str = "mediastorm",
    instruction: str = "",
    ref_text: str = "",
    use_prompt_text: bool = False,
    cfg_value: float = 2.0,
    only_ids: set[int] | None = None,
    polyphone_fix: bool = True,
    retries: int = 7,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    script = load_script(script_path, polyphone_fix=polyphone_fix)
    unified_inst = instruction or script.get("instruction") or STYLE_PRESETS.get(style, STYLE_PRESETS["mediastorm"])
    final_ref_text = ref_text or script.get("ref_text", "")

    base_url = VOXCPM2_BASE if engine == "voxcpm2" else QWEN3_BASE
    print(f"Uploading reference audio to {base_url}...", flush=True)
    srv_path = gradio_upload(base_url, ref_wav)
    file_obj = {
        "path": srv_path,
        "url": f"{base_url}/gradio_api/file={srv_path}",
        "orig_name": ref_wav.name,
        "mime_type": "audio/wav",
        "meta": {"_type": "gradio.FileData"},
    }

    manifest_segs: list[dict[str, Any]] = []
    for seg in script["segments"]:
        sid = seg["id"]
        ext = ".mp3" if engine == "voxcpm2" else ".wav"
        dst = work_dir / f"seg_{sid:02d}{ext}"
        manifest_segs.append({**seg, "file": dst.name})

        if only_ids and sid not in only_ids:
            if not dst.exists():
                print(f"Warning: --only specified, but existing {dst.name} is missing.", flush=True)
            continue

        print(f"[Seg {sid:02d}] Generating ({engine}): {seg['say']}", flush=True)
        if engine == "voxcpm2":
            payload = [
                seg["say"],
                "" if use_prompt_text else unified_inst,
                file_obj,
                bool(use_prompt_text),
                final_ref_text if use_prompt_text else "",
                float(cfg_value),
                False,
                False,
            ]
            elapsed = gradio_call_audio(base_url, "/generate", payload, dst, retries=retries)
        elif engine == "qwen3":
            if not final_ref_text and not use_prompt_text:
                use_xvec = True
            else:
                use_xvec = not bool(final_ref_text)
            payload = [file_obj, final_ref_text, seg["say"], "Chinese", use_xvec, "1.7B"]
            elapsed = gradio_call_audio(base_url, "/generate_voice_clone", payload, dst, retries=retries)
        else:
            raise ValueError(f"Unsupported engine: {engine}")

        print(f"[Seg {sid:02d}] Saved {dst.name} ({dst.stat().st_size} bytes, {elapsed:.1f}s)", flush=True)
        time.sleep(1.0)

    manifest = {
        "title": script["title"],
        "artist": script["artist"],
        "engine": engine,
        "instruction": unified_inst,
        "segments": manifest_segs,
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def measure_lufs(audio_path: Path) -> dict[str, Any]:
    p = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(audio_path),
            "-af",
            "loudnorm=I=-15.5:TP=-1.8:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    if not m:
        raise RuntimeError(f"Could not parse loudnorm stats for {audio_path}")
    return json.loads(m.group(0))


def trim_natural_segment(y: np.ndarray, sr: int, head: float = 0.04, tail: float = 0.16) -> tuple[np.ndarray, float, float, bool]:
    """Trim segment with -40 dB onset & -46 dB offset to preserve vocal breath tails and prevent clipping."""
    hop = int(0.01 * sr)
    n = len(y) // hop
    if n < 5:
        return y.copy(), 0.0, 0.0, False
    r = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-14)
    db = 20.0 * np.log10(r / (np.percentile(r, 95) + 1e-12))
    run3 = np.convolve((db > -40.0).astype(float), np.ones(3), mode="same") >= 3
    on = np.nonzero(run3)[0]
    off = np.nonzero(db > -46.0)[0]
    s_on = max(0.0, (on[0] - 1) * 0.01) if len(on) else 0.0
    s_off = (off[-1] + 1) * 0.01 if len(off) else len(y) / sr
    c0 = max(0.0, s_on - head)
    c1 = min(len(y) / sr, s_off + tail)
    out = y[int(c0 * sr) : int(c1 * sr)].copy()

    fi = min(len(out), int((s_on - c0 + 0.01) * sr))
    if fi > 1:
        out[:fi] *= 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fi)))

    k_off = min(len(db) - 1, max(0, int(s_off / 0.01) - 1))
    cliff = k_off >= 2 and db[k_off - 2] > -35.0
    f0, f1 = ((s_off - 0.055, s_off + 0.03) if cliff else (s_off + 0.04, c1))
    i0 = max(0, int((f0 - c0) * sr))
    i1 = min(len(out), int((f1 - c0) * sr))
    if i1 > i0:
        out[i0:i1] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, i1 - i0)))
        out[i1:] = 0.0
    return out, s_on - c0, c1 - s_off, cliff


def master_audio(
    work_dir: Path,
    output_mp3: Path,
    sample_rate: int = 44100,
    target_lufs: float = -15.5,
    room_db: float = -62.0,
    bitrate: str = "256k",
    show_report: bool = False,
) -> dict[str, Any]:
    """Assemble and apply ear-friendly, zero-metallic studio mastering (no reverb, no harsh 2-4 kHz boost)."""
    manifest_path = work_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seg_entries = manifest["segments"]
        title = manifest.get("title", output_mp3.stem)
        artist = manifest.get("artist", "natural-voice-clone")
    else:
        files = sorted(list(work_dir.glob("seg_*.mp3")) + list(work_dir.glob("seg_*.wav")))
        if not files:
            raise FileNotFoundError(f"No manifest.json or seg_* audio files found in {work_dir}")
        seg_entries = [
            {"id": i + 1, "file": f.name, "pause": DEFAULT_PAUSES[i % len(DEFAULT_PAUSES)]}
            for i, f in enumerate(files)
        ]
        seg_entries[-1]["pause"] = 0.32
        title = output_mp3.stem
        artist = "natural-voice-clone"

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pieces: list[tuple[np.ndarray, float, float, float]] = []
        for seg in seg_entries:
            fpath = work_dir / seg["file"]
            w_raw = td_path / f"s{seg['id']:02d}.wav"
            run_ffmpeg(["-i", str(fpath), "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_f32le", str(w_raw)])
            _, y = wavfile.read(w_raw)
            y_trim, h_s, t_s, cliff = trim_natural_segment(y.astype(np.float64), sample_rate)
            w_trim = td_path / f"s{seg['id']:02d}_t.wav"
            wavfile.write(w_trim, sample_rate, y_trim.astype(np.float32))
            lufs_i = float(measure_lufs(w_trim)["input_i"])
            y_lvl = y_trim * (10.0 ** ((-18.0 - lufs_i) / 20.0))
            pieces.append((y_lvl, h_s, t_s, float(seg.get("pause", 0.46))))
            print(
                f"  [Master Seg {seg['id']:02d}] dur={len(y_lvl)/sample_rate:.2f}s | {lufs_i:.1f} -> -18.0 LUFS | cliff={cliff}",
                flush=True,
            )

        chunks: list[np.ndarray] = [np.zeros(int(0.08 * sample_rate), dtype=np.float64)]
        for k, (y_lvl, _h_s, t_s, pause_s) in enumerate(pieces):
            chunks.append(y_lvl)
            if k < len(pieces) - 1:
                next_h = pieces[k + 1][1]
                ins = max(0.04, pause_s - t_s - next_h)
                chunks.append(np.zeros(int(round(ins * sample_rate)), dtype=np.float64))
            else:
                chunks.append(np.zeros(int(round(pause_s * sample_rate)), dtype=np.float64))

        full = np.concatenate(chunks)

        # Restrained, warm, ear-friendly EQ (NO 2-4 kHz harsh boost, NO reverb)
        y_eq = sosfilt(butter(2, 75.0, "highpass", fs=sample_rate, output="sos"), full)
        y_eq = sosfilt(biquad_peaking(300.0, -1.0, 1.1, sample_rate), y_eq)
        y_eq = sosfilt(biquad_peaking(6800.0, -1.6, 1.8, sample_rate), y_eq)

        eq_wav = td_path / "eq.wav"
        comp_wav = td_path / "comp.wav"
        final_wav = td_path / "final.wav"
        wavfile.write(eq_wav, sample_rate, y_eq.astype(np.float32))

        run_ffmpeg([
            "-i", str(eq_wav),
            "-af", "acompressor=threshold=-21dB:ratio=1.35:attack=20:release=160:knee=5:makeup=1.0",
            "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_f32le",
            str(comp_wav),
        ])
        cur_lufs = float(measure_lufs(comp_wav)["input_i"])
        _, yc = wavfile.read(comp_wav)
        yc = yc.astype(np.float64) * (10.0 ** ((target_lufs - cur_lufs) / 20.0))

        # Continuous warm studio room-tone bed ducked under speech
        bed = synth_warm_room_tone(len(yc), sample_rate, target_dbfs=room_db, seed=2026)
        fi_b = min(len(bed) // 4, int(0.03 * sample_rate))
        fo_b = min(len(bed) // 4, int(0.18 * sample_rate))
        if fi_b > 1:
            bed[:fi_b] *= 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fi_b)))
        if fo_b > 1:
            bed[-fo_b:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fo_b)))
        win_e = max(1, int(0.03 * sample_rate))
        env = np.sqrt(np.convolve(yc**2, np.ones(win_e) / win_e, mode="same") + 1e-14)
        bed_g = np.clip((-46.0 - 20.0 * np.log10(env)) / 14.0, 0.15, 1.0)
        sm = max(1, int(0.04 * sample_rate))
        bed_g = np.convolve(bed_g, np.ones(sm) / sm, mode="same")
        y_mix = yc + bed * bed_g
        wavfile.write(final_wav, sample_rate, y_mix.astype(np.float32))

        output_mp3.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg([
            "-i", str(final_wav),
            "-af", "alimiter=limit=0.812:attack=6:release=60:level=false",
            "-ar", str(sample_rate), "-ac", "1",
            "-c:a", "libmp3lame", "-b:a", bitrate,
            "-metadata", f"artist={artist}",
            "-metadata", f"title={title}",
            str(output_mp3),
        ])

        report = inspect_audio(output_mp3, sample_rate=sample_rate)
        if show_report:
            print("\n=== Acoustic Quality Report ===")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return report


def inspect_audio(audio_path: Path, sample_rate: int = 44100) -> dict[str, Any]:
    stats = measure_lufs(audio_path)
    with tempfile.TemporaryDirectory() as td:
        chk = Path(td) / "chk.wav"
        run_ffmpeg(["-i", str(audio_path), "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_f32le", str(chk)])
        _, y = wavfile.read(chk)
        y = y.astype(np.float64)
    db = frame_rms_db(y, sample_rate, 0.01)
    p95 = float(np.percentile(db, 95))
    inner = db[5:-5] if len(db) > 12 else db
    zero_frames = int(np.sum(inner < -85.0))
    speech_mask = np.repeat(db > (p95 - 32.0), int(0.01 * sample_rate))
    ys = y[: len(speech_mask)][speech_mask]
    freqs, pxx = welch(ys if len(ys) > 1024 else y, fs=sample_rate, nperseg=min(4096, max(256, len(ys))))
    total = float(np.sum(pxx) + 1e-20)
    bands = {}
    for name, lo, hi in [
        ("warmth_80_220Hz", 80, 220),
        ("lowmid_220_500Hz", 220, 500),
        ("mid_500_2000Hz", 500, 2000),
        ("presence_2000_4500Hz", 2000, 4500),
        ("sibilance_4500_9000Hz", 4500, 9000),
    ]:
        m = (freqs >= lo) & (freqs < hi)
        bands[name] = round(100.0 * float(np.sum(pxx[m])) / total, 2)
    return {
        "file": str(audio_path),
        "duration_sec": round(len(y) / sample_rate, 2),
        "size_bytes": audio_path.stat().st_size,
        "lufs": float(stats["input_i"]),
        "true_peak_dbtp": float(stats["input_tp"]),
        "lra_lu": float(stats["input_lra"]),
        "noise_floor_p10_dbfs": round(float(np.percentile(db, 10)), 1),
        "internal_zero_cliff_frames": zero_frames,
        "spectral_pct": bands,
    }


def play_audio(audio_path: Path, blocking: bool = False) -> None:
    """Reliable Windows/cross-platform playback via ASCII temp WAV + sounddevice."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Convert to pure-ASCII temp WAV (prevents Windows codepage / OpenWith / ffplay bugs)
    tmp_dir = Path(tempfile.gettempdir())
    tmp_wav = tmp_dir / "nvc_playback.wav"
    run_ffmpeg(["-i", str(audio_path), "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(tmp_wav)])

    if blocking:
        import sounddevice as sd
        sr, data = wavfile.read(tmp_wav)
        print(f"Playing on: {sd.query_devices(kind='output')['name']}", flush=True)
        sd.play(data, sr)
        sd.wait()
        return

    helper = tmp_dir / "nvc_playback_runner.py"
    helper.write_text(
        "import sys, sounddevice as sd\n"
        "from scipy.io import wavfile\n"
        "sr, data = wavfile.read(sys.argv[1])\n"
        "sd.play(data, sr)\n"
        "sd.wait()\n",
        encoding="utf-8",
    )

    if sys.platform == "win32":
        # Clean up any previous nvc_playback_runner process before launching a new one
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'nvc_playback_runner' } "
                "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
            ],
            capture_output=True,
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        proc = subprocess.Popen([sys.executable, str(helper), str(tmp_wav)], creationflags=creationflags)
    else:
        proc = subprocess.Popen([sys.executable, str(helper), str(tmp_wav)], start_new_session=True)

    time.sleep(1.8)
    if proc.poll() is not None:
        raise RuntimeError(f"Playback exited unexpectedly with code {proc.returncode}")
    print(f"PLAYING_OK PID={proc.pid} file={audio_path.name}")


def run_doctor() -> None:
    print("=== Natural Voice Clone Environment Check ===")
    print(f"Python : {sys.version.split()[0]} ({sys.executable})")
    ff = shutil.which("ffmpeg")
    print(f"FFmpeg : {ff or 'MISSING (install ffmpeg)'}")
    for mod in ("numpy", "scipy", "sounddevice"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "ok")
            print(f"Module : {mod} ({ver})")
        except Exception as exc:
            print(f"Module : {mod} MISSING ({exc})")
    try:
        import sounddevice as sd
        print(f"Speaker: {sd.query_devices(kind='output')['name']}")
    except Exception as exc:
        print(f"Speaker: unavailable ({exc})")
    for name, url in [("VoxCPM2", VOXCPM2_BASE), ("Qwen3-TTS", QWEN3_BASE)]:
        try:
            req = urllib.request.Request(f"{url}/gradio_api/info", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                print(f"Remote : {name} ({url}) -> HTTP {r.status} OK")
        except Exception as exc:
            print(f"Remote : {name} ({url}) -> FAIL ({exc})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voice_pipeline", description="Natural Voice Clone & Comfort Mastering CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="Check local dependencies, audio device, and remote endpoints")

    p_ref = sub.add_parser("prep-ref", help="Prepare lossless, zero-atempo reference WAV")
    p_ref.add_argument("input", type=Path)
    p_ref.add_argument("output", type=Path)
    p_ref.add_argument("--engine", choices=["voxcpm2", "qwen3"], default="voxcpm2")

    p_cl = sub.add_parser("clone", help="Clone script segments using VoxCPM2 or Qwen3-TTS")
    p_cl.add_argument("ref_wav", type=Path)
    p_cl.add_argument("script", type=Path)
    p_cl.add_argument("work_dir", type=Path)
    p_cl.add_argument("--engine", choices=["voxcpm2", "qwen3"], default="voxcpm2")
    p_cl.add_argument("--style", choices=list(STYLE_PRESETS), default="mediastorm")
    p_cl.add_argument("--instruction", default="")
    p_cl.add_argument("--ref-text", default="")
    p_cl.add_argument("--use-prompt-text", action="store_true")
    p_cl.add_argument("--cfg", type=float, default=2.0)
    p_cl.add_argument("--only", default="", help="Comma-separated segment IDs to regenerate, e.g. 5,6")
    p_cl.add_argument("--no-polyphone-fix", action="store_true")
    p_cl.add_argument("--retries", type=int, default=7)

    p_ma = sub.add_parser("master", help="Assemble segments with ear-friendly zero-metallic mastering")
    p_ma.add_argument("work_dir", type=Path)
    p_ma.add_argument("output", type=Path)
    p_ma.add_argument("--sr", type=int, default=44100)
    p_ma.add_argument("--target-lufs", type=float, default=-15.5)
    p_ma.add_argument("--room-db", type=float, default=-62.0)
    p_ma.add_argument("--bitrate", default="256k")
    p_ma.add_argument("--report", action="store_true")

    p_pl = sub.add_parser("play", help="Play audio directly on default system speaker")
    p_pl.add_argument("audio", type=Path)
    p_pl.add_argument("--blocking", action="store_true")

    p_run = sub.add_parser("run", help="End-to-end: prep-ref -> clone -> master [-> play]")
    p_run.add_argument("ref_audio", type=Path)
    p_run.add_argument("script", type=Path)
    p_run.add_argument("output", type=Path)
    p_run.add_argument("--work-dir", type=Path, default=None)
    p_run.add_argument("--engine", choices=["voxcpm2", "qwen3"], default="voxcpm2")
    p_run.add_argument("--style", choices=list(STYLE_PRESETS), default="mediastorm")
    p_run.add_argument("--instruction", default="")
    p_run.add_argument("--ref-text", default="")
    p_run.add_argument("--use-prompt-text", action="store_true")
    p_run.add_argument("--cfg", type=float, default=2.0)
    p_run.add_argument("--target-lufs", type=float, default=-15.5)
    p_run.add_argument("--room-db", type=float, default=-62.0)
    p_run.add_argument("--bitrate", default="256k")
    p_run.add_argument("--no-polyphone-fix", action="store_true")
    p_run.add_argument("--report", action="store_true")
    p_run.add_argument("--play", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "doctor":
        run_doctor()
        return 0
    if args.cmd == "prep-ref":
        res = prepare_reference_audio(args.input, args.output, engine=args.engine)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "clone":
        only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None
        mp = clone_segments(
            args.ref_wav,
            args.script,
            args.work_dir,
            engine=args.engine,
            style=args.style,
            instruction=args.instruction,
            ref_text=args.ref_text,
            use_prompt_text=args.use_prompt_text,
            cfg_value=args.cfg,
            only_ids=only,
            polyphone_fix=not args.no_polyphone_fix,
            retries=args.retries,
        )
        print(f"Manifest written: {mp}")
        return 0
    if args.cmd == "master":
        master_audio(
            args.work_dir,
            args.output,
            sample_rate=args.sr,
            target_lufs=args.target_lufs,
            room_db=args.room_db,
            bitrate=args.bitrate,
            show_report=args.report,
        )
        print(f"Mastered audio saved: {args.output}")
        return 0
    if args.cmd == "play":
        play_audio(args.audio, blocking=args.blocking)
        return 0
    if args.cmd == "run":
        wd = args.work_dir or (args.output.parent / f".{args.output.stem}_work")
        ref_wav = wd / f"ref_{args.engine}.wav"
        ref_info = prepare_reference_audio(args.ref_audio, ref_wav, engine=args.engine)
        print("Reference prepared:", json.dumps(ref_info, ensure_ascii=False))
        clone_segments(
            ref_wav,
            args.script,
            wd,
            engine=args.engine,
            style=args.style,
            instruction=args.instruction,
            ref_text=args.ref_text,
            use_prompt_text=args.use_prompt_text,
            cfg_value=args.cfg,
            polyphone_fix=not args.no_polyphone_fix,
        )
        sr = 44100 if args.engine == "voxcpm2" else 48000
        master_audio(
            wd,
            args.output,
            sample_rate=sr,
            target_lufs=args.target_lufs,
            room_db=args.room_db,
            bitrate=args.bitrate,
            show_report=args.report,
        )
        print(f"Done: {args.output}")
        if args.play:
            play_audio(args.output)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
