import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import voice_pipeline as vp


def make_synthetic_voice(sr: int, dur_s: float, freq: float = 180.0, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(sr * dur_s)) / sr
    # Rich harmonic tone approximating voiced speech envelope + subtle natural floor
    sig = (
        0.55 * np.sin(2 * np.pi * freq * t)
        + 0.30 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.15 * np.sin(2 * np.pi * freq * 3 * t)
    )
    env = 0.5 * (1.0 - np.cos(2 * np.pi * np.clip(t / 0.08, 0, 0.5)))
    env *= 0.5 * (1.0 - np.cos(2 * np.pi * np.clip((dur_s - t) / 0.08, 0, 0.5)))
    rng = np.random.default_rng(7)
    floor = rng.standard_normal(len(t)) * 1e-3
    return (sig * env * amp + floor).astype(np.float64)


class VoicePipelineTests(unittest.TestCase):
    def test_polyphone_fix(self) -> None:
        self.assertEqual(vp.apply_polyphone_fixes("经过自校准和盲校准引擎"), "经过自较准和盲较准引擎")

    def test_bridge_digital_zero_gaps(self) -> None:
        sr = 44100
        a = make_synthetic_voice(sr, 0.6)
        dead_zero = np.zeros(int(0.35 * sr), dtype=np.float64)
        b = make_synthetic_voice(sr, 0.6, freq=210.0)
        y = np.concatenate([a, dead_zero, b])

        fixed, bridged = vp.bridge_digital_zero_gaps(y, sr)
        self.assertEqual(bridged, 1)
        mid = fixed[int(0.70 * sr) : int(0.85 * sr)]
        mid_db = 20.0 * np.log10(np.sqrt(np.mean(mid**2)) + 1e-14)
        self.assertGreater(mid_db, -80.0)
        self.assertLess(mid_db, -45.0)

    def test_prepare_reference_audio_engines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            src_wav = td_p / "raw_input.wav"
            sr = 32000
            a = make_synthetic_voice(sr, 1.0)
            gap = np.zeros(int(0.3 * sr))
            b = make_synthetic_voice(sr, 1.0, freq=200.0)
            y = np.concatenate([np.zeros(int(0.1 * sr)), a, gap, b, np.zeros(int(0.2 * sr))])
            wavfile.write(src_wav, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))

            out_vox = td_p / "ref_vox.wav"
            info_vox = vp.prepare_reference_audio(src_wav, out_vox, engine="voxcpm2")
            self.assertEqual(info_vox["sample_rate"], 44100)
            self.assertGreaterEqual(info_vox["bridged_zero_gaps"], 1)
            self.assertTrue(out_vox.exists())

            out_qwen = td_p / "ref_qwen.wav"
            info_qwen = vp.prepare_reference_audio(src_wav, out_qwen, engine="qwen3")
            self.assertEqual(info_qwen["sample_rate"], 24000)
            self.assertTrue(out_qwen.exists())

    def test_load_script_json_and_txt(self) -> None:
        sample = Path(__file__).resolve().parents[1] / "examples" / "sample_script.json"
        parsed = vp.load_script(sample)
        self.assertEqual(len(parsed["segments"]), 6)
        self.assertIn("较准", parsed["segments"][3]["say"])

        with tempfile.TemporaryDirectory() as td:
            txt = Path(td) / "demo.txt"
            txt.write_text("第一句用来测试自校准功能。\n第二句检查自动切段。\n", encoding="utf-8")
            parsed_txt = vp.load_script(txt)
            self.assertEqual(len(parsed_txt["segments"]), 1)
            self.assertIn("自较准", parsed_txt["segments"][0]["say"])

    def test_end_to_end_master_no_zero_cliffs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            work = td_p / "work"
            work.mkdir()
            sr = 44100
            segs = []
            for idx, freq in enumerate([175.0, 205.0], 1):
                sig = make_synthetic_voice(sr, 1.2, freq=freq, amp=0.12 if idx == 1 else 0.30)
                padded = np.concatenate([np.zeros(int(0.15 * sr)), sig, np.zeros(int(0.2 * sr))])
                fname = f"seg_{idx:02d}.wav"
                wavfile.write(work / fname, sr, (np.clip(padded, -1, 1) * 32767).astype(np.int16))
                segs.append({"id": idx, "file": fname, "pause": 0.45 if idx == 1 else 0.30})
            (work / "manifest.json").write_text(json.dumps({"title": "unit-test", "segments": segs}), encoding="utf-8")

            out_mp3 = td_p / "final.mp3"
            report = vp.master_audio(work, out_mp3, sample_rate=44100, target_lufs=-15.5, room_db=-62.0)
            self.assertTrue(out_mp3.exists())
            self.assertEqual(report["internal_zero_cliff_frames"], 0)
            self.assertLessEqual(report["true_peak_dbtp"], -1.2)
            self.assertGreater(report["lufs"], -18.5)
            self.assertLess(report["lufs"], -13.0)


if __name__ == "__main__":
    unittest.main()
