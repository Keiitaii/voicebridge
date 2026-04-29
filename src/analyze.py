"""Per-utterance audio analysis (P1.1 — speech rate / F0 / jitter / shimmer).

Runs *after* whisper has produced a transcription. Takes the raw float32
audio array (the same one we just transcribed) and a few cheap libraries
(librosa, praat-parselmouth) to extract paralinguistic features. Output is
a flat dict, formatted into a structured block that's appended to the
prompt sent to claude.

Why: the user can say the same text in very different tones — calm vs
urgent, neutral vs upset. Whisper's transcription throws all that away.
Surfacing pitch / loudness / speaking rate to claude lets it tailor its
reply (shorter when the user sounds rushed; gentler when they sound tired).

Heavy / GPU-bound features (YAMNet sound events, wav2vec2 emotion) live in
P1.2 / P1.3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AnalysisResult:
    duration_s: float = 0.0
    rms: float = 0.0
    spectral_centroid_hz: float = 0.0
    f0_mean_hz: float = 0.0
    f0_min_hz: float = 0.0
    f0_max_hz: float = 0.0
    f0_std_hz: float = 0.0
    voiced_ratio: float = 0.0
    jitter_local: Optional[float] = None
    shimmer_local: Optional[float] = None
    speaking_rate_cps: float = 0.0  # filled later from transcribed text length

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _safe_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def analyze_pcm_float(audio: np.ndarray, sample_rate: int = 16000) -> AnalysisResult:
    """audio: 1D float32 in [-1, 1]. sample_rate: Hz."""
    res = AnalysisResult()
    if audio is None or audio.size == 0:
        return res
    audio = audio.astype(np.float32, copy=False)
    res.duration_s = float(audio.shape[0]) / sample_rate
    if res.duration_s < 0.2:
        return res

    # ---- librosa: RMS + spectral centroid ----
    try:
        import librosa
        rms = librosa.feature.rms(y=audio)
        res.rms = float(np.mean(rms))
        sc = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        res.spectral_centroid_hz = float(np.mean(sc))
    except Exception as e:
        # Don't fail analysis just because librosa errored on a weird block
        pass

    # ---- parselmouth: pitch + jitter + shimmer ----
    try:
        import parselmouth
        snd = parselmouth.Sound(audio, sampling_frequency=sample_rate)
        # pitch_floor=75 Hz, pitch_ceiling=500 Hz (covers male+female human range)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
        f0_arr = pitch.selected_array["frequency"]
        voiced = f0_arr[f0_arr > 0]
        res.voiced_ratio = float(voiced.size) / max(1, f0_arr.size)
        if voiced.size:
            res.f0_mean_hz = float(np.mean(voiced))
            res.f0_min_hz = float(np.min(voiced))
            res.f0_max_hz = float(np.max(voiced))
            res.f0_std_hz = float(np.std(voiced))

        # Jitter / shimmer require a PointProcess. Wrapped in try because if
        # there's not enough voiced signal the praat call raises.
        try:
            pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75.0, 500.0)
            res.jitter_local = _safe_float(parselmouth.praat.call(
                pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
            ))
            res.shimmer_local = _safe_float(parselmouth.praat.call(
                [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            ))
        except Exception:
            pass
    except Exception:
        pass

    return res


def attach_speaking_rate(res: AnalysisResult, transcribed_text: str) -> None:
    """Compute chars-per-second once we know the transcript length."""
    if res.duration_s > 0 and transcribed_text:
        res.speaking_rate_cps = len(transcribed_text) / res.duration_s


def _categorize_volume(rms: float) -> str:
    if rms < 0.015: return "弱"
    if rms < 0.05:  return "中"
    if rms < 0.15:  return "较大"
    return "大"


def _categorize_pitch(f0_mean: float, f0_std: float) -> str:
    if f0_mean < 1: return "(无人声)"
    base = "低" if f0_mean < 130 else ("中" if f0_mean < 220 else "高")
    if f0_std < 20: spread = "平稳"
    elif f0_std < 50: spread = "起伏自然"
    else: spread = "起伏大/激动"
    return f"{base},{spread}"


def format_for_prompt(res: AnalysisResult, transcribed_text: str) -> str:
    """Compose the prompt: transcribed text is the main signal, the
    paralinguistic metadata is a *secondary* hint at the bottom, framed so
    claude knows to weight the words much higher than the meta.
    """
    if res.duration_s < 0.2:
        return transcribed_text

    text = transcribed_text.strip()
    meta_bits = []
    meta_bits.append(f"语速{res.speaking_rate_cps:.1f}字/秒,音量{_categorize_volume(res.rms)}")
    if res.f0_mean_hz > 0:
        meta_bits.append(f"音高{_categorize_pitch(res.f0_mean_hz, res.f0_std_hz)}")
    if res.jitter_local is not None and res.shimmer_local is not None:
        j_pct = res.jitter_local * 100
        s_pct = res.shimmer_local * 100
        if j_pct > 1.5 or s_pct > 6:
            meta_bits.append("嗓音稍紧")
    meta_line = " · ".join(meta_bits)

    return f"{text}\n\n[副语音:{meta_line}]"


# ---------- standalone smoke ----------


def _demo():
    """Record 4 seconds of mic, transcribe with small whisper (if available),
    run analysis, print the formatted prompt block."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import sounddevice as sd
    import time
    duration = 4
    print(f"recording {duration}s @ 16kHz...")
    audio = sd.rec(duration * 16000, samplerate=16000, channels=1, dtype="float32",
                    device=10)
    sd.wait()
    a = audio[:, 0] if audio.ndim == 2 else audio
    res = analyze_pcm_float(a, 16000)
    res.speaking_rate_cps = 0  # no transcript
    print(format_for_prompt(res, "[demo]"))


if __name__ == "__main__":
    _demo()
