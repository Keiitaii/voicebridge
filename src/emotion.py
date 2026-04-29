"""Wav2Vec2 emotion classification (P1.3, dual zh/en model).

Two backends are kept loaded (lazily): one tuned for Chinese audio, one for
English. The active model is selected per-utterance based on the language
returned by the main ASR pass. If the user switches languages mid-dialog,
the next utterance picks the appropriate model automatically.

Default models:
  - English:  superb/wav2vec2-base-superb-er  (4-class IEMOCAP, ~360 MB)
  - Chinese:  same  (no widely-available open-source Chinese-specific
              wav2vec2 emotion checkpoint at the time of writing; the user
              can override with a better model from HF / modelscope by
              setting `tts.emotion.zh_model` in config.json)

For Chinese SOTA emotion you'd want emotion2vec from FunASR/ModelScope —
that's a larger framework integration we haven't done yet (left for P3+).
Once you wire it up, just point `zh_model` at the wrapper and this module
keeps working unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

DEFAULT_EN_MODEL = "superb/wav2vec2-base-superb-er"
# Chinese SOTA: emotion2vec_plus_seed via funasr (Alibaba). Loaded by the
# `iic/...` prefix → routed to the funasr backend below.
DEFAULT_ZH_MODEL = "iic/emotion2vec_plus_seed"
SAMPLE_RATE = 16000
MAX_INPUT_S = 3.0  # cap to bound CPU latency


@dataclass
class EmotionResult:
    valence: Optional[float] = None
    arousal: Optional[float] = None
    dominance: Optional[float] = None
    label: Optional[str] = None
    language: str = ""

    def is_valid(self) -> bool:
        return any(v is not None for v in
                    (self.valence, self.arousal, self.dominance, self.label))


class _Emotion2vecBackend:
    """funasr-hosted emotion2vec model (e.g. iic/emotion2vec_plus_seed).
    Outputs categorical Chinese labels: angry/disgusted/fearful/happy/
    neutral/other/sad/surprised/unknown.

    First-time loading downloads ~1.4 GB from modelscope.cn — slow over a
    flaky link, can take 20-60 min. To keep the agent's submit pipeline
    unblocked we load asynchronously: predict() returns an empty result
    while the load is in flight, instead of blocking on it.
    """

    def __init__(self, model_name: str = "iic/emotion2vec_plus_seed"):
        self.model_name = model_name
        self._model = None
        self._load_lock = __import__("threading").Lock()
        self._load_started = False
        self._load_failed = False

    def _ensure_load_started(self):
        with self._load_lock:
            if self._model is not None or self._load_started:
                return
            self._load_started = True
            import threading
            threading.Thread(target=self._load_blocking, daemon=True).start()

    def _load_blocking(self):
        try:
            from funasr import AutoModel
            self._model = AutoModel(model=self.model_name, disable_update=True)
            print(f"[emotion] {self.model_name} loaded")
        except Exception as e:
            self._load_failed = True
            print(f"[emotion] {self.model_name} load failed: {e}")

    def ensure_loaded(self):
        # Compat alias used elsewhere; non-blocking.
        self._ensure_load_started()

    def predict(self, audio: np.ndarray) -> EmotionResult:
        if audio is None or audio.size == 0:
            return EmotionResult()
        if self._model is None:
            self._ensure_load_started()
            return EmotionResult()  # not ready yet — silently skip this turn
        max_samples = int(MAX_INPUT_S * SAMPLE_RATE)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]
        a = audio.astype(np.float32, copy=False)
        try:
            res = self._model.generate(a, granularity="utterance", extract_embedding=False)
        except Exception:
            return EmotionResult()
        if not res:
            return EmotionResult()
        item = res[0]
        labels = item.get("labels", [])
        scores = item.get("scores", [])
        if not labels:
            return EmotionResult()
        idx = int(np.argmax(scores)) if scores else 0
        # emotion2vec_plus_seed labels are like "高兴/Happy" etc.; keep raw string.
        label_text = labels[idx]
        score = float(scores[idx]) if scores else 0.0
        return EmotionResult(label=f"{label_text} ({score:.2f})")


class _SingleModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._is_dim_model = ("msp-dim" in model_name) or model_name.startswith("audeering/")

    def ensure_loaded(self):
        if self._model is not None:
            return
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        from transformers import (
            Wav2Vec2Processor,
            Wav2Vec2FeatureExtractor,
            AutoModelForAudioClassification,
            AutoModel,
        )
        try:
            self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        except Exception:
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        if self._is_dim_model:
            self._model = AutoModel.from_pretrained(self.model_name)
        else:
            self._model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        self._model.eval()

    def predict(self, audio: np.ndarray) -> EmotionResult:
        if audio is None or audio.size == 0:
            return EmotionResult()
        self.ensure_loaded()
        import torch
        max_samples = int(MAX_INPUT_S * SAMPLE_RATE)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]
        a = audio.astype(np.float32, copy=False)
        try:
            inputs = self._processor(a, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        except TypeError:
            inputs = self._processor(audio=a, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)

        if self._is_dim_model:
            logits = getattr(outputs, "logits", None)
            if logits is None:
                return EmotionResult()
            arr = logits[0].detach().cpu().numpy()
            if arr.shape[0] >= 3:
                return EmotionResult(
                    valence=float(arr[0]),
                    arousal=float(arr[1]),
                    dominance=float(arr[2]),
                )
            return EmotionResult()
        logits = outputs.logits[0].detach().cpu().numpy()
        probs = _softmax(logits)
        id2label = self._model.config.id2label
        best = int(np.argmax(probs))
        return EmotionResult(label=f"{id2label[best]} ({probs[best]:.2f})")


class EmotionDetector:
    """Holds one _SingleModel per language tag, loaded lazily on first use.

    `predict(audio, language)` picks the model matching the language code; if
    the requested language isn't configured, falls back to the english model
    (which is also the safest default for unknown locales).
    """

    def __init__(self, zh_model: str = DEFAULT_ZH_MODEL,
                 en_model: str = DEFAULT_EN_MODEL):
        self._configs: Dict[str, str] = {"zh": zh_model, "en": en_model}
        self._cache: Dict[str, _SingleModel] = {}

    def _resolve(self, language: Optional[str]) -> str:
        lang = (language or "en").lower()
        # Normalize variants
        if lang.startswith("zh") or lang in ("chinese", "cmn", "yue"):
            return "zh"
        return "en"

    def _get(self, lang: str):
        if lang not in self._cache:
            name = self._configs[lang]
            # Route iic/* and emotion2vec_* to funasr; everything else to
            # transformers' Wav2Vec2ForSequenceClassification.
            if name.startswith("iic/") or "emotion2vec" in name:
                self._cache[lang] = _Emotion2vecBackend(name)
            else:
                self._cache[lang] = _SingleModel(name)
        return self._cache[lang]

    def ensure_loaded(self, language: Optional[str] = None):
        self._get(self._resolve(language)).ensure_loaded()

    def predict(self, audio: np.ndarray,
                language: Optional[str] = None) -> EmotionResult:
        lang = self._resolve(language)
        try:
            res = self._get(lang).predict(audio)
        except Exception as e:
            # Fall back to en model if zh is broken (or vice versa)
            other = "en" if lang == "zh" else "zh"
            try:
                res = self._get(other).predict(audio)
                lang = other
            except Exception:
                return EmotionResult(language=lang)
        res.language = lang
        return res


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def format_emotion(res: EmotionResult) -> str:
    if not res.is_valid():
        return ""
    if res.label is not None:
        return f"情绪标签:{res.label} ({res.language})"
    parts = []
    if res.valence is not None:
        v = res.valence
        if v < 0.4: tag = "偏不悦"
        elif v < 0.6: tag = "中性"
        else: tag = "偏愉悦"
        parts.append(f"愉悦度 {v:.2f} ({tag})")
    if res.arousal is not None:
        a = res.arousal
        if a < 0.4: tag = "平静"
        elif a < 0.65: tag = "中等"
        else: tag = "激动/急切"
        parts.append(f"激动度 {a:.2f} ({tag})")
    if res.dominance is not None:
        parts.append(f"主导感 {res.dominance:.2f}")
    return " · ".join(parts) + (f" ({res.language})" if res.language else "")
