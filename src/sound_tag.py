"""YAMNet sound-event tagging (P1.2).

YAMNet is a 521-class audio event classifier from Google. Here we use it to
characterize the acoustic environment around the user's utterance — quiet
room, keyboard typing, music, dog bark, traffic, etc. The top-k labels go
into the prompt as a "<<环境声>>" hint so claude knows the context.

Loads tensorflow-hub model lazily on first call (~17 MB download cached
under the user's TFHub cache dir). Runs on CPU; YAMNet is small so a 5-15s
utterance takes ~30-100 ms, comfortable to inline in the submit path.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Tensorflow takes a beat to import; we lazy-load to keep agent startup snappy.

# Mapping from YAMNet display names → short Chinese hints. Only the labels we
# care about are mapped; unknown labels fall through as English to keep things
# transparent. Top-3 should usually all map.
_NAME_ZH = {
    "Silence": "安静",
    "Speech": "人声",
    "Conversation": "对话声",
    "Narration, monologue": "独白",
    "Music": "音乐",
    "Musical instrument": "乐器声",
    "Singing": "歌声",
    "Whispering": "低语",
    "Laughter": "笑声",
    "Crying, sobbing": "哭声",
    "Whistling": "口哨",
    "Breathing": "呼吸声",
    "Cough": "咳嗽",
    "Sneeze": "喷嚏",
    "Sigh": "叹气",
    "Yawn": "哈欠",
    "Computer keyboard": "键盘敲击",
    "Typing": "打字声",
    "Mouse click": "鼠标点击",
    "Tap": "敲击声",
    "Door": "开关门声",
    "Knock": "敲门声",
    "Footsteps": "脚步声",
    "Telephone": "电话铃",
    "Telephone bell ringing": "电话铃",
    "Cell phone": "手机声",
    "Notification": "提示音",
    "Air conditioning": "空调声",
    "Fan": "风扇声",
    "Mechanical fan": "风扇声",
    "Vacuum cleaner": "吸尘器",
    "Dishes, pots, and pans": "餐具声",
    "Cutlery, silverware": "餐具声",
    "Water tap, faucet": "水龙头",
    "Pour": "倒水声",
    "Boiling": "煮水声",
    "Microwave oven": "微波炉",
    "Frying (food)": "煎炒声",
    "Chopping (food)": "切菜声",
    "Cat": "猫叫",
    "Meow": "猫叫",
    "Purr": "猫呼噜",
    "Dog": "狗叫",
    "Bark": "狗叫",
    "Bird": "鸟叫",
    "Bird vocalization, bird call, bird song": "鸟叫",
    "Vehicle": "车声",
    "Car": "车声",
    "Engine": "引擎声",
    "Truck": "卡车声",
    "Motorcycle": "摩托车",
    "Aircraft": "飞机声",
    "Wind": "风声",
    "Rain": "雨声",
    "Thunderstorm": "雷声",
    "Television": "电视声",
    "Radio": "广播声",
    "Applause": "掌声",
    "Cheering": "欢呼",
    "Inside, small room": "室内/小空间",
    "Inside, large room or hall": "室内/大空间",
    "Outside, urban or manmade": "室外/街道",
    "Outside, rural or natural": "室外/自然",
    "Background noise": "背景噪音",
    "White noise": "白噪声",
    "Pink noise": "粉噪",
    "Hum": "嗡嗡声",
    "Buzz": "蜂鸣",
    "Hiss": "嘶嘶声",
    "Static": "静电声",
    "Mains hum": "电源杂音",
}


@dataclass
class SoundTags:
    top: List[Tuple[str, float]] = field(default_factory=list)  # (display_name, score)
    rms: float = 0.0


class SoundTagger:
    """Lazy-loaded YAMNet wrapper."""

    def __init__(self):
        self._model = None
        self._class_names: Optional[List[str]] = None

    def ensure_loaded(self):
        if self._model is not None:
            return
        # Suppress TF logging noise
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow_hub as hub
        self._model = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map_path = self._model.class_map_path().numpy().decode()
        names = []
        with open(class_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names.append(row["display_name"])
        self._class_names = names

    def predict(self, audio: np.ndarray, top_k: int = 3) -> SoundTags:
        """audio: 1D float32 mono @ 16 kHz, in [-1, 1]."""
        if audio is None or audio.size == 0:
            return SoundTags()
        self.ensure_loaded()
        # YAMNet expects exactly 16 kHz float32
        a = audio.astype(np.float32, copy=False)
        scores, embeddings, log_mel = self._model(a)
        mean_scores = np.mean(scores.numpy(), axis=0)
        # Top-k indices (descending)
        top_idx = np.argsort(mean_scores)[-top_k:][::-1]
        top = [(self._class_names[i], float(mean_scores[i])) for i in top_idx]
        rms = float(np.sqrt(np.mean(a ** 2)))
        return SoundTags(top=top, rms=rms)


def to_zh(label: str) -> str:
    return _NAME_ZH.get(label, label)


def format_tags(tags: SoundTags) -> str:
    """Compose a 1-line description for the prompt block."""
    if not tags.top:
        return ""
    parts = []
    for name, score in tags.top:
        zh = to_zh(name)
        parts.append(f"{zh}({score:.2f})")
    return " · ".join(parts)


# ---------- standalone smoke ----------


if __name__ == "__main__":
    import sounddevice as sd
    print("recording 4s @ 16 kHz...")
    audio = sd.rec(4 * 16000, samplerate=16000, channels=1, dtype="float32", device=10)
    sd.wait()
    a = audio[:, 0] if audio.ndim == 2 else audio
    print("loading YAMNet...")
    tagger = SoundTagger()
    tagger.ensure_loaded()
    print("predicting...")
    tags = tagger.predict(a, top_k=5)
    print(f"top: {tags.top}")
    print(f"formatted: {format_tags(tags)}")
