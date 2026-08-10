"""Carga config.yaml en dataclasses tipadas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraConfig:
    adapter: str = "picamera2"
    width: int = 1296
    height: int = 972
    format: str = "RGB888"

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class VisionConfig:
    adapter: str = "opencv_haar"
    models_path: str = "models"
    min_face_size: int = 80
    # ai_camera_imx500: .rpk del modelo on-sensor y umbral de confianza
    rpk_path: str = (
        "/usr/share/imx500-models/"
        "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    )
    threshold: float = 0.55


@dataclass(frozen=True)
class BrainConfig:
    adapter: str = "ollama"
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"


@dataclass(frozen=True)
class VoiceConfig:
    adapter: str = "none"  # none (sin voz) | piper (TTS neuronal offline)
    model_path: str = "models/en_US-lessac-medium.onnx"
    device: str = ""  # aplay -D (p. ej. plughw:1,0 para la placa USB); vacío = default
    sample_rate: int = 22050


@dataclass(frozen=True)
class HearingConfig:
    adapter: str = "none"  # none (sin micrófono) | vosk (STT offline)
    model_path: str = "models/vosk-model-small-en-us-0.15"
    device: str = ""  # arecord -D (p. ej. plughw:2,0 para la placa USB); vacío = default
    sample_rate: int = 16000  # 16kHz: el estándar para speech-to-text
    wake_word: str = "hello"  # la palabra que activa a BMO


@dataclass(frozen=True)
class ScreenConfig:
    enabled: bool = True
    fullscreen: bool = True
    fps: int = 4


@dataclass(frozen=True)
class Config:
    camera: CameraConfig
    vision: VisionConfig
    brain: BrainConfig
    voice: VoiceConfig
    hearing: HearingConfig
    screen: ScreenConfig
    # debug: True muestra TODO en el log; False solo entrada/salida.
    debug: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = data or {}
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            vision=VisionConfig(**data.get("vision", {})),
            brain=BrainConfig(**data.get("brain", {})),
            voice=VoiceConfig(**data.get("voice", {})),
            hearing=HearingConfig(**data.get("hearing", {})),
            screen=ScreenConfig(**data.get("screen", {})),
            debug=bool(data.get("debug", False)),
        )

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
