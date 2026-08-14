"""Carga config.yaml en dataclasses tipadas."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraConfig:
    adapter: str = "ai_camera_imx500"
    width: int = 1296
    height: int = 972
    format: str = "RGB888"

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class VisionConfig:
    adapter: str = "ai_camera_imx500"
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
    max_history: int = 0


@dataclass(frozen=True)
class VoiceConfig:
    adapter: str = "none" 
    model_path: str = "models/en_US-lessac-medium.onnx"
    device: str = ""
    sample_rate: int = 22050


@dataclass(frozen=True)
class HearingConfig:
    adapter: str = "none"
    model_path: str = "models/vosk-model-small-en-us-0.15"
    device: str = ""
    sample_rate: int = 16000
    wake_word: str = "hello"


@dataclass(frozen=True)
class ScreenConfig:
    enabled: bool = True
    fullscreen: bool = True
    fps: int = 4


@dataclass(frozen=True)
class NotesConfig:
    enabled: bool = True
    path: str = "notes" 


@dataclass(frozen=True)
class Config:
    camera: CameraConfig
    vision: VisionConfig
    brain: BrainConfig
    voice: VoiceConfig
    hearing: HearingConfig
    screen: ScreenConfig
    notes: NotesConfig
    debug: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = data or {}
        return cls(
            camera=CameraConfig(**_known(CameraConfig, data.get("camera"))),
            vision=VisionConfig(**_known(VisionConfig, data.get("vision"))),
            brain=BrainConfig(**_known(BrainConfig, data.get("brain"))),
            voice=VoiceConfig(**_known(VoiceConfig, data.get("voice"))),
            hearing=HearingConfig(**_known(HearingConfig, data.get("hearing"))),
            screen=ScreenConfig(**_known(ScreenConfig, data.get("screen"))),
            notes=NotesConfig(**_known(NotesConfig, data.get("notes"))),
            debug=bool(data.get("debug", False)),
        )

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def _known(config_cls: type, section: dict[str, Any] | None) -> dict[str, Any]:
    """Filtra una sección a las claves que el dataclass conoce.

    """
    valid = {f.name for f in fields(config_cls)}
    return {k: v for k, v in (section or {}).items() if k in valid}
