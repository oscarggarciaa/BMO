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
    model: str = "moondream"
    host: str = "http://localhost:11434"


@dataclass(frozen=True)
class BrainConfig:
    adapter: str = "ollama"
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"


@dataclass(frozen=True)
class WebConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 5000


@dataclass(frozen=True)
class Config:
    camera: CameraConfig
    vision: VisionConfig
    brain: BrainConfig
    web: WebConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = data or {}
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            vision=VisionConfig(**data.get("vision", {})),
            brain=BrainConfig(**data.get("brain", {})),
            web=WebConfig(**data.get("web", {})),
        )

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
