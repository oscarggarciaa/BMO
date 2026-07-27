"""Adapter de vision multimodal con Ollama (moondream)."""

from __future__ import annotations

from typing import Any, Optional

from bmo.config import VisionConfig
from bmo.domain.models import Perception
from bmo.ports.vision import VisionPort

_DEFAULT_PROMPT = "Describe what you see in this image in one short, simple sentence."


class MoondreamVision(VisionPort):
    """Vision multimodal: le pide a un modelo de Ollama que describa la imagen."""

    def __init__(
        self,
        model: str,
        host: str,
        client: Optional[Any] = None,
        prompt: str = _DEFAULT_PROMPT,
        debug_path: Optional[str] = None,
    ) -> None:
        self._model = model
        self._prompt = prompt
        self._debug_path = debug_path
        if client is None:
            import ollama

            client = ollama.Client(host=host)
        self._client = client

    @classmethod
    def from_config(cls, config: VisionConfig) -> "MoondreamVision":
        return cls(model=config.model, host=config.host, debug_path="last_look.jpg")

    def analyze(self, frame: Any) -> Perception:
        import cv2

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return Perception(description="no pude codificar la imagen")

        if self._debug_path is not None:
            with open(self._debug_path, "wb") as f:
                f.write(buffer.tobytes())

        response = self._client.chat(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": self._prompt,
                    "images": [buffer.tobytes()],
                }
            ],
        )
        text = (response["message"].get("content") or "").strip()
        return Perception(description=text or "no vi nada claro")
