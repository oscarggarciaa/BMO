"""Cerebro de BMO sobre el NPU Hailo-10H (AI HAT+ 2), vía servidor hailo-ollama.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from bmo.adapters.brain.ollama_brain import OllamaBrain
from bmo.config import BrainConfig


class HailoOllamaClient:
    """Cliente HTTP minimalista para hailo-ollama.
    """

    # hailo-ollama re-serializa los mensajes a JSON para su template de chat y
    # NO escapa los saltos de linea: un system prompt multilinea revienta el
    # parseo ("control character U+000A must be escaped") y devuelve 500
    # HAILO_INTERNAL_FAILURE. Se colapsan los control chars a un solo espacio
    # antes de enviar; el prompt se lee igual en una linea.
    _CONTROL_CHARS = re.compile(r"\s*[\r\n\t]+\s*")

    def __init__(self, host: str, http: Optional[Any] = None) -> None:
        self._url = host.rstrip("/") + "/api/chat"
        self._http = http

    def chat(self, model: str, messages: List[dict], **_ignored: Any) -> dict:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": m["role"],
                    "content": self._CONTROL_CHARS.sub(" ", m["content"]),
                }
                for m in messages
            ],
            "stream": False,
        }
        http = self._http
        if http is None:
            import httpx

            http = httpx
        response = http.post(self._url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()


class HailoBrain(OllamaBrain):
    """OllamaBrain apuntando a hailo-ollama, con limpieza de tokens de plantilla."""

    # a partir de ese token todo ruido
    _SPECIAL_TOKENS = re.compile(r"<\|.*", re.DOTALL)
    # regex para quitar los bloques <think>
    _THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    _THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)

    @classmethod
    def from_config(cls, config: BrainConfig) -> "HailoBrain":
        return cls(
            model=config.model,
            host=config.host,
            client=HailoOllamaClient(config.host),
        )

    def _sanitize_content(self, text: str) -> str:
        text = self._THINK_BLOCK.sub("", text)
        text = self._THINK_OPEN.sub("", text)
        return self._SPECIAL_TOKENS.sub("", text).strip()

