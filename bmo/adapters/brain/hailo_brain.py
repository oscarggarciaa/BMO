"""Cerebro de BMO sobre el NPU Hailo-10H (AI HAT+ 2), vía servidor hailo-ollama.

`hailo-ollama` expone la MISMA API REST que Ollama (endpoint /api/chat, misma
estructura de respuesta), por eso este adapter REUSA toda la logica de
OllamaBrain heredando de el.

Hay DOS diferencias reales con Ollama:

1. Tokens de plantilla: hailo-ollama NO filtra los tokens especiales de Llama 3
   (<|start_header_id|>, <|eot_id|>, ...) y el modelo suele alucinar un turno
   nuevo detras de ellos. Se recortan en `_sanitize_content`.

2. Request minimalista: el parser interno de hailo-ollama (oatpp) es RIGIDO y
   devuelve 500 ("Node is NOT a STRING") si el request incluye campos como
   `tools` u `options` con valores null. El cliente estandar de `ollama` los
   agrega solo, asi que aqui usamos un cliente HTTP propio que manda UNICAMENTE
   {model, messages, stream:false}. Confirmado en el foro oficial de Hailo.

Se activa con `adapter: hailo` y `host: http://localhost:8000` en config.yaml.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from bmo.adapters.brain.ollama_brain import OllamaBrain
from bmo.config import BrainConfig


class HailoOllamaClient:
    """Cliente HTTP minimalista para hailo-ollama.

    Imita la interfaz `.chat(model, messages, **kwargs)` del cliente `ollama`,
    pero manda SOLO los campos que el parser rigido de hailo-ollama acepta y
    DESCARTA el resto (options, tools, etc.) que provocan el 500.
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

    # Todo lo que va desde el primer token especial de Llama 3 en adelante es
    # ruido de plantilla (o un turno alucinado): se descarta.
    _SPECIAL_TOKENS = re.compile(r"<\|.*", re.DOTALL)

    # Modelos "thinking" (qwen3) razonan entre <think>...</think> ANTES de la
    # respuesta real. hailo-ollama NO filtra nada, asi que hay que quitarlo aqui
    # o BMO diria su monologo interno en voz alta. Primero los bloques cerrados;
    # luego cualquier <think> huerfano (respuesta cortada sin </think>).
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

