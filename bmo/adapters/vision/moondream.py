"""Adapter de vision usando un modelo multimodal de Ollama (moondream).

A diferencia de OpenCV+Haar (que SOLO encuentra caras/ojos/sonrisas y devuelve
un conteo), este adapter le pasa la imagen entera a un modelo que VE los pixeles
y describe la escena en lenguaje natural (objetos, colores, contexto).

Implementa el mismo VisionPort: recibe un frame y devuelve una Perception. La
diferencia es que la Perception viaja con una `description` de texto libre en vez
de detecciones estructuradas. El resto del sistema (la tool `look`, el agente) ni
se entera: sigue llamando `analyze(frame).summary()`.

Ojo con la RAM: moondream pesa ~1.7GB cargado. En una Pi de 4GB conviene usarlo
junto a un modelo de texto chico (ej. gemma3:1b).
"""

from __future__ import annotations

from typing import Any, Optional

from bmo.config import VisionConfig
from bmo.domain.models import Perception
from bmo.ports.vision import VisionPort

# Prompt en ingles a proposito: moondream describe mejor en ingles. Despues el
# cerebro de BMO (modelo de texto) reexpresa la descripcion en espaniol y con su
# personalidad, porque el resultado vuelve al loop del agente como texto.
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
        # Import lazy: en maquinas de desarrollo sin 'ollama' el modulo igual se
        # importa; solo se necesita el paquete al instanciar de verdad. El cliente
        # es inyectable para poder verificar sin Ollama.
        if client is None:
            import ollama

            client = ollama.Client(host=host)
        self._client = client

    @classmethod
    def from_config(cls, config: VisionConfig) -> "MoondreamVision":
        return cls(model=config.model, host=config.host, debug_path="last_look.jpg")

    def analyze(self, frame: Any) -> Perception:
        # Import lazy de cv2: solo se necesita al analizar de verdad (en la Pi).
        import cv2

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return Perception(description="no pude codificar la imagen")

        # DEBUG temporal: guarda la imagen EXACTA que se le manda a moondream,
        # para poder verificar si moondream alucina o si la camara da una imagen
        # mala. Borrar esta linea una vez diagnosticado.
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
