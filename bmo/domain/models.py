"""Value objects del dominio de BMO."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BoundingBox:
    """Rectángulo en coordenadas absolutas del frame."""

    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Detection:
    """Algo detectado en la escena (una cara, un ojo, una sonrisa...)."""

    label: str
    box: BoundingBox


@dataclass(frozen=True)
class Perception:
    """Lo que BMO 've' en un instante: la lista de detecciones.

    Es el resultado puro de la visión, SIN píxeles dibujados. El agente razona
    sobre esto; la capa de presentación (web/debug) es la que dibuja cajas.
    """

    detections: List[Detection] = field(default_factory=list)
    description: Optional[str] = None

    def labels(self) -> List[str]:
        return [d.label for d in self.detections]

    def count_by_label(self) -> dict[str, int]:
        conteo: dict[str, int] = {}
        for label in self.labels():
            conteo[label] = conteo.get(label, 0) + 1
        return conteo

    def summary(self) -> str:
        """Texto legible de lo que se ve, útil para el cerebro y el HUD."""
        if self.description:
            return self.description
        conteo = self.count_by_label()
        if not conteo:
            return "veo: nada"
        return "veo: " + ", ".join(f"{k} x{n}" for k, n in conteo.items())


@dataclass(frozen=True)
class Utterance:
    """Algo dicho: por el humano (escuchado vía STT) o por BMO."""

    text: str
    speaker: str = "human"


@dataclass(frozen=True)
class Note:
    """Una nota que BMO guarda para recordar (estilo Obsidian).

    - title: título corto para listarla en el menú táctil.
    - content: el texto completo de la nota.
    - created_at: momento en que se guardó (ordena el listado, la más nueva arriba).
    - id: identidad opaca en el almacén (p. ej. la ruta del fichero). La rellena
      el adapter al leer; permite borrar la nota exacta sin ambigüedad.
    """

    title: str
    content: str
    created_at: datetime
    id: str = ""


class Expression(str, Enum):
    """Estados emocionales de la cara de BMO."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    LISTENING = "listening"
    TALKING = "talking"
    THINKING = "thinking"
    SAD = "sad"
    ANGRY = "angry"
    CAPTURING = "capturing"
    WARMUP = "warmup"


@dataclass(frozen=True)
class ToolCall:
    """Una decisión del brain: 'ejecuta la tool `name` con estos `arguments`'.

    Nota: el brain solo DECIDE la llamada, no la ejecuta. El agente es quien
    ejecuta realmente la tool y le devuelve el resultado al brain.
    """

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """Un mensaje de la conversación que el agente le pasa al brain.

    role: 'system' (personalidad/instrucciones) | 'user' (el humano) |
    'assistant' (BMO) | 'tool' (resultado de ejecutar una tool).
    El brain es STATELESS: recibe la lista completa de mensajes en cada llamada,
    no guarda memoria. La memoria la lleva el agente.

    - tool_calls: presente cuando el assistant (BMO) pidió ejecutar tools. Es el
      turno que Ollama espera ANTES de los resultados 'tool'.
    - name: en un mensaje 'tool', de qué tool viene el resultado.
    """

    role: str
    content: str
    tool_calls: Tuple["ToolCall", ...] = ()
    name: Optional[str] = None


@dataclass(frozen=True)
class BrainDecision:
    """Lo que el brain resolvió tras razonar un paso.

    Es una de dos cosas:
    - `reply`: la respuesta final de BMO (no hacen falta más tools).
    - `tool_calls`: tools que el agente debe ejecutar antes de volver a pensar.
    """

    reply: Optional[Utterance] = None
    tool_calls: Tuple[ToolCall, ...] = ()

    @property
    def wants_tools(self) -> bool:
        """True si el brain pidió ejecutar tools (el agente debe seguir el loop)."""
        return len(self.tool_calls) > 0
