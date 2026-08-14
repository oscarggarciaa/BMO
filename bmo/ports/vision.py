"""Port de visión: convierte un frame en una Perception (datos puros)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bmo.domain.models import Perception


class VisionPort(ABC):
    """Analiza un frame y devuelve QUÉ se ve, sin dibujar nada.

    Implementación actual: AI Camera IMX500 (la inferencia corre en el propio
    sensor). El port permite cambiarla sin que el dominio se entere.
    """

    @abstractmethod
    def analyze(self, frame: Any, question: str = "") -> Perception:
        """Recibe un frame (array BGR) y una pregunta opcional del usuario.

        `question` da contexto al modelo de visión cuando lo soporta.
        """
