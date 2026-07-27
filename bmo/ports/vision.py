"""Port de vision: convierte un frame en una Perception (datos puros)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bmo.domain.models import Perception


class VisionPort(ABC):
    """Analiza un frame y devuelve QUE se ve, sin dibujar nada.

    Implementaciones: OpenCV+Haar (CPU) hoy; AI Camera IMX500 (inferencia en la
    propia camara) cuando llegue el hardware.
    """

    @abstractmethod
    def analyze(self, frame: Any) -> Perception:
        """Recibe un frame (array BGR) y devuelve la Perception detectada."""
