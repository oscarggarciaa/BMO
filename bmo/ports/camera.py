"""Port de la fuente de cámara: entrega frames crudos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CameraSourcePort(ABC):
    """Fuente de imagen. Implementaciones: picamera2 hoy, USB o AI Camera mañana."""

    @abstractmethod
    def start(self) -> None:
        """Inicializa y arranca la captura."""

    @abstractmethod
    def capture(self) -> Any:
        """Devuelve el último frame como array (formato BGR, listo para OpenCV).

        Retorna None si no hay frame disponible.
        """

    @abstractmethod
    def stop(self) -> None:
        """Detiene la captura y libera recursos."""
