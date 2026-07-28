"""Port de la cara: la salida expresiva de BMO."""

from __future__ import annotations

from abc import ABC, abstractmethod

from bmo.domain.models import Expression


class FacePort(ABC):
    """Recibe una expresion y la muestra en la cara de BMO."""

    @abstractmethod
    def show(self, expression: Expression) -> None:
        """Muestra `expression` en la cara. No devuelve nada: es una salida."""

    @property
    def available(self) -> bool:
        """Indica si la cara pudo inicializarse. Por defecto True."""
        return True

    def start(self) -> None:
        """Inicializa la salida (ventana, recursos). Por defecto no hace nada."""

    def run(self) -> None:
        """Corre el loop bloqueante de la cara. Por defecto no hace nada."""

    def stop(self) -> None:
        """Cierra la cara y libera recursos. Por defecto no hace nada."""


class NullFace(FacePort):
    """Cara que no hace nada.

    Patron Null Object: cuando no hay pantalla (tests, consola pura), el agente
    igual llama a `face.show(...)` sin tener que preguntar `if face is not None`.
    """

    def show(self, expression: Expression) -> None:
        return None
