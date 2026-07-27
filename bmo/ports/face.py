"""Port de la cara: la salida expresiva de BMO."""

from __future__ import annotations

from abc import ABC, abstractmethod

from bmo.domain.models import Expression


class FacePort(ABC):
    """Recibe una expresion y la muestra en la cara de BMO."""

    @abstractmethod
    def show(self, expression: Expression) -> None:
        """Muestra `expression` en la cara. No devuelve nada: es una salida."""


class NullFace(FacePort):
    """Cara que no hace nada.

    Patron Null Object: cuando no hay pantalla (tests, consola pura), el agente
    igual llama a `face.show(...)` sin tener que preguntar `if face is not None`.
    """

    def show(self, expression: Expression) -> None:
        return None
