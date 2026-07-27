"""Port de la cara: la salida EXPRESIVA de BMO.

Asi como `VisionPort` es un puerto de ENTRADA (BMO percibe el mundo), `FacePort`
es un puerto de SALIDA (BMO le muestra al mundo como se siente). El agente
'empuja' expresiones a este puerto sin saber si detras hay una pantalla web,
una AI HAT con LEDs o nada.

Implementaciones:
- NullFace: no hace nada (default, para tests y modo consola sin web).
- FaceWebServer (bmo/interfaces/web): pinta la cara en un navegador via SSE.
"""

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

    def show(self, expression: Expression) -> None:  # noqa: D401 - no-op
        return None
