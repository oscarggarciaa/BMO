"""Port del oído: la entrada de voz de BMO.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class HearingPort(ABC):
    """Escucha el microfono y devuelve lo dicho como texto."""

    @abstractmethod
    def listen(self) -> str:
        """Bloquea hasta captar una frase y la devuelve transcrita.

        Devuelve "" si no se entendió nada (silencio, ruido, error de audio).
        """

    @property
    def available(self) -> bool:
        """Indica si el microfono pudo inicializarse. Por defecto True."""
        return True


class NullHearing(HearingPort):
    """Oído que no escucha nada.

    Patrón Null Object: cuando no hay micrófono (tests, BMO sin micrófono), el
    código igual puede llamar a `hearing.listen()` sin preguntar `if hearing is None`.
    `available` es False para que el arranque NO inicie el bucle de escucha.
    """

    def listen(self) -> str:
        return ""

    @property
    def available(self) -> bool:
        return False
