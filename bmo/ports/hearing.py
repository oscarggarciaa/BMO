"""Port del oido: la entrada de voz de BMO.

Es el simetrico del VoicePort. Asi como `VoicePort.speak(text)` SACA voz,
`HearingPort.listen()` METE voz transcribida a texto. El adapter solo hace
audio -> texto; decidir si ese texto despierta a BMO es cosa del dominio.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class HearingPort(ABC):
    """Escucha el microfono y devuelve lo dicho como texto."""

    @abstractmethod
    def listen(self) -> str:
        """Bloquea hasta captar una frase y la devuelve transcripta.

        Devuelve "" si no se entendio nada (silencio, ruido, error de audio).
        """

    @property
    def available(self) -> bool:
        """Indica si el microfono pudo inicializarse. Por defecto True."""
        return True


class NullHearing(HearingPort):
    """Oido que no escucha nada.

    Patron Null Object: cuando no hay microfono (tests, BMO sordo), el codigo
    igual puede llamar a `hearing.listen()` sin preguntar `if hearing is None`.
    `available` es False para que el arranque NO inicie el bucle de escucha.
    """

    def listen(self) -> str:
        return ""

    @property
    def available(self) -> bool:
        return False
