"""Port de la voz: la salida hablada de BMO."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional


class VoicePort(ABC):
    """Recibe un texto y lo reproduce como voz por los altavoces de BMO."""

    @abstractmethod
    def speak(
        self, text: str, on_audio_start: Optional[Callable[[], None]] = None
    ) -> None:
        """Dice `text` en voz alta. No devuelve nada: es una salida.

        `on_audio_start` (opcional) se dispara JUSTO cuando el sonido empieza a
        reproducirse, para sincronizar la cara (cambiarla a TALKING) con el
        audio real y no antes de sintetizar.
        """

    @property
    def available(self) -> bool:
        """Indica si la voz pudo inicializarse. Por defecto True."""
        return True


class NullVoice(VoicePort):
    """Voz que no hace nada.

    Patron Null Object: cuando no hay altavoces (tests, consola muda), el agente
    igual llama a `voice.speak(...)` sin tener que preguntar `if voice is not None`.
    Aun sin audio, dispara `on_audio_start` al instante para que la cara igual
    pueda cambiar a TALKING (la parte visual no depende de tener parlantes).
    """

    def speak(
        self, text: str, on_audio_start: Optional[Callable[[], None]] = None
    ) -> None:
        if on_audio_start is not None:
            on_audio_start()
        return None
