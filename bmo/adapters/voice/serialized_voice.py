"""Voz thread-safe: serializa `speak()` para que no se solapen dos audios.

Es un decorador de `VoicePort` (patrón Decorator): envuelve cualquier voz real
y añade una única responsabilidad —garantizar que solo suene UN `speak()` a la
vez—. Varias entradas (conversación, reacción táctil...) comparten una sola
tarjeta de sonido; sin esto, dos `speak()` simultáneos lanzarían dos `aplay` a
la par y ALSA daría 'device busy' o mezclaría los audios. El lock hace que el
segundo `speak()` ESPERE a que termine el primero.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from bmo.ports.voice import VoicePort


class SerializedVoice(VoicePort):
    """Envuelve otra voz y garantiza que solo suene un `speak()` a la vez."""

    def __init__(self, inner: VoicePort) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._inner.available

    def speak(
        self, text: str, on_audio_start: Optional[Callable[[], None]] = None
    ) -> None:
        with self._lock:
            self._inner.speak(text, on_audio_start)
