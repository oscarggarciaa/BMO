"""
Thread-safe para que no se duplique la voz
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
