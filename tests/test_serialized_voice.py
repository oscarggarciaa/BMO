"""Tests del decorador de voz thread-safe (SerializedVoice)."""

from __future__ import annotations

import threading
from typing import Callable, List, Optional, Tuple

from bmo.adapters.voice.serialized_voice import SerializedVoice
from bmo.ports.voice import VoicePort


class RecordingVoice(VoicePort):
    """Voz que registra cada llamada a speak()."""

    def __init__(self, available: bool = True) -> None:
        self.calls: List[Tuple[str, Optional[Callable[[], None]]]] = []
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def speak(
        self, text: str, on_audio_start: Optional[Callable[[], None]] = None
    ) -> None:
        self.calls.append((text, on_audio_start))


def test_speak_delegates_text_and_callback() -> None:
    inner = RecordingVoice()
    callback = lambda: None

    SerializedVoice(inner).speak("hi", callback)

    assert inner.calls == [("hi", callback)]


def test_available_delegates_to_inner() -> None:
    assert SerializedVoice(RecordingVoice(available=False)).available is False
    assert SerializedVoice(RecordingVoice(available=True)).available is True


def test_speak_never_overlaps() -> None:
    """Dos speak() concurrentes NO deben ejecutarse a la vez."""
    inside = threading.Event()
    proceed = threading.Event()
    guard = threading.Lock()
    concurrency = {"now": 0, "max": 0}

    class SlowVoice(VoicePort):
        def speak(
            self, text: str, on_audio_start: Optional[Callable[[], None]] = None
        ) -> None:
            with guard:
                concurrency["now"] += 1
                concurrency["max"] = max(concurrency["max"], concurrency["now"])
            inside.set()
            proceed.wait(timeout=2)
            with guard:
                concurrency["now"] -= 1

    voice = SerializedVoice(SlowVoice())

    first = threading.Thread(target=voice.speak, args=("a",))
    first.start()
    assert inside.wait(timeout=2)  # el primero ya está dentro y tiene el lock
    inside.clear()

    second = threading.Thread(target=voice.speak, args=("b",))
    second.start()
    # el segundo NO debe poder entrar mientras el primero tiene el lock
    assert not inside.wait(timeout=0.2)

    proceed.set()  # libera al primero; entonces el segundo puede entrar
    first.join(timeout=2)
    second.join(timeout=2)

    assert concurrency["max"] == 1  # nunca dos audios simultáneos
