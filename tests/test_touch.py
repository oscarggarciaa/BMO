"""Tests de la reacción táctil de BMO (build_touch_reaction)."""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

from bmo.domain.models import Expression
from bmo.interfaces.touch import DEFAULT_PHRASE, build_touch_reaction


class FakeFace:
    """Cara falsa: registra las expresiones que le piden mostrar."""

    def __init__(self) -> None:
        self.shown: List[Expression] = []

    def show(self, expression: Expression) -> None:
        self.shown.append(expression)


class FakeVoice:
    """Voz falsa: registra lo que le piden decir y dispara on_audio_start."""

    def __init__(self) -> None:
        self.said: List[str] = []

    def speak(self, text: str, on_audio_start: Optional[Callable[[], None]] = None) -> None:
        if on_audio_start is not None:
            on_audio_start()  # como piper: avisa que el audio arranca
        self.said.append(text)


def test_touch_shows_angry_then_reverts_after_speaking() -> None:
    face = FakeFace()
    voice = FakeVoice()
    react = build_touch_reaction(face, voice)

    react()

    # cara enfadada al instante, y vuelve a NEUTRAL al terminar de hablar
    assert face.shown == [Expression.ANGRY, Expression.NEUTRAL]
    assert voice.said == [DEFAULT_PHRASE]


def test_touch_uses_custom_phrase() -> None:
    face = FakeFace()
    voice = FakeVoice()
    react = build_touch_reaction(face, voice, phrase="oww")

    react()

    assert voice.said == ["oww"]


def test_touch_reverts_even_if_voice_fails() -> None:
    face = FakeFace()

    class BoomVoice:
        def speak(self, text: str, on_audio_start: Optional[object] = None) -> None:
            raise RuntimeError("no speakers")

    react = build_touch_reaction(face, BoomVoice())

    try:
        react()
    except RuntimeError:
        pass

    # aunque la voz reviente, la cara vuelve a NEUTRAL (bloque finally)
    assert face.shown[-1] is Expression.NEUTRAL


def test_touch_debounces_overlapping_reactions() -> None:
    face = FakeFace()
    inside = threading.Event()
    release = threading.Event()
    said: List[str] = []

    class BlockingVoice:
        """Se queda dentro de speak() hasta que se libere: sostiene el lock."""

        def speak(self, text: str, on_audio_start: Optional[object] = None) -> None:
            said.append(text)
            inside.set()
            release.wait(timeout=2)

    react = build_touch_reaction(face, BlockingVoice())

    worker = threading.Thread(target=react)
    worker.start()
    assert inside.wait(timeout=2)  # la primera reacción tiene el lock tomado

    react()  # segundo toque simultáneo: debe ignorarse (no toma el lock)

    release.set()
    worker.join(timeout=2)

    assert said == [DEFAULT_PHRASE]  # solo una queja, no dos
