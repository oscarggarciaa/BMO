"""Tests del oido de BMO: escuchar por microfono con wake-word 'hello'.

El diseno separa dos responsabilidades:
- ADAPTER (VoskHearing): audio del microfono -> texto. Nada mas.
- DOMINIO (WakeWord): decide si el texto despierta a BMO. Puro y testeable.
"""

from __future__ import annotations

from typing import Callable, List

from bmo.adapters.hearing.vosk_hearing import VoskHearing
from bmo.domain.models import Utterance
from bmo.domain.wakeword import WakeWord
from bmo.interfaces.microphone import listen_loop
from bmo.ports.hearing import HearingPort, NullHearing


class ScriptedHearing(HearingPort):
    """Oido falso: devuelve transcripts en orden y corta con KeyboardInterrupt."""

    def __init__(self, transcripts: List[str]) -> None:
        self._transcripts = list(transcripts)

    def listen(self) -> str:
        if not self._transcripts:
            raise KeyboardInterrupt
        return self._transcripts.pop(0)


class SpyAgent:
    """Agente espia: guarda lo que se le pregunto, en orden."""

    def __init__(self) -> None:
        self.asked: List[str] = []

    def ask(self, text: str) -> Utterance:
        self.asked.append(text)
        return Utterance(text="ok", speaker="bmo")


# --- WakeWord: la logica de 'hello' es dominio puro ------------------------


def test_wakeword_ignores_transcript_without_hello() -> None:
    assert WakeWord().detect("what do you see") is None


def test_wakeword_alone_returns_empty_command() -> None:
    # Solo "hello": BMO despierta pero no hay comando (se saludara).
    assert WakeWord().detect("hello") == ""


def test_wakeword_returns_command_after_hello() -> None:
    assert WakeWord().detect("hello what do you see") == "what do you see"


def test_wakeword_is_case_insensitive() -> None:
    assert WakeWord().detect("Hello there") == "there"


def test_wakeword_only_wakes_when_hello_is_first() -> None:
    # "hello" en el medio NO despierta: la palabra magica va al principio.
    assert WakeWord().detect("say hello to me") is None


def test_wakeword_ignores_empty_transcript() -> None:
    assert WakeWord().detect("") is None
    assert WakeWord().detect("   ") is None


# --- NullHearing: BMO sordo (sin microfono) --------------------------------


def test_null_hearing_never_hears_and_is_unavailable() -> None:
    ear = NullHearing()
    assert ear.listen() == ""
    assert ear.available is False


# --- listen_loop: el pegamento entre oido, wake-word y agente --------------


def test_listen_loop_asks_agent_only_after_wake_word() -> None:
    agent = SpyAgent()
    hearing = ScriptedHearing(
        ["what time is it", "hello", "hello what do you see"]
    )

    listen_loop(agent, hearing, WakeWord())

    # "what time is it" se ignora (sin wake-word); "hello" solo -> saludo;
    # "hello what do you see" -> se pregunta el comando.
    assert agent.asked == ["hello", "what do you see"]


# --- VoskHearing: el adapter delega en una sesion inyectable ---------------


def test_vosk_hearing_reuses_injected_session() -> None:
    # La sesion (modelo + stream) se crea UNA vez y se reutiliza en cada listen.
    calls: List[int] = []

    def session() -> str:
        calls.append(1)
        return "hello world"

    ear = VoskHearing(model_path="m", device="plughw:2,0", session=session)

    assert ear.listen() == "hello world"
    assert ear.listen() == "hello world"
    assert len(calls) == 2


def test_vosk_hearing_swallows_errors_and_returns_empty() -> None:
    # Un fallo del microfono no debe tumbar a BMO: devuelve "" y sigue.
    def boom() -> str:
        raise RuntimeError("no mic")

    ear = VoskHearing(model_path="m", session=boom)

    assert ear.listen() == ""


def test_vosk_hearing_unavailable_when_model_missing(tmp_path) -> None:
    # Si el modelo no esta descargado, el oido NO esta disponible: el arranque
    # cae a teclado en vez de entrar en un bucle de fallos.
    ear = VoskHearing(model_path=str(tmp_path / "no-existe"))

    assert ear.available is False


def test_vosk_hearing_available_with_injected_session() -> None:
    # Con sesion inyectada (tests) se considera disponible sin tocar disco.
    ear = VoskHearing(model_path="m", session=lambda: "hi")

    assert ear.available is True
