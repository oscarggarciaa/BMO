"""Tests del oido de BMO: escuchar por microfono con wake-word 'hello'.

El diseno separa dos responsabilidades:
- ADAPTER (VoskHearing): audio del microfono -> texto. Nada mas.
- DOMINIO (WakeWord): decide si el texto despierta a BMO. Puro y testeable.
"""

from __future__ import annotations

from typing import Callable, List

from bmo.adapters.hearing.vosk_hearing import VoskHearing, _capture_phrase
from bmo.domain.models import Utterance
from bmo.domain.wakeword import WakeWord
from bmo.interfaces.microphone import listen_loop
from bmo.ports.hearing import HearingPort, NullHearing
import json


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


def test_listen_loop_prints_clean_user_and_bmo_lines(capsys) -> None:
    # En modo normal la entrada/salida se muestra con print limpio (USER>/BMO>),
    # sin timestamps ni logger, y sin duplicar la respuesta.
    agent = SpyAgent()
    hearing = ScriptedHearing(["hello how are you"])

    listen_loop(agent, hearing, WakeWord())

    out = capsys.readouterr().out
    assert "USER> how are you" in out
    assert "BMO> ok" in out


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


# --- _capture_phrase: leer UNA frase de un stream + recognizer -------------


class _FakeRecognizer:
    """Recognizer Vosk falso: acepta la frase tras N bloques leidos."""

    def __init__(self, accept_on: int, text: str = "hello world") -> None:
        self._accept_on = accept_on
        self._text = text
        self._reads = 0

    def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802 - API de Vosk
        self._reads += 1
        return self._reads >= self._accept_on

    def Result(self) -> str:  # noqa: N802 - API de Vosk
        return json.dumps({"text": self._text})


class _FakeStream:
    """Stream de audio falso: entrega bloques y luego b'' (fin)."""

    def __init__(self, chunks: List[bytes]) -> None:
        self._chunks = list(chunks)

    def read(self, _n: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""


def test_capture_phrase_returns_text_when_recognizer_accepts() -> None:
    # Lee bloques hasta que el recognizer detecta fin de frase y transcribe.
    stream = _FakeStream([b"aaaa", b"bbbb"])
    recognizer = _FakeRecognizer(accept_on=2)

    assert _capture_phrase(stream, recognizer, chunk_bytes=4) == "hello world"


def test_capture_phrase_returns_empty_when_stream_ends() -> None:
    # Si el stream se corta antes de detectar la frase, devuelve "" sin colgarse.
    stream = _FakeStream([])
    recognizer = _FakeRecognizer(accept_on=99)

    assert _capture_phrase(stream, recognizer, chunk_bytes=4) == ""
