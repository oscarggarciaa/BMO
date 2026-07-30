"""Tests de la voz de BMO: puerto NullVoice, integracion con Agent y PiperVoice."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from bmo.adapters.voice.piper_voice import PiperVoice, _aplay_command, _piper_command
from bmo.domain.agent import Agent
from bmo.domain.models import BrainDecision, Expression, ToolCall, Utterance
from bmo.ports.face import FacePort
from bmo.ports.voice import NullVoice, VoicePort
from bmo.tools.tool import Tool, ToolRegistry


class RecordingVoice(VoicePort):
    """Voz espia: guarda lo que BMO 'dice', en orden.

    Simula el arranque real del audio disparando `on_audio_start` cuando
    'reproduce', para poder verificar la sincronizacion con la cara.
    """

    def __init__(self) -> None:
        self.spoken: List[str] = []

    def speak(self, text: str, on_audio_start: Optional[Callable[[], None]] = None) -> None:
        self.spoken.append(text)
        if on_audio_start is not None:
            on_audio_start()


class MuteVoice(VoicePort):
    """Voz que 'habla' pero NUNCA dispara on_audio_start (simula fallo de audio)."""

    def speak(self, text: str, on_audio_start: Optional[Callable[[], None]] = None) -> None:
        return None


class RecordingFace(FacePort):
    """Cara espia: guarda las expresiones mostradas, en orden."""

    def __init__(self) -> None:
        self.shown: List[Expression] = []

    def show(self, expression: Expression) -> None:
        self.shown.append(expression)


class ScriptedBrain:
    def __init__(self, decisions: List[BrainDecision]) -> None:
        self._decisions = list(decisions)

    def decide(self, messages, tools) -> BrainDecision:
        return self._decisions.pop(0)


# --- puerto ------------------------------------------------------------------


def test_null_voice_speak_does_nothing() -> None:
    assert NullVoice().speak("hola") is None


def test_null_voice_fires_on_audio_start_immediately() -> None:
    # Sin audio real, la cara igual debe poder cambiar a TALKING: el Null
    # dispara el callback al instante para no romper la parte visual.
    fired: List[bool] = []
    NullVoice().speak("hola", on_audio_start=lambda: fired.append(True))
    assert fired == [True]


def test_agent_without_voice_uses_null_object() -> None:
    brain = ScriptedBrain([BrainDecision(reply=Utterance(text="ok", speaker="bmo"))])
    agent = Agent(brain=brain, tools=ToolRegistry())

    assert agent.ask("hola").text == "ok"


# --- integracion con el Agent ------------------------------------------------


def test_agent_speaks_the_brain_reply() -> None:
    voice = RecordingVoice()
    brain = ScriptedBrain(
        [BrainDecision(reply=Utterance(text="hi there!", speaker="bmo"))]
    )
    agent = Agent(brain=brain, tools=ToolRegistry(), voice=voice)

    agent.ask("hello")

    assert voice.spoken == ["hi there!"]


def test_agent_speaks_a_direct_tool_result() -> None:
    voice = RecordingVoice()
    registry = ToolRegistry()
    registry.register(
        Tool(name="say", description="", func=lambda **_: "direct answer", direct=True)
    )
    brain = ScriptedBrain([BrainDecision(tool_calls=(ToolCall(name="say"),))])
    agent = Agent(brain=brain, tools=registry, voice=voice)

    reply = agent.ask("x")

    assert reply.text == "direct answer"
    assert voice.spoken == ["direct answer"]


# --- sincronizacion cara <-> audio -------------------------------------------


def test_face_shows_talking_when_audio_starts() -> None:
    # La cara debe cambiar a TALKING cuando el audio arranca de verdad
    # (via on_audio_start), no antes de sintetizar.
    voice = RecordingVoice()  # dispara el callback al 'reproducir'
    face = RecordingFace()
    brain = ScriptedBrain([BrainDecision(reply=Utterance(text="hi", speaker="bmo"))])
    agent = Agent(brain=brain, tools=ToolRegistry(), face=face, voice=voice)

    agent.ask("hello")

    assert Expression.TALKING in face.shown


def test_face_does_not_show_talking_before_audio_starts() -> None:
    # Si el audio nunca arranca (no se dispara on_audio_start), la cara NO
    # debe quedar en TALKING: se prueba que ya no hay un show(TALKING) suelto.
    voice = MuteVoice()  # nunca dispara el callback
    face = RecordingFace()
    brain = ScriptedBrain([BrainDecision(reply=Utterance(text="hi", speaker="bmo"))])
    agent = Agent(brain=brain, tools=ToolRegistry(), face=face, voice=voice)

    agent.ask("hello")

    assert Expression.TALKING not in face.shown


# --- PiperVoice adapter ------------------------------------------------------


def test_piper_skips_empty_text() -> None:
    calls: List[Tuple[Any, ...]] = []
    voice = PiperVoice(model_path="m.onnx", run=lambda *a: calls.append(a))

    voice.speak("   ")

    assert calls == []


def test_piper_delegates_to_run_with_stripped_text() -> None:
    calls: List[Tuple[Any, ...]] = []
    cb: Callable[[], None] = lambda: None
    voice = PiperVoice(
        model_path="m.onnx",
        device="plughw:1,0",
        sample_rate=22050,
        run=lambda *a: calls.append(a),
    )

    voice.speak("  hello  ", on_audio_start=cb)

    assert calls == [("hello", "m.onnx", "plughw:1,0", 22050, cb)]


def test_piper_swallows_run_errors() -> None:
    def boom(*_: Any) -> None:
        raise RuntimeError("no audio device")

    voice = PiperVoice(model_path="m.onnx", run=boom)

    voice.speak("hi")  # no debe lanzar: un fallo de audio no tumba la charla


def test_aplay_command_includes_device_when_set() -> None:
    cmd = _aplay_command("plughw:1,0", 22050)
    assert "-D" in cmd and "plughw:1,0" in cmd
    assert "S16_LE" in cmd and "22050" in cmd


def test_aplay_command_omits_device_when_empty() -> None:
    assert "-D" not in _aplay_command("", 22050)


def test_piper_command_has_model_and_output_raw() -> None:
    cmd = _piper_command("voice.onnx")
    assert "voice.onnx" in cmd and "--output-raw" in cmd
