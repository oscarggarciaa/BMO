"""Tests de la tool look: pregunta hacia la vision y respuesta directa."""

from __future__ import annotations

from typing import List

from bmo.domain.agent import Agent
from bmo.domain.models import BrainDecision, Perception, ToolCall
from bmo.ports.vision import VisionPort
from bmo.tools.look import build_look_tool
from bmo.tools.tool import Tool, ToolRegistry


class FakeCamera:
    def __init__(self, frame: str = "frame") -> None:
        self._frame = frame

    def start(self) -> None: ...

    def capture(self):
        return self._frame

    def stop(self) -> None: ...


class RecordingVision(VisionPort):
    """Vision espia: guarda la pregunta que recibe y devuelve una descripcion fija."""

    def __init__(self, description: str = "veo una persona con lentes") -> None:
        self.questions: List[str] = []
        self._description = description

    def analyze(self, frame, question: str = "") -> Perception:
        self.questions.append(question)
        return Perception(description=self._description)


class ScriptedBrain:
    def __init__(self, decisions: List[BrainDecision]) -> None:
        self._decisions = list(decisions)
        self.calls = 0

    def decide(self, messages, tools) -> BrainDecision:
        self.calls += 1
        return self._decisions.pop(0)


def _registry(camera, vision) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_look_tool(camera, vision))
    return registry


def test_tool_run_ignores_unexpected_kwargs() -> None:
    tool = Tool(name="ping", description="", func=lambda: "pong")
    assert tool.run(question="lo que sea") == "pong"


def test_tool_direct_defaults_false() -> None:
    tool = Tool(name="ping", description="", func=lambda: "pong")
    assert tool.direct is False


def test_look_tool_is_direct() -> None:
    tool = build_look_tool(FakeCamera(), RecordingVision())
    assert tool.direct is True


def test_look_passes_question_to_vision() -> None:
    vision = RecordingVision()
    tool = build_look_tool(FakeCamera(), vision)
    tool.run(question="hay alguien con barba?")
    assert vision.questions == ["hay alguien con barba?"]


def test_direct_tool_result_is_the_reply_without_second_brain_call() -> None:
    vision = RecordingVision(description="veo una silla azul")
    brain = ScriptedBrain([BrainDecision(tool_calls=(ToolCall(name="look"),))])
    agent = Agent(brain=brain, tools=_registry(FakeCamera(), vision))

    reply = agent.ask("que ves?")

    assert reply.text == "veo una silla azul"
    assert brain.calls == 1


def test_agent_forwards_user_question_to_look() -> None:
    vision = RecordingVision()
    brain = ScriptedBrain([BrainDecision(tool_calls=(ToolCall(name="look"),))])
    agent = Agent(brain=brain, tools=_registry(FakeCamera(), vision))

    agent.ask("hay un libro rojo?")

    assert vision.questions == ["hay un libro rojo?"]
