"""Tests de la cara: el Agent emite las expresiones correctas en cada fase."""

from __future__ import annotations

from typing import List

from bmo.domain.agent import Agent
from bmo.domain.models import BrainDecision, Expression, ToolCall, Utterance
from bmo.ports.face import FacePort, NullFace
from bmo.tools.tool import Tool, ToolRegistry


class RecordingFace(FacePort):
    """Cara espia: guarda las expresiones que le mandan, en orden."""

    def __init__(self) -> None:
        self.calls: List[Expression] = []

    def show(self, expression: Expression) -> None:
        self.calls.append(expression)


class ScriptedBrain:
    """Brain falso: devuelve decisiones de una lista, una por llamada."""

    def __init__(self, decisions: List[BrainDecision]) -> None:
        self._decisions = list(decisions)

    def decide(self, messages, tools) -> BrainDecision:
        return self._decisions.pop(0)


def _registry_with_look() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(name="look", description="mira", func=lambda: "veo: nada"))
    return registry


def test_null_face_show_does_nothing() -> None:
    assert NullFace().show(Expression.HAPPY) is None


def test_agent_without_face_uses_null_object() -> None:
    brain = ScriptedBrain([BrainDecision(reply=Utterance(text="ok", speaker="bmo"))])
    agent = Agent(brain=brain, tools=ToolRegistry())

    reply = agent.ask("hola")

    assert reply.text == "ok"


def test_emits_thinking_then_talking_on_direct_reply() -> None:
    face = RecordingFace()
    brain = ScriptedBrain(
        [BrainDecision(reply=Utterance(text="hola!", speaker="bmo"))]
    )
    agent = Agent(brain=brain, tools=ToolRegistry(), face=face)

    agent.ask("hola")

    assert face.calls == [Expression.THINKING, Expression.TALKING]


def test_thinking_first_and_talking_last_when_using_a_tool() -> None:
    face = RecordingFace()
    brain = ScriptedBrain(
        [
            BrainDecision(tool_calls=(ToolCall(name="look"),)),
            BrainDecision(reply=Utterance(text="veo una cara", speaker="bmo")),
        ]
    )
    agent = Agent(brain=brain, tools=_registry_with_look(), face=face)

    agent.ask("que ves?")

    assert face.calls[0] == Expression.THINKING
    assert face.calls[-1] == Expression.TALKING


def test_emits_sad_on_safety_cutoff() -> None:
    face = RecordingFace()
    brain = ScriptedBrain(
        [BrainDecision(tool_calls=(ToolCall(name="look"),)) for _ in range(10)]
    )
    agent = Agent(brain=brain, tools=_registry_with_look(), face=face, max_steps=3)

    agent.ask("loop infinito")

    assert face.calls[-1] == Expression.SAD
