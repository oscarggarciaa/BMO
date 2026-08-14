"""Tests del guard anti-loop del Agent.

Un modelo pequeño (qwen3:1.7b en Hailo) a veces NO para: pide tools una y otra
vez sin llegar a responder, agotando max_steps y disparando el corte de
seguridad feo. El Agent debe cortar en cuanto detecta que se repite EXACTAMENTE
la misma llamada (mismo nombre + mismos argumentos) y responder con lo que ya
tiene, en vez de gastar todos los pasos.
"""

from __future__ import annotations

from typing import List

from bmo.domain.agent import Agent
from bmo.domain.models import BrainDecision, ToolCall, Utterance
from bmo.tools.tool import Tool, ToolRegistry


class LoopBrain:
    """Cerebro que SIEMPRE pide la misma tool: simula un modelo que no para."""

    def __init__(self, call: ToolCall) -> None:
        self._call = call
        self.calls = 0

    def decide(self, messages, tools) -> BrainDecision:
        self.calls += 1
        return BrainDecision(tool_calls=(self._call,))


class ScriptedBrain:
    """Cerebro con decisiones prefijadas."""

    def __init__(self, decisions: List[BrainDecision]) -> None:
        self._decisions = list(decisions)

    def decide(self, messages, tools) -> BrainDecision:
        return self._decisions.pop(0)


def _echo_tool() -> Tool:
    return Tool(name="ping", description="", func=lambda **_: "pong", direct=False)


def test_agent_stops_when_the_brain_repeats_the_same_tool_call() -> None:
    registry = ToolRegistry()
    registry.register(_echo_tool())
    brain = LoopBrain(ToolCall(name="ping", arguments={}))
    agent = Agent(brain=brain, tools=registry, max_steps=5)

    reply = agent.ask("hi")

    # respondió con el resultado de la tool, NO con el corte de seguridad
    assert reply.text == "pong"
    # cortó el loop pronto en vez de agotar los 5 pasos
    assert brain.calls <= 2


def test_agent_does_not_hit_the_safety_cutoff_on_a_loop() -> None:
    registry = ToolRegistry()
    registry.register(_echo_tool())
    brain = LoopBrain(ToolCall(name="ping", arguments={}))
    agent = Agent(brain=brain, tools=registry, max_steps=5)

    reply = agent.ask("hi")

    assert "corte de seguridad" not in reply.text
    assert "agotaron" not in reply.text


def test_agent_runs_two_distinct_tool_calls_then_replies() -> None:
    # El guard SOLO corta ante duplicados EXACTOS: dos llamadas con args
    # distintos son legítimas y deben ejecutarse ambas.
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="note",
            description="",
            func=lambda content="", **_: f"saved {content}",
            parameters={"content": {"type": "string", "description": ""}},
            direct=False,
        )
    )
    brain = ScriptedBrain(
        [
            BrainDecision(tool_calls=(ToolCall(name="note", arguments={"content": "a"}),)),
            BrainDecision(tool_calls=(ToolCall(name="note", arguments={"content": "b"}),)),
            BrainDecision(reply=Utterance(text="done", speaker="bmo")),
        ]
    )
    agent = Agent(brain=brain, tools=registry, max_steps=5)

    reply = agent.ask("hi")

    assert reply.text == "done"
