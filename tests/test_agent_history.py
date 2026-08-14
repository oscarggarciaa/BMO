"""Tests de la ventana de historial del Agent.

Enviar TODO el historial en cada turno llena la caché de conversación del NPU
Hailo y el modelo empieza a alucinar. El Agent debe poder limitar el historial
a una ventana (system prompt + últimos N mensajes), recortando al INICIO de cada
turno para no romper el flujo de tools dentro del turno.
"""

from __future__ import annotations

from typing import List

from bmo.domain.agent import Agent
from bmo.domain.models import BrainDecision, ToolCall, Utterance
from bmo.tools.tool import Tool, ToolRegistry


class EchoBrain:
    """Cerebro que siempre responde 'ok' sin usar tools."""

    def decide(self, messages, tools) -> BrainDecision:
        return BrainDecision(reply=Utterance(text="ok", speaker="bmo"))


class ScriptedBrain:
    def __init__(self, decisions: List[BrainDecision]) -> None:
        self._decisions = list(decisions)

    def decide(self, messages, tools) -> BrainDecision:
        return self._decisions.pop(0)


def test_history_is_bounded_across_turns() -> None:
    agent = Agent(
        brain=EchoBrain(), tools=ToolRegistry(), system_prompt="SYS", max_history=2
    )

    for i in range(6):
        agent.ask(f"msg {i}")

    contents = [m.content for m in agent.history]
    assert "SYS" in contents  # el system prompt SIEMPRE se conserva
    assert "msg 0" not in contents  # los turnos viejos se descartan
    assert "msg 5" in contents  # los recientes se conservan
    non_system = [m for m in agent.history if m.role != "system"]
    assert len(non_system) <= 4  # acotado: no crece con el nº de turnos


def test_history_unlimited_by_default() -> None:
    agent = Agent(brain=EchoBrain(), tools=ToolRegistry(), system_prompt="SYS")

    for i in range(5):
        agent.ask(f"msg {i}")

    contents = [m.content for m in agent.history]
    assert "msg 0" in contents  # sin límite: no se descarta nada


def test_tool_flow_works_even_with_a_tiny_window() -> None:
    # Recortar SOLO al inicio del turno: los mensajes de la tool se añaden
    # después, así el brain los ve y puede responder aunque la ventana sea 1.
    registry = ToolRegistry()
    registry.register(
        Tool(name="look", description="", func=lambda **_: "veo: nada", direct=False)
    )
    brain = ScriptedBrain(
        [
            BrainDecision(tool_calls=(ToolCall(name="look"),)),
            BrainDecision(reply=Utterance(text="nothing", speaker="bmo")),
        ]
    )
    agent = Agent(
        brain=brain, tools=registry, system_prompt="SYS", max_history=1
    )

    reply = agent.ask("what do you see")

    assert reply.text == "nothing"
