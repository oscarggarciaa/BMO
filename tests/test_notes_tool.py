"""Tests de la tool save_note: guardar notas por decisión del cerebro."""

from __future__ import annotations

from datetime import datetime
from typing import List

from bmo.domain.agent import Agent
from bmo.domain.models import BrainDecision, Note, ToolCall, Utterance
from bmo.ports.notes import NotesPort
from bmo.tools.notes import build_save_note_tool
from bmo.tools.tool import ToolRegistry


class FakeNotes(NotesPort):
    """Almacén de notas en memoria: registra lo que se guarda."""

    def __init__(self) -> None:
        self.saved: List[str] = []

    def save(self, content: str, title: str = "") -> Note:
        self.saved.append(content)
        return Note(
            title=title or content[:20],
            content=content,
            created_at=datetime.now(),
        )

    def all(self) -> List[Note]:
        return []


class ScriptedBrain:
    """Cerebro con decisiones prefijadas para tests end-to-end."""

    def __init__(self, decisions: List[BrainDecision]) -> None:
        self._decisions = list(decisions)

    def decide(self, messages, tools) -> BrainDecision:
        return self._decisions.pop(0)


def test_save_note_tool_is_named_save_note() -> None:
    tool = build_save_note_tool(FakeNotes())
    assert tool.name == "save_note"


def test_save_note_tool_is_not_direct() -> None:
    # no es directa: BMO narra la confirmación en su personalidad
    tool = build_save_note_tool(FakeNotes())
    assert tool.direct is False


def test_save_note_persists_the_content() -> None:
    notes = FakeNotes()
    tool = build_save_note_tool(notes)

    tool.run(content="buy milk")

    assert notes.saved == ["buy milk"]


def test_save_note_confirms_with_the_title() -> None:
    notes = FakeNotes()
    tool = build_save_note_tool(notes)

    result = tool.run(content="buy milk")

    assert "buy milk" in result


def test_save_note_rejects_empty_content() -> None:
    notes = FakeNotes()
    tool = build_save_note_tool(notes)

    result = tool.run(content="   ")

    assert notes.saved == []
    assert "empty" in result.lower()


def test_save_note_prefers_content_over_the_raw_message() -> None:
    notes = FakeNotes()
    tool = build_save_note_tool(notes)

    tool.run(content="buy milk", question="hey BMO please remember buy milk")

    assert notes.saved == ["buy milk"]


def test_save_note_falls_back_to_the_raw_message() -> None:
    # Si el modelo dispara la accion sin 'content', se guarda el mensaje completo
    # en vez de perder la nota.
    notes = FakeNotes()
    tool = build_save_note_tool(notes)

    tool.run(question="remember to water the plants")

    assert notes.saved == ["remember to water the plants"]


def test_agent_saves_a_note_end_to_end() -> None:
    # El cerebro decide save_note con contenido; el agente lo ejecuta y la nota
    # llega REALMENTE al almacen (esta era la parte rota).
    notes = FakeNotes()
    registry = ToolRegistry()
    registry.register(build_save_note_tool(notes))
    brain = ScriptedBrain(
        [
            BrainDecision(
                tool_calls=(ToolCall(name="save_note", arguments={"content": "buy milk"}),)
            ),
            BrainDecision(reply=Utterance(text="Done, I'll remember it!", speaker="bmo")),
        ]
    )
    agent = Agent(brain=brain, tools=registry)

    agent.ask("remember to buy milk")

    assert notes.saved == ["buy milk"]


def test_save_note_schema_requires_content() -> None:
    tool = build_save_note_tool(FakeNotes())

    schema = tool.to_schema()["function"]

    assert "content" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["content"]
