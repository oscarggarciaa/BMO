"""Tests de la tool read_notes: con tema busca; sin tema, devuelve todas."""

from __future__ import annotations

from datetime import datetime
from typing import List

from bmo.domain.agent import Agent
from bmo.domain.models import BrainDecision, Note, ToolCall, Utterance
from bmo.ports.notes import NotesPort
from bmo.tools.notes import build_read_notes_tool
from bmo.tools.tool import ToolRegistry


class FakeNotes(NotesPort):
    """Almacén de notas en memoria; usa el search por defecto del puerto."""

    def __init__(self) -> None:
        self._notes: List[Note] = []

    def save(self, content: str, title: str = "") -> Note:
        note = Note(
            title=title or content[:20],
            content=content,
            created_at=datetime.now(),
        )
        self._notes.append(note)
        return note

    def all(self) -> List[Note]:
        return list(reversed(self._notes))  # más nueva primero

    def delete(self, note: Note) -> None:
        self._notes = [n for n in self._notes if n.content != note.content]


class ScriptedBrain:
    def __init__(self, decisions: List[BrainDecision]) -> None:
        self._decisions = list(decisions)

    def decide(self, messages, tools) -> BrainDecision:
        return self._decisions.pop(0)


def test_read_notes_tool_is_named_read_notes() -> None:
    tool = build_read_notes_tool(FakeNotes())
    assert tool.name == "read_notes"


def test_read_notes_tool_is_direct() -> None:
    # directa: devuelve las notas tal cual, sin que el modelo las resuma/invente
    tool = build_read_notes_tool(FakeNotes())
    assert tool.direct is True


def test_read_notes_without_query_returns_every_note() -> None:
    notes = FakeNotes()
    notes.save("buy milk")
    notes.save("call the dentist")
    tool = build_read_notes_tool(notes)

    result = tool.run()

    assert "buy milk" in result
    assert "call the dentist" in result


def test_read_notes_with_query_searches_by_topic() -> None:
    notes = FakeNotes()
    notes.save("buy milk and eggs")
    notes.save("call the dentist on monday")
    tool = build_read_notes_tool(notes)

    result = tool.run(query="milk")

    assert "milk" in result.lower()
    assert "dentist" not in result.lower()


def test_read_notes_reports_when_topic_has_no_match() -> None:
    notes = FakeNotes()
    notes.save("buy milk")
    tool = build_read_notes_tool(notes)

    result = tool.run(query="dinosaurs")

    assert "milk" not in result.lower()


def test_read_notes_reports_when_empty() -> None:
    tool = build_read_notes_tool(FakeNotes())

    result = tool.run()

    assert isinstance(result, str) and result
    assert "buy" not in result.lower()


def test_agent_reads_notes_end_to_end() -> None:
    # read_notes es direct=True: su salida ES la respuesta final del agente.
    notes = FakeNotes()
    notes.save("buy milk")
    registry = ToolRegistry()
    registry.register(build_read_notes_tool(notes))
    brain = ScriptedBrain(
        [BrainDecision(tool_calls=(ToolCall(name="read_notes", arguments={}),))]
    )
    agent = Agent(brain=brain, tools=registry)

    reply = agent.ask("what do I have to remember?")

    assert "buy milk" in reply.text
