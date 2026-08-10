"""Tool `save_note`: BMO apunta una nota cuando el usuario se lo pide."""

from __future__ import annotations

from bmo.ports.notes import NotesPort
from bmo.tools.tool import Tool


def build_save_note_tool(notes: NotesPort) -> Tool:
    """Arma la tool `save_note` cableando un almacén de notas concreto."""

    def save_note(content: str = "", question: str = "") -> str:
        # content = lo que el modelo extrajo; question = el mensaje completo (respaldo)
        text = (content or "").strip() or (question or "").strip()
        if not text:
            return "I couldn't save the note because it was empty."
        note = notes.save(text)
        return f"Saved a note: '{note.title}'."

    return Tool(
        name="save_note",
        description=(
            "Write down and remember a note for the user. Use this whenever the "
            "user asks BMO to remember something, take a note, write something "
            "down, jot it down, or save an idea, a reminder, or a task. Pass the "
            "full text to remember as `content`."
        ),
        func=save_note,
        parameters={
            "content": {
                "type": "string",
                "description": "The full text of the note to remember.",
            },
        },
        direct=False,
    )
