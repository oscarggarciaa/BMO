"""Tools de notas: apunta (`save_note`), recuerda (`recall_note`), lista (`list_notes`)."""

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
            "Write down and remember NEW information for the user. Use this "
            "whenever the user asks BMO to remember something, take a note, "
            "write something down, jot it down, or save an idea, a reminder or "
            "a task (for example 'remember to buy milk'). After saving, just "
            "confirm in one short sentence: do NOT look it up. Pass the full "
            "text to remember as `content`."
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


def build_recall_note_tool(notes: NotesPort) -> Tool:
    """Arma la tool `recall_note` cableando un almacén de notas concreto."""

    def recall_note(query: str = "", question: str = "") -> str:
        # query = lo que el modelo extrajo; question = el mensaje completo (respaldo)
        text = (query or "").strip() or (question or "").strip()
        if not text:
            return "I don't know what to look for in my notes."
        found = notes.search(text, limit=3)
        if not found:
            return "I don't have any note about that."
        lines = [f"- {note.content}" for note in found]
        return "Here is what I remember:\n" + "\n".join(lines)

    return Tool(
        name="recall_note",
        description=(
            "Look up notes the user asked BMO to remember EARLIER. Use this "
            "ONLY when the user ASKS about the past: what BMO has saved, what "
            "is on their list or reminders, or a question a past note could "
            "answer (for example 'what did I ask you to buy', 'what do you "
            "remember', 'what is on my list'). Do NOT use it when the user is "
            "giving you something NEW to remember: that is save_note. Pass the "
            "keywords or topic to look for as `query`."
        ),
        func=recall_note,
        parameters={
            "query": {
                "type": "string",
                "description": "Keywords or topic to search for in the saved notes.",
            },
        },
        direct=False,
    )


def build_list_notes_tool(notes: NotesPort) -> Tool:
    """Arma la tool `list_notes` cableando un almacén de notas concreto."""

    def list_notes(question: str = "") -> str:
        found = notes.all()
        if not found:
            return "You don't have any notes yet."
        lines = [f"- {note.content}" for note in found]
        return "Here is everything I remember:\n" + "\n".join(lines)

    return Tool(
        name="list_notes",
        description=(
            "List ALL the notes the user asked BMO to remember. Use this when "
            "the user asks what they have to remember, what is on their list, "
            "to tell them everything, or to list all their notes, reminders or "
            "tasks (for example 'what do I have to remember', 'tell me "
            "everything', 'what is on my list'). Takes no arguments."
        ),
        func=list_notes,
        parameters={},
        direct=True,
    )
