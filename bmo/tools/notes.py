"""Tools de notas: BMO apunta (`save_note`) y lee (`read_notes`)."""

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
            "Store NEW information the user gives you. Use this ONLY when the "
            "user TELLS or COMMANDS you to remember, save, note or write down "
            "something, as a statement (for example 'remember to buy milk', "
            "'note that the door is broken'). NEVER use it for a QUESTION like "
            "'what do I have to remember' or 'what did I save': those are "
            "list_notes or recall_note. Pass the text to store as `content`."
        ),
        func=save_note,
        parameters={
            "content": {
                "type": "string",
                "description": "The full text of the note to remember.",
            },
        },
        # directa: la confirmación ES la respuesta; el modelo NO vuelve a decidir
        # (si no, un modelo pequeño encadena tools extra tras guardar).
        direct=True,
    )


def build_read_notes_tool(notes: NotesPort) -> Tool:
    """Arma la tool `read_notes`: con tema busca; sin tema, devuelve todas.

    Fusiona 'buscar por tema' y 'listar todo' en UNA sola tool, porque un
    modelo pequeño no distingue fiablemente entre las dos.
    """

    def read_notes(query: str = "", question: str = "") -> str:
        topic = (query or "").strip()
        if topic:
            found = notes.search(topic, limit=5)
            if not found:
                return "I don't have any note about that."
        else:
            found = notes.all()
            if not found:
                return "You don't have any notes yet."
        lines = [f"- {note.content}" for note in found]
        return "Here is what I remember:\n" + "\n".join(lines)

    return Tool(
        name="read_notes",
        description=(
            "Read the user's saved notes. Use this for ANY QUESTION about what "
            "the user has saved, has to remember, or what is on their list "
            "(for example 'what do I have to remember', 'what did I ask you to "
            "buy', 'tell me everything', 'what is on my list'). Give a `query` "
            "with a topic to search, or leave it EMPTY to read ALL notes. "
            "NEVER use it to store new info: that is save_note."
        ),
        func=read_notes,
        parameters={
            "query": {
                "type": "string",
                "description": (
                    "Topic to search for; leave empty to read all notes."
                ),
            },
        },
        # directa: devuelve las notas tal cual, sin que el modelo las resuma ni
        # invente (la persona lo obliga a responder corto).
        direct=True,
    )
