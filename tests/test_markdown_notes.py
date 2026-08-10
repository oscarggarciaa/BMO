"""Tests del adapter de notas markdown (MarkdownNotes)."""

from __future__ import annotations

from pathlib import Path

from bmo.adapters.memory.markdown_notes import MarkdownNotes
from bmo.domain.models import Note
from bmo.ports.notes import NotesPort


def test_markdown_notes_is_a_notes_port(tmp_path: Path) -> None:
    assert isinstance(MarkdownNotes(tmp_path), NotesPort)


def test_save_creates_a_markdown_file(tmp_path: Path) -> None:
    notes = MarkdownNotes(tmp_path)

    note = notes.save("Buy milk on the way home")

    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    assert isinstance(note, Note)
    assert note.content == "Buy milk on the way home"


def test_save_creates_directory_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "does" / "not" / "exist"
    notes = MarkdownNotes(target)

    notes.save("hello")

    assert list(target.glob("*.md"))


def test_save_derives_title_from_first_line(tmp_path: Path) -> None:
    notes = MarkdownNotes(tmp_path)

    note = notes.save("Remember the milk\nand also eggs")

    assert note.title == "Remember the milk"


def test_all_returns_saved_notes(tmp_path: Path) -> None:
    notes = MarkdownNotes(tmp_path)
    notes.save("first note")
    notes.save("second note")

    contents = {n.content for n in notes.all()}

    assert contents == {"first note", "second note"}


def test_all_is_empty_when_directory_missing(tmp_path: Path) -> None:
    notes = MarkdownNotes(tmp_path / "nope")

    assert notes.all() == []


def test_all_sorts_newest_first(tmp_path: Path) -> None:
    notes = MarkdownNotes(tmp_path)
    notes.save("older")
    newer = notes.save("newer")

    # empatar la fecha por segundo no debe importar: el orden lo fija created_at
    result = notes.all()

    assert result[0].created_at >= result[-1].created_at
    assert newer.content in {n.content for n in result}


def test_roundtrip_preserves_title_and_content(tmp_path: Path) -> None:
    notes = MarkdownNotes(tmp_path)
    saved = notes.save("Groceries\nmilk, eggs, bread")

    loaded = notes.all()[0]

    assert loaded.title == saved.title
    assert loaded.content == saved.content


def test_empty_note_falls_back_to_untitled(tmp_path: Path) -> None:
    notes = MarkdownNotes(tmp_path)

    note = notes.save("   ")

    assert note.title == "Untitled note"


def test_two_notes_with_same_title_do_not_overwrite(tmp_path: Path) -> None:
    notes = MarkdownNotes(tmp_path)
    notes.save("duplicate")
    notes.save("duplicate")

    assert len(list(tmp_path.glob("*.md"))) == 2
