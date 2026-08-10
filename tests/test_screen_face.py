"""Tests del adapter de pantalla (TkFace).

Probamos solo la logica que NO necesita un display real: el estado de la
expresion actual (thread-safe) y el descubrimiento de frames en disco. El
loop de tkinter y la carga con PIL se prueban a mano en la Pi, no aca.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from bmo.domain.models import Expression, Note
from bmo.interfaces.screen.tk_face import TkFace, format_note_preview
from bmo.ports.face import FacePort

_FACES_DIR = Path(__file__).resolve().parents[1] / "bmo" / "interfaces" / "screen" / "faces"


def test_tk_face_is_a_face_port() -> None:
    assert isinstance(TkFace(faces_dir=_FACES_DIR), FacePort)


def test_show_updates_current_expression() -> None:
    face = TkFace(faces_dir=_FACES_DIR)

    face.show(Expression.HAPPY)

    assert face.current is Expression.HAPPY


def test_notes_provider_is_stored_and_used() -> None:
    face = TkFace(faces_dir=_FACES_DIR)
    note = Note(title="Milk", content="Buy milk", created_at=datetime.now())
    face.set_notes_provider(lambda: [note])

    assert face._notes_to_show() == [note]


def test_notes_to_show_is_empty_without_provider() -> None:
    face = TkFace(faces_dir=_FACES_DIR)

    assert face._notes_to_show() == []


def test_notes_to_show_survives_a_failing_provider() -> None:
    face = TkFace(faces_dir=_FACES_DIR)

    def boom() -> list:
        raise RuntimeError("disk on fire")

    face.set_notes_provider(boom)

    assert face._notes_to_show() == []


def test_handle_touch_without_window_is_noop() -> None:
    face = TkFace(faces_dir=_FACES_DIR)
    face.set_notes_provider(lambda: [])

    # sin ventana (tests headless) tocar no debe intentar dibujar ni romper
    assert face._handle_touch() is None
    assert face.menu_open is False


def test_format_note_preview_flattens_and_truncates() -> None:
    note = Note(
        title="t",
        content="line one\n   line two   with   spaces",
        created_at=datetime.now(),
    )

    assert format_note_preview(note) == "line one line two with spaces"


def test_format_note_preview_truncates_long_bodies() -> None:
    note = Note(title="t", content="x" * 500, created_at=datetime.now())

    preview = format_note_preview(note, max_len=10)

    assert preview == "x" * 10 + "…"



def test_frames_for_returns_sorted_pngs() -> None:
    face = TkFace(faces_dir=_FACES_DIR)

    frames = face._frames_for(Expression.THINKING)

    names = [p.name for p in frames]
    assert names == ["01.png", "02.png", "03.png", "04.png"]


def test_frames_for_single_frame_expression() -> None:
    face = TkFace(faces_dir=_FACES_DIR)

    frames = face._frames_for(Expression.NEUTRAL)

    assert len(frames) == 1


def test_frames_for_unknown_folder_is_empty(tmp_path: Path) -> None:
    face = TkFace(faces_dir=tmp_path)

    assert face._frames_for(Expression.HAPPY) == []


def test_lifecycle_methods_default_to_noop_on_null_face() -> None:
    from bmo.ports.face import NullFace

    face = NullFace()

    assert face.start() is None
    assert face.run() is None
    assert face.stop() is None


def test_available_is_false_before_start() -> None:
    face = TkFace(faces_dir=_FACES_DIR)

    assert face.available is False


def test_run_is_noop_before_start() -> None:
    face = TkFace(faces_dir=_FACES_DIR)

    assert face.run() is None


def test_start_survives_when_display_unavailable(monkeypatch) -> None:
    import tkinter

    def boom(*_args, **_kwargs):
        raise tkinter.TclError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(tkinter, "Tk", boom)
    face = TkFace(faces_dir=_FACES_DIR)

    face.start()

    assert face.available is False

