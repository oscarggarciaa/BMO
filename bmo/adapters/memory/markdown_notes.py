"""Adapter de notas: guarda cada nota como un fichero markdown (estilo Obsidian).

Una nota = un fichero `.md` en una carpeta. El fichero lleva un pequeño
frontmatter YAML con el título y la fecha, y debajo el cuerpo de la nota. Así
las notas son legibles y editables a mano, e incluso se pueden abrir con Obsidian.

    ---
    title: Buy milk
    created: 2026-08-10T14:30:00
    ---
    Buy milk on the way home.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from bmo.domain.models import Note
from bmo.ports.notes import NotesPort

_LOG = logging.getLogger(__name__)
_FRONTMATTER = "---"
_MAX_TITLE_LEN = 60
_SLUG_LEN = 40


class MarkdownNotes(NotesPort):
    """Persiste las notas como ficheros markdown en una carpeta."""

    def __init__(self, directory: Path | str) -> None:
        self._dir = Path(directory)

    def save(self, content: str, title: str = "") -> Note:
        content = (content or "").strip()
        title = (title or "").strip() or _derive_title(content)
        note = Note(title=title, content=content, created_at=datetime.now())

        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._unique_path(note)
        path.write_text(_render(note), encoding="utf-8")
        return note

    def all(self) -> List[Note]:
        if not self._dir.is_dir():
            return []
        notes: List[Note] = []
        for path in self._dir.glob("*.md"):
            note = _parse(path)
            if note is not None:
                notes.append(note)
        notes.sort(key=lambda n: n.created_at, reverse=True)
        return notes

    def delete(self, note: Note) -> None:
        if not note.id:
            return
        path = Path(note.id)
        # seguridad: solo se borra dentro de la carpeta de notas (anti traversal)
        try:
            path.resolve().relative_to(self._dir.resolve())
        except ValueError:
            _LOG.warning("se ignoró un borrado fuera de la carpeta de notas: %s", path)
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            _LOG.warning("no se pudo borrar la nota %s", path)

    def _unique_path(self, note: Note) -> Path:
        """Nombre de fichero legible y único: fecha + slug del título."""
        stamp = note.created_at.strftime("%Y%m%d-%H%M%S")
        slug = _slugify(note.title)
        base = f"{stamp}-{slug}" if slug else stamp
        path = self._dir / f"{base}.md"
        n = 2
        while path.exists():
            path = self._dir / f"{base}-{n}.md"
            n += 1
        return path


def _derive_title(content: str) -> str:
    """Título por defecto: la primera línea recortada."""
    first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
    if not first_line:
        return "Nota sin título"
    if len(first_line) <= _MAX_TITLE_LEN:
        return first_line
    return first_line[:_MAX_TITLE_LEN].rstrip() + "…"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:_SLUG_LEN].strip("-")


def _render(note: Note) -> str:
    return (
        f"{_FRONTMATTER}\n"
        f"title: {note.title}\n"
        f"created: {note.created_at.isoformat(timespec='seconds')}\n"
        f"{_FRONTMATTER}\n"
        f"{note.content}\n"
    )


def _parse(path: Path) -> Optional[Note]:
    """Lee un fichero markdown y reconstruye la nota. Tolera formatos raros."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        _LOG.warning("no se pudo leer la nota %s", path)
        return None

    title = ""
    created: Optional[datetime] = None
    body = raw

    lines = raw.splitlines()
    if lines and lines[0].strip() == _FRONTMATTER:
        end = next(
            (i for i in range(1, len(lines)) if lines[i].strip() == _FRONTMATTER),
            None,
        )
        if end is not None:
            for line in lines[1:end]:
                key, _, value = line.partition(":")
                key, value = key.strip().lower(), value.strip()
                if key == "title":
                    title = value
                elif key == "created":
                    created = _parse_date(value)
            body = "\n".join(lines[end + 1:]).strip()

    content = body.strip()
    if not title:
        title = _derive_title(content)
    if created is None:
        created = _mtime(path)
    return Note(title=title, content=content, created_at=created, id=str(path))


def _parse_date(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _mtime(path: Path) -> datetime:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.now()
