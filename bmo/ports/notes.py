"""Port de las notas: la memoria escrita de BMO (estilo Obsidian)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from bmo.domain.models import Note


class NotesPort(ABC):
    """Guarda y recupera las notas de BMO.

    El adapter concreto decide DÓNDE viven (ficheros markdown, base de datos...).
    El dominio y las tools solo conocen esta interfaz.
    """

    @abstractmethod
    def save(self, content: str, title: str = "") -> Note:
        """Guarda una nota y la devuelve ya normalizada (con título y fecha)."""

    @abstractmethod
    def all(self) -> List[Note]:
        """Devuelve todas las notas, de la más nueva a la más vieja."""

    @abstractmethod
    def delete(self, note: Note) -> None:
        """Borra la nota indicada (identificada por su `id`). Idempotente."""


class NullNotes(NotesPort):
    """Notas que no persisten nada.

    Patrón Null Object: si las notas están deshabilitadas, el resto del código
    puede llamar a `save`/`all` sin preguntar `if notes is not None`.
    """

    def save(self, content: str, title: str = "") -> Note:
        from datetime import datetime

        return Note(title=title, content=content, created_at=datetime.now())

    def all(self) -> List[Note]:
        return []

    def delete(self, note: Note) -> None:
        return None
