"""Port de las notas: la memoria escrita de BMO (estilo Obsidian)."""

from __future__ import annotations

import re
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

    def search(self, query: str, limit: int = 5) -> List[Note]:
        """Devuelve las notas más relevantes para `query`, la más relevante primero.
        """
        terms = _terms(query)
        if not terms:
            return []
        scored = [
            (score, note)
            for note in self.all()
            if (score := _score(note, terms)) > 0
        ]
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [note for _, note in scored[:limit]]


class NullNotes(NotesPort):
    """Notas que no persisten nada.
    """

    def save(self, content: str, title: str = "") -> Note:
        from datetime import datetime

        return Note(title=title, content=content, created_at=datetime.now())

    def all(self) -> List[Note]:
        return []

    def delete(self, note: Note) -> None:
        return None


def _terms(query: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (query or "").lower())


def _score(note: Note, terms: List[str]) -> int:
    title = note.title.lower()
    body = note.content.lower()
    score = 0
    for term in terms:
        if term in title:
            score += 2  # el título pesa más que el cuerpo
        if term in body:
            score += 1
    return score
