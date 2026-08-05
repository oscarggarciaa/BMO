"""Wake-word: la palabra que activa a BMO.

Lógica de DOMINIO, pura y sin dependencias de hardware. El adapter de audio
solo entrega texto; aquí se decide si ese texto activa a BMO y cuál es el
comando. Es trivial de testear porque no toca ni micrófono ni STT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WakeWord:
    """Detecta la palabra de activación al principio de lo escuchado.

    Por defecto es 'hello': BMO solo responde si la frase ARRANCA con ella.
    """

    word: str = "hello"

    def detect(self, transcript: str) -> Optional[str]:
        """Devuelve el comando si `transcript` empieza con la palabra de activación.
        La comparación es case-insensitive y por palabra completa: "hello" en
        el medio de la frase NO activa a BMO.
        """
        words = transcript.strip().lower().split()
        if not words or words[0] != self.word.lower():
            return None
        return " ".join(words[1:])
