"""Wake-word: la palabra magica que despierta a BMO.

Logica de DOMINIO, pura y sin dependencias de hardware. El adapter de audio
solo entrega texto; aca se decide si ese texto despierta a BMO y cual es el
comando. Es trivial de testear porque no toca ni microfono ni STT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WakeWord:
    """Detecta la palabra magica al principio de lo escuchado.

    Por defecto es 'hello': BMO solo responde si la frase ARRANCA con ella.
    """

    word: str = "hello"

    def detect(self, transcript: str) -> Optional[str]:
        """Devuelve el comando si `transcript` empieza con la palabra magica.

        - "hello what do you see" -> "what do you see" (el comando)
        - "hello"                 -> "" (desperto, sin comando: se saludara)
        - "what do you see"       -> None (sin wake-word: BMO sigue dormido)

        La comparacion es case-insensitive y por palabra completa: "hello" en
        el medio de la frase NO despierta a BMO.
        """
        words = transcript.strip().lower().split()
        if not words or words[0] != self.word.lower():
            return None
        return " ".join(words[1:])
