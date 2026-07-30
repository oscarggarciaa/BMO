"""Reacción táctil de BMO: al tocar la pantalla, se queja y vuelve a su cara.

La pantalla táctil y la cara son el mismo dispositivo, así que el toque entra
por el FacePort. Este módulo arma la reacción (cara enfadada al instante +
queja hablada + vuelta a la cara neutra al terminar de hablar) como un callable
que el composition root registra con `face.set_on_touch(...)`.
"""

from __future__ import annotations

import threading
from typing import Callable

from bmo.domain.models import Expression
from bmo.ports.face import FacePort
from bmo.ports.voice import VoicePort

DEFAULT_PHRASE = "Ouch! That hurts!"


def build_touch_reaction(
    face: FacePort,
    voice: VoicePort,
    phrase: str = DEFAULT_PHRASE,
    hurt: Expression = Expression.ANGRY,
    idle: Expression = Expression.NEUTRAL,
) -> Callable[[], None]:
    """Arma la reacción a un toque: cara enfadada al instante y queja.

    Devuelve un callable sin argumentos, pensado para registrarse con
    `face.set_on_touch(...)`. La cara enfadada aparece al tocar y se mantiene
    hasta que BMO termina de hablar. Un lock evita que toques repetidos solapen
    la reacción: mientras BMO se está quejando, ignora los toques nuevos.
    """
    lock = threading.Lock()

    def react() -> None:
        if not lock.acquire(blocking=False):
            return  # ya se está quejando: ignora el toque repetido
        try:
            face.show(hurt)      # cara enfadada YA, al tocar (feedback inmediato)
            voice.speak(phrase)  # bloquea hasta que termina de hablar
        finally:
            face.show(idle)      # recupera la cara al dejar de hablar
            lock.release()

    return react
