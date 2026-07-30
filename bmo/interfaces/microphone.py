"""Interfaz de micrófono: conversar con BMO por voz con wake-word.
"""

from __future__ import annotations

import logging
from typing import Optional

from bmo.domain.agent import Agent
from bmo.domain.models import Expression
from bmo.domain.wakeword import WakeWord
from bmo.ports.face import FacePort
from bmo.ports.hearing import HearingPort


def listen_loop(
    agent: Agent,
    hearing: HearingPort,
    wake: WakeWord,
    face: Optional[FacePort] = None,
) -> None:
    """Bucle de conversación por micrófono: se activa con la wake-word HELLO.
    """
    print("BMO escuchando. Di 'hello' para despertarlo (Ctrl+C para salir).")
    try:
        while True:
            if face is not None:
                face.show(Expression.LISTENING)
            # crea el loop de escucha (bloqueante) y espera a que diga algo
            transcript = hearing.listen()
            # si hay algo que detectar, lo pasa al agente y devuelve la respuesta
            command = wake.detect(transcript)
            if command is None:
                continue  # sin wake-word: BMO sigue inactivo
            try:
                # empieza la conversación
                reply = agent.ask(command or "hello")
                print(f"BMO> {reply.text}")
            except Exception:
                logging.getLogger(__name__).exception("fallo procesando la voz")
                print("BMO> (ocurrió un error, revisa el log de arriba)")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        if face is not None:
            face.stop()
