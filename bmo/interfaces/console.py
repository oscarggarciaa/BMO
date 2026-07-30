"""Interfaz de teclado: conversar con BMO por consola (driving adapter).

Es una de las formas en que el humano CONTROLA a BMO desde el exterior. Por eso
vive en `interfaces/`, al lado de la pantalla.
"""

from __future__ import annotations

import logging
from typing import Optional

from bmo.domain.agent import Agent
from bmo.domain.models import Expression
from bmo.ports.face import FacePort


def repl(agent: Agent, face: Optional[FacePort] = None) -> None:
    """Bucle de conversación por consola. 'exit' para terminar.

    Corre en un hilo aparte cuando hay pantalla (el mainloop de tkinter se
    queda con el hilo principal). Al salir, cierra la cara para que el
    proceso termine de forma limpia.
    """
    print("BMO despierto. Escríbele algo (o 'salir').")
    try:
        while True:
            if face is not None:
                face.show(Expression.LISTENING)
            try:
                text = input("USER> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in {"exit"}:
                break
            try:
                reply = agent.ask(text)
                print(f"BMO> {reply.text}")
            except Exception:
                logging.getLogger(__name__).exception("fallo procesando el mensaje")
                print("BMO> (ocurrió un error, revisa el log de arriba)")
    finally:
        if face is not None:
            face.stop()
