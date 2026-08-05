"""
Punto de entrada de BMO: carga la config, construye con el builder y arranca.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from bmo.builder import (
    build_agent,
    build_brain,
    build_devices,
    build_face,
    build_hearing,
    build_voice,
)
from bmo.config import Config
from bmo.domain.models import Expression
from bmo.domain.wakeword import WakeWord
from bmo.adapters.voice.serialized_voice import SerializedVoice
from bmo.interfaces.console import repl
from bmo.interfaces.microphone import listen_loop
from bmo.interfaces.touch import build_touch_reaction
from bmo.ports.voice import NullVoice


def warm_up_then_serve(
    brain,
    serve: Callable[[], None],
    on_warmup_start: Optional[Callable[[], None]] = None,
) -> None:
    """Calienta el cerebro y después arranca la atención al usuario.
    """
    if on_warmup_start is not None:
        on_warmup_start()
    brain.warm_up()
    serve()


def run(config: Config) -> None:
    """Arma BMO desde la config y lo pone a atender (por voz o por teclado)."""
    print(f"BMO arrancando... cerebro={config.brain.adapter}:{config.brain.model}")

    camera, vision = build_devices(config)
    brain = build_brain(config)
    face = build_face(config)
    # una sola tarjeta de sonido para diferentes features no usen el mismo driver a la vez
    voice = SerializedVoice(build_voice(config) or NullVoice())
    hearing = build_hearing(config)
    agent = build_agent(brain, camera, vision, face, voice)
    # tactil: si hay pantalla, tocarla hace que BMO se queje y vuelva a su cara
    if face is not None:
        face.set_on_touch(build_touch_reaction(face, voice))
    # si hay micrófono, guarda el callable con el bucle de escucha
    if hearing is not None and hearing.available:
        wake = WakeWord(config.hearing.wake_word)
        serve: Callable[[], None] = lambda: listen_loop(agent, hearing, wake, face)
    # si no, guarda el bucle con entrada por terminal
    else:
        serve = lambda: repl(agent, face)

    camera.start()
    if face is not None:
        face.start()

    if face is not None and face.available:
        def boot() -> None:
            warm_up_then_serve(
                brain,
                serve=serve,
                on_warmup_start=lambda: face.show(Expression.WARMUP),
            )
        # conversación en thread secundario
        boot_thread = threading.Thread(target=boot, name="bmo-boot", daemon=True)
        boot_thread.start()
        try:
            # pantalla en el hilo principal
            face.run()
        except KeyboardInterrupt:
            face.stop()
        finally:
            camera.stop()
    else:
        try:
            warm_up_then_serve(
                brain,
                serve=serve,
                on_warmup_start=lambda: print(
                    "Calentando el modelo (primera carga en RAM)..."
                ),
            )
        except KeyboardInterrupt:
            pass
        finally:
            camera.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # carga la config del fichero con todos los parámetros
    config = Config.load("config.yaml")
    run(config)


if __name__ == "__main__":
    main()

