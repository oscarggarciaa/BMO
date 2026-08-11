"""
Punto de entrada de BMO: carga la config, construye con el builder y arranca.
"""

from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
import os
import sys
import threading
from typing import Callable, List, Optional

from bmo.builder import (
    build_agent,
    build_brain,
    build_devices,
    build_face,
    build_hearing,
    build_notes,
    build_voice,
)
from bmo.config import Config
from bmo.domain.models import Expression
from bmo.domain.wakeword import WakeWord
from bmo.adapters.voice.serialized_voice import SerializedVoice
from bmo.interfaces.console import repl
from bmo.interfaces.microphone import listen_loop
from bmo.ports.voice import NullVoice


def warm_up_then_serve(
    brain,
    serve: Callable[[], None],
    on_warmup_start: Optional[Callable[[], None]] = None,
    voice=None,
) -> None:
    """Calienta el cerebro y después arranca la atención al usuario."""
    if on_warmup_start is not None:
        on_warmup_start()
    brain.warm_up()
    if voice is not None:
        voice.speak("BMO is ready to talk!")
    serve()


def run(config: Config) -> None:
    """Arma BMO desde la config y lo pone a atender (por voz o por teclado)."""
    logger.debug(f"BMO arrancando... cerebro={config.brain.adapter}:{config.brain.model}")

    camera, vision = build_devices(config)
    brain = build_brain(config)
    face = build_face(config)
    # una sola tarjeta de sonido para diferentes features no usen el mismo driver a la vez
    voice = SerializedVoice(build_voice(config) or NullVoice())
    hearing = build_hearing(config)
    notes = build_notes(config)
    agent = build_agent(brain, camera, vision, face, voice, notes)
    # tactil: si hay pantalla y notas, tocarla abre el menu de notas guardadas
    if face is not None and notes is not None:
        face.set_notes_provider(notes.all)
        face.set_notes_deleter(notes.delete)
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
                voice=voice
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


def setup_logging(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.CRITICAL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    os.environ["LIBCAMERA_LOG_LEVELS"] = "*:INFO" if debug else "*:FATAL"


def resolve_debug(argv: List[str], config_debug: bool) -> bool:
    if argv and argv[0].strip().lower() == "debug":
        return True
    return config_debug


def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    # carga la config del fichero con todos los parámetros
    config = Config.load("config.yaml")
    setup_logging(resolve_debug(argv, config.debug))
    run(config)


if __name__ == "__main__":
    main()

