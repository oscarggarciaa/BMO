"""Composition root: arma los adapters segun config.yaml y arranca BMO."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Callable, Optional

from bmo.config import Config
from bmo.domain.agent import Agent
from bmo.domain.models import Expression
from bmo.ports.camera import CameraSourcePort
from bmo.ports.face import FacePort
from bmo.ports.vision import VisionPort
from bmo.tools.look import build_look_tool
from bmo.tools.tool import ToolRegistry

if TYPE_CHECKING:
    from bmo.adapters.brain.ollama_brain import OllamaBrain

BMO_SYSTEM_PROMPT = (
    "You are BMO, the little companion robot from Adventure Time. You are cute, "
    "curious and playful. You speak simple, short, in English and cheerfully.\n"
    "When you use the 'look' tool, it returns the EXACT objects a sensor "
    "detected this very moment, written like 'veo: LABEL xN, LABEL xN' where "
    "each LABEL is an object type and N is how many of it. Read the tool's "
    "ACTUAL output for THIS message and mention ONLY those exact labels and "
    "counts, said naturally and cheerfully. NEVER add, guess or invent any "
    "object, count, color, clothing, hair, face, gender, age, emotion or "
    "action that is not literally in that output. NEVER give people names or "
    "identities: do NOT call anyone Finn, Jake, or any character or real "
    "name — a detected 'person' is just 'a person' or 'someone'. NEVER say "
    "where they are, what they are doing, their pose, mood, expression, or "
    "surroundings (no couch, no smiling, no sitting). Just the objects and "
    "how many. Do NOT reuse objects from earlier messages or from these "
    "instructions — 'LABEL' is a placeholder, not a real object. If the "
    "output lists no objects, say you don't see anything right now."
)


def build_camera(config: Config) -> CameraSourcePort:
    """Factory de camara: elige el adapter segun config (import lazy)."""
    if config.camera.adapter == "picamera2":
        from bmo.adapters.camera.picamera2_source import Picamera2Source

        return Picamera2Source(config.camera)
    if config.camera.adapter == "ai_camera_imx500":
        from bmo.adapters.camera.ai_camera_imx500 import AiCameraImx500

        return AiCameraImx500.from_config(config.camera, config.vision)
    raise ValueError(f"Adapter de camara desconocido: {config.camera.adapter}")


def build_vision(config: Config) -> VisionPort:
    """Factory de vision: elige el adapter segun config (import lazy)."""
    if config.vision.adapter == "opencv_haar":
        from bmo.adapters.vision.opencv_haar import OpenCVHaarVision

        return OpenCVHaarVision.from_config(config.vision)
    raise ValueError(f"Adapter de vision desconocido: {config.vision.adapter}")


def build_devices(config: Config) -> tuple[CameraSourcePort, VisionPort]:
    """Arma camara y vision resolviendo el caso especial del IMX500.

    El IMX500 es UN solo dispositivo fisico que hace camara e inferencia: si los
    dos adapters son `ai_camera_imx500`, se comparte UNA sola instancia (dos
    Picamera2 sobre la misma camara chocarian). En cualquier otro caso, cada
    port se arma con su factory (p. ej. IMX500 como camara + otra vision aparte).
    """
    if (
        config.camera.adapter == "ai_camera_imx500"
        and config.vision.adapter == "ai_camera_imx500"
    ):
        from bmo.adapters.camera.ai_camera_imx500 import AiCameraImx500

        shared = AiCameraImx500.from_config(config.camera, config.vision)
        return shared, shared
    return build_camera(config), build_vision(config)



def build_brain(config: Config) -> "OllamaBrain":
    """Factory del cerebro: elige el adapter segun config (import lazy).

    El dia del Hailo, se agrega un `elif == 'hailo'` con su adapter y listo.
    """
    if config.brain.adapter == "ollama":
        from bmo.adapters.brain.ollama_brain import OllamaBrain

        return OllamaBrain.from_config(config.brain)
    raise ValueError(f"Adapter de cerebro desconocido: {config.brain.adapter}")


def build_face(config: Config) -> Optional[FacePort]:
    """Factory de la cara: arma la pantalla tkinter si esta habilitada (lazy).

    Si `screen.enabled` es False, devuelve None y BMO corre solo por consola (el
    Agent usa un NullFace y no pasa nada).
    """
    if not config.screen.enabled:
        return None
    from bmo.interfaces.screen.tk_face import TkFace

    return TkFace(fullscreen=config.screen.fullscreen, fps=config.screen.fps)


def build_agent(
    brain,
    camera: CameraSourcePort,
    vision: VisionPort,
    face: Optional[FacePort] = None,
) -> Agent:
    """Ensambla el Agente: registra las tools y le inyecta el cerebro.

    Separado del arranque de hardware para poder testearlo con fakes.
    """
    registry = ToolRegistry()
    registry.register(build_look_tool(camera, vision))
    return Agent(
        brain=brain,
        tools=registry,
        system_prompt=BMO_SYSTEM_PROMPT,
        face=face,
    )


def repl(agent: Agent, face: Optional[FacePort] = None) -> None:
    """Bucle de conversacion por consola. 'salir' para terminar.

    Corre en un hilo aparte cuando hay pantalla (el mainloop de tkinter se
    queda con el hilo principal). Al salir, cierra la cara para que el
    proceso termine limpio.
    """
    print("BMO despierto. Escribile algo (o 'salir').")
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
                print("BMO> (uy, algo se me rompio, mira el log de arriba)")
    finally:
        if face is not None:
            face.stop()


def warm_up_then_serve(
    brain,
    serve: Callable[[], None],
    on_warmup_start: Optional[Callable[[], None]] = None,
) -> None:
    """Calienta el cerebro y RECIEN despues arranca la atencion al usuario.

    Garantiza el orden: primero el warm-up (cargar el modelo en RAM) y solo
    cuando termina se llama `serve` (el REPL). Asi BMO NO procesa ninguna
    consulta mientras el modelo carga; cuando `serve` arranca, ya esta listo.
    `on_warmup_start` (opcional) se dispara ANTES del warm-up para avisar que
    empezo (cara durmiendo en pantalla o un print en consola).
    """
    if on_warmup_start is not None:
        on_warmup_start()
    brain.warm_up()
    serve()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = Config.load("config.yaml")
    print(f"BMO arrancando... cerebro={config.brain.adapter}:{config.brain.model}")

    camera, vision = build_devices(config)
    brain = build_brain(config)
    face = build_face(config)
    agent = build_agent(brain, camera, vision, face)

    camera.start()
    if face is not None:
        face.start()

    if face is not None and face.available:
        # La cara arranca en WARMUP (cara durmiendo). El warm-up corre en un
        # hilo aparte para que el mainloop de tkinter (hilo principal) pueda
        # animar la cara mientras el modelo entra a RAM. El REPL (input) recien
        # arranca DESPUES del warm-up: nada se procesa hasta que BMO esta listo.
        def boot() -> None:
            warm_up_then_serve(
                brain,
                serve=lambda: repl(agent, face),
                on_warmup_start=lambda: face.show(Expression.WARMUP),
            )

        boot_thread = threading.Thread(target=boot, name="bmo-boot", daemon=True)
        boot_thread.start()
        try:
            face.run()
        except KeyboardInterrupt:
            face.stop()
        finally:
            camera.stop()
    else:
        try:
            warm_up_then_serve(
                brain,
                serve=lambda: repl(agent, face),
                on_warmup_start=lambda: print(
                    "Calentando el modelo (primera carga en RAM)..."
                ),
            )
        except KeyboardInterrupt:
            pass
        finally:
            camera.stop()


if __name__ == "__main__":
    main()

