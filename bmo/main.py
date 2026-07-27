"""Composition root: arma los adapters segun config.yaml y arranca BMO."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

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
    "curious and playful. You speak simple, short, in English and cheerfully."
)


def build_camera(config: Config) -> CameraSourcePort:
    """Factory de camara: elige el adapter segun config (import lazy)."""
    if config.camera.adapter == "picamera2":
        from bmo.adapters.camera.picamera2_source import Picamera2Source

        return Picamera2Source(config.camera)
    raise ValueError(f"Adapter de camara desconocido: {config.camera.adapter}")


def build_vision(config: Config) -> VisionPort:
    """Factory de vision: elige el adapter segun config (import lazy)."""
    if config.vision.adapter == "opencv_haar":
        from bmo.adapters.vision.opencv_haar import OpenCVHaarVision

        return OpenCVHaarVision.from_config(config.vision)
    if config.vision.adapter == "moondream":
        from bmo.adapters.vision.moondream import MoondreamVision

        return MoondreamVision.from_config(config.vision)
    raise ValueError(f"Adapter de vision desconocido: {config.vision.adapter}")


def build_brain(config: Config) -> "OllamaBrain":
    """Factory del cerebro: elige el adapter segun config (import lazy).

    El dia del Hailo, se agrega un `elif == 'hailo'` con su adapter y listo.
    """
    if config.brain.adapter == "ollama":
        from bmo.adapters.brain.ollama_brain import OllamaBrain

        return OllamaBrain.from_config(config.brain)
    raise ValueError(f"Adapter de cerebro desconocido: {config.brain.adapter}")


def build_face(config: Config) -> Optional[FacePort]:
    """Factory de la cara: arma el server web si esta habilitado (import lazy).

    Si `web.enabled` es False, devuelve None y BMO corre solo por consola (el
    Agent usa un NullFace y no pasa nada).
    """
    if not config.web.enabled:
        return None
    from bmo.interfaces.web.face_web import FaceWebServer

    return FaceWebServer(host=config.web.host, port=config.web.port)


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
    """Bucle de conversacion por consola. 'salir' para terminar."""
    print("BMO despierto. Escribile algo (o 'salir').")
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
        if text.lower() in {"salir", "exit", "quit", "chau"}:
            break
        try:
            reply = agent.ask(text)
            print(f"BMO> {reply.text}")
        except Exception:
            logging.getLogger(__name__).exception("fallo procesando el mensaje")
            print("BMO> (uy, algo se me rompio, mira el log de arriba)")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = Config.load("config.yaml")
    print(f"BMO arrancando... cerebro={config.brain.adapter}:{config.brain.model}")

    camera = build_camera(config)
    vision = build_vision(config)
    brain = build_brain(config)
    face = build_face(config)
    agent = build_agent(brain, camera, vision, face)

    camera.start()
    if face is not None:
        face.start()
        print(f"cara de BMO en http://{config.web.host}:{config.web.port}")
    try:
        repl(agent, face)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()

