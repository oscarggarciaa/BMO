"""Tool `look`: captura un frame y lo describe (los ojos de BMO)."""

from __future__ import annotations

from bmo.ports.camera import CameraSourcePort
from bmo.ports.vision import VisionPort
from bmo.tools.tool import Tool


def build_look_tool(camera: CameraSourcePort, vision: VisionPort) -> Tool:
    """Arma la tool `look` cableando una camara y una vision concretas."""

    def look() -> str:
        frame = camera.capture()
        if frame is None:
            return "no pude capturar imagen de la camara"
        perception = vision.analyze(frame)
        return perception.summary()

    return Tool(
        name="look",
        description=(
            "Mira a traves de la camara y describe que ve BMO ahora mismo "
            "(caras, ojos, sonrisas). Esta es la unica forma que tiene BMO de ver "
            "o sacar una foto. Usala siempre que el usuario pida ver algo, mirar, "
            "sacar o hacer una foto, o pregunte que ves, que hay adelante o quien "
            "esta ahi."
        ),
        func=look,
    )
