"""Tool `look`: le da ojos a BMO.

Envuelve la camara + la vision detras de una tool que el brain puede invocar.
Cuando el LLM decide 'look', el agente ejecuta esta funcion: captura un frame,
lo analiza y devuelve en texto que ve BMO.

Es una factory que recibe los ports por inyeccion: no sabe si la camara es
picamera2 o la AI Camera, ni si la vision es Haar o el Hailo. Solo habla ports.
"""

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
