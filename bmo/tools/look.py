"""Tool `look`: captura un frame y lo describe (los ojos de BMO)."""

from __future__ import annotations

from bmo.ports.camera import CameraSourcePort
from bmo.ports.vision import VisionPort
from bmo.tools.tool import Tool


def build_look_tool(camera: CameraSourcePort, vision: VisionPort) -> Tool:
    """Arma la tool `look` cableando una camara y una vision concretas."""

    def look(question: str = "") -> str:
        frame = camera.capture()
        if frame is None:
            return "I couldn't capture an image from the camera"
        perception = vision.analyze(frame, question=question)
        return perception.summary()

    return Tool(
        name="look",
        description=(
            "Look through the camera and describe what BMO sees right now "
            "(faces, eyes, smiles). This is the ONLY way BMO has to see or take "
            "a photo. Use it whenever the user asks to see something, to look, "
            "to take a photo, or asks what you see, what is in front, or who is "
            "there."
        ),
        func=look,
        direct=True,
    )
