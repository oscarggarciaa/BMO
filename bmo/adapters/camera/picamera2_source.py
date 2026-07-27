"""Adapter de camara usando picamera2 (Raspberry Pi).

picamera2 solo existe en la Raspberry Pi, por eso se importa dentro del __init__
del adapter (lazy): asi el paquete se puede importar y testear en Windows/Mac.
"""

from __future__ import annotations

from typing import Any

from bmo.config import CameraConfig
from bmo.ports.camera import CameraSourcePort


class Picamera2Source(CameraSourcePort):
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._picam2: Any = None

    def start(self) -> None:
        from picamera2 import Picamera2  # import lazy: solo en la Pi

        self._picam2 = Picamera2()
        self._picam2.configure(
            self._picam2.create_preview_configuration(
                main={"size": self._config.size, "format": self._config.format}
            )
        )
        self._picam2.start()

    def capture(self) -> Any:
        if self._picam2 is None:
            raise RuntimeError("La camara no fue iniciada. Llama a start() primero.")
        # picamera2 con formato 'RGB888' entrega el array en orden BGR:
        # ya sirve directo para OpenCV.
        return self._picam2.capture_array()

    def stop(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2 = None
