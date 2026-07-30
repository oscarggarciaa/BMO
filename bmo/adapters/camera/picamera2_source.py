"""Adapter de camara con picamera2 (Raspberry Pi)."""

from __future__ import annotations

import logging
from typing import Any

from bmo.config import CameraConfig
from bmo.ports.camera import CameraSourcePort

_LOG = logging.getLogger(__name__)


class Picamera2Source(CameraSourcePort):
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._picam2: Any = None

    def start(self) -> None:
        try:
            from picamera2 import Picamera2

            self._picam2 = Picamera2()
            self._picam2.configure(
                self._picam2.create_preview_configuration(
                    main={"size": self._config.size, "format": self._config.format}
                )
            )
            self._picam2.start()
        except Exception:  # noqa: BLE001 - sin cámara, BMO sigue funcionando sin visión
            self._picam2 = None
            _LOG.warning(
                "no se pudo inicializar la cámara; BMO sigue funcionando sin visión"
            )

    def capture(self) -> Any:
        if self._picam2 is None:
            return None
        return self._picam2.capture_array()

    def stop(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2 = None
