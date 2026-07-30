"""Adapter de vision con OpenCV + Haar cascades (CPU)."""

from __future__ import annotations

import os
from typing import Any, Protocol

import cv2

from bmo.config import VisionConfig
from bmo.domain.models import BoundingBox, Detection, Perception
from bmo.ports.vision import VisionPort


class Cascade(Protocol):
    """Contrato mínimo de un clasificador Haar (lo que usamos de cv2)."""

    def detectMultiScale(self, image: Any, *args: Any, **kwargs: Any) -> Any: ...


def load_cascade(name: str, models_path: str) -> "cv2.CascadeClassifier":
    """Carga una cascada: primero desde models_path, luego desde las de OpenCV."""
    ruta = os.path.join(models_path, name)
    if not os.path.exists(ruta):
        ruta = os.path.join(cv2.data.haarcascades, name)
    cascada = cv2.CascadeClassifier(ruta)
    if cascada.empty():
        raise RuntimeError(f"No se pudo cargar la cascada: {name}")
    return cascada


class OpenCVHaarVision(VisionPort):
    def __init__(
        self,
        face_cascade: Cascade,
        eye_cascade: Cascade,
        smile_cascade: Cascade,
        min_face_size: int = 80,
    ) -> None:
        self._face = face_cascade
        self._eye = eye_cascade
        self._smile = smile_cascade
        self._min_face_size = min_face_size

    @classmethod
    def from_config(cls, config: VisionConfig) -> "OpenCVHaarVision":
        """Construye el adapter cargando las cascadas reales desde disco."""
        return cls(
            face_cascade=load_cascade(
                "haarcascade_frontalface_default.xml", config.models_path
            ),
            eye_cascade=load_cascade("haarcascade_eye.xml", config.models_path),
            smile_cascade=load_cascade("haarcascade_smile.xml", config.models_path),
            min_face_size=config.min_face_size,
        )

    def analyze(self, frame: Any, question: str = "") -> Perception:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        detections: list[Detection] = []

        caras = self._face.detectMultiScale(
            gray, 1.1, 6, minSize=(self._min_face_size, self._min_face_size)
        )

        for (x, y, w, h) in caras:
            detections.append(Detection("face", BoundingBox(x, y, w, h)))

            roi_ojos = gray[y : y + h // 2, x : x + w]
            ojos = self._eye.detectMultiScale(
                roi_ojos, 1.1, 10, minSize=(w // 8, w // 8)
            )
            for (ex, ey, ew, eh) in ojos[:2]:
                detections.append(
                    Detection("eye", BoundingBox(x + ex, y + ey, ew, eh))
                )

            oy = y + (2 * h) // 3
            roi_boca = gray[oy : y + h, x : x + w]
            bocas = self._smile.detectMultiScale(
                roi_boca, 1.3, 25, minSize=(w // 4, h // 10)
            )
            if len(bocas) > 0:
                sx, sy, sw, sh = bocas[0]
                detections.append(
                    Detection("smile", BoundingBox(x + sx, oy + sy, sw, sh))
                )

        return Perception(detections=detections)
