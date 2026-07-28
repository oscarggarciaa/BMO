"""Adapter de la Raspberry Pi AI Camera (Sony IMX500): inferencia on-sensor.

A diferencia de picamera2_source + moondream, el IMX500 ACOPLA camara e
inferencia: la red neuronal corre DENTRO del sensor y las detecciones salen del
metadata de su propio pipeline, no de un frame arbitrario. Ademas `IMX500(...)`
debe crearse ANTES que `Picamera2()`. Por eso es UN solo adapter que cumple los
dos ports (`CameraSourcePort` + `VisionPort`): el composition root inyecta la
misma instancia como camara y como vision.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, List, Sequence

from bmo.config import CameraConfig, VisionConfig
from bmo.domain.models import BoundingBox, Detection, Perception
from bmo.ports.camera import CameraSourcePort
from bmo.ports.vision import VisionPort

_LOG = logging.getLogger(__name__)


def build_perception(
    boxes: Iterable[Any],
    scores: Iterable[float],
    classes: Iterable[float],
    labels: Sequence[str],
    threshold: float,
    to_box: Callable[[Any], BoundingBox],
) -> Perception:
    """Convierte los tensores de salida del IMX500 en una Perception (logica pura).

    - `boxes`, `scores`, `classes`: iterables alineados (una entrada por deteccion).
    - `labels`: nombres de clase indexados por el id que devuelve la red.
    - `threshold`: se descartan las detecciones con score <= threshold.
    - `to_box`: convierte las coords crudas del tensor a un BoundingBox ya escalado
      al espacio de la imagen (en produccion usa `convert_inference_coords`).
    """
    detections: List[Detection] = []
    for coords, score, cls in zip(boxes, scores, classes):
        if float(score) <= threshold:
            continue
        idx = int(cls)
        label = labels[idx] if 0 <= idx < len(labels) else str(idx)
        detections.append(Detection(label, to_box(coords)))
    return Perception(detections=detections)


def prepare_boxes(
    boxes: Any,
    bbox_normalization: bool,
    bbox_order: str,
    input_h: int,
) -> Any:
    """Normaliza y reordena las cajas crudas segun los intrinsics del modelo.

    - `bbox_normalization`: si el modelo entrega coords en pixeles del tensor de
      entrada, se dividen por `input_h` para llevarlas a [0, 1].
    - `bbox_order`: 'yx' -> (y0, x0, y1, x1) (lo que espera convert_inference_coords);
      'xy' -> (x0, y0, x1, y1), que hay que reordenar a 'yx'.
    """
    if bbox_normalization:
        boxes = boxes / input_h
    if bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]
    return boxes


class AiCameraImx500(CameraSourcePort, VisionPort):
    """Camara + vision on-sensor en un solo dispositivo (Sony IMX500)."""

    def __init__(
        self,
        size: tuple[int, int],
        rpk_path: str,
        threshold: float = 0.55,
        fmt: str = "RGB888",
    ) -> None:
        self._size = size
        self._rpk_path = rpk_path
        self._threshold = threshold
        self._format = fmt
        self._picam2: Any = None
        self._imx500: Any = None
        self._intrinsics: Any = None
        self._labels: List[str] = []

    @classmethod
    def from_config(
        cls, camera: CameraConfig, vision: VisionConfig
    ) -> "AiCameraImx500":
        return cls(
            size=camera.size,
            rpk_path=vision.rpk_path,
            threshold=vision.threshold,
            fmt=camera.format,
        )

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
            from picamera2.devices import IMX500

            # OJO: el IMX500 debe existir ANTES de instanciar Picamera2.
            self._imx500 = IMX500(self._rpk_path)
            intrinsics = self._imx500.network_intrinsics
            self._intrinsics = intrinsics
            if intrinsics is not None and intrinsics.labels:
                self._labels = list(intrinsics.labels)

            self._picam2 = Picamera2(self._imx500.camera_num)
            self._picam2.configure(
                self._picam2.create_preview_configuration(
                    main={"size": self._size, "format": self._format}
                )
            )
            self._picam2.start()
        except Exception:  # noqa: BLE001 - sin AI Camera, BMO sigue vivo pero ciego
            self._picam2 = None
            self._imx500 = None
            self._intrinsics = None
            _LOG.warning(
                "no se pudo inicializar la AI Camera IMX500; "
                "BMO sigue funcionando sin vista"
            )

    def capture(self) -> Any:
        if self._picam2 is None:
            return None
        return self._picam2.capture_array()

    def analyze(self, frame: Any, question: str = "") -> Perception:
        # El IMX500 no analiza un frame arbitrario: lee las detecciones frescas
        # del metadata de su propio pipeline. `frame` y `question` se ignoran.
        if self._picam2 is None or self._imx500 is None:
            return Perception()

        metadata = self._picam2.capture_metadata()
        # add_batch=True: los tensores vienen como (1, N, ...); [0] saca el batch.
        np_outputs = self._imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return Perception()

        boxes = np_outputs[0][0]
        scores = np_outputs[1][0]
        classes = np_outputs[2][0]

        _, input_h = self._imx500.get_input_size()
        bbox_normalization = getattr(self._intrinsics, "bbox_normalization", False)
        bbox_order = getattr(self._intrinsics, "bbox_order", "yx")
        boxes = prepare_boxes(boxes, bbox_normalization, bbox_order, input_h)

        def to_box(coords: Any) -> BoundingBox:
            # convert_inference_coords devuelve una tupla (x, y, w, h), NO un objeto.
            x, y, w, h = self._imx500.convert_inference_coords(
                coords, metadata, self._picam2
            )
            return BoundingBox(int(x), int(y), int(w), int(h))

        return build_perception(
            boxes, scores, classes, self._labels, self._threshold, to_box
        )

    def stop(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()
        self._picam2 = None
        self._imx500 = None
        self._intrinsics = None
