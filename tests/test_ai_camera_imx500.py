"""Tests del adapter AI Camera IMX500 (inferencia on-sensor).

El hardware (crear IMX500, Picamera2) no se puede testear en Windows, asi que:
- Aislamos el parseo PURO de tensores -> Perception y lo testeamos a fondo.
- Verificamos la resiliencia: sin hardware, el adapter degrada con gracia igual
  que picamera2_source (no explota; capture() devuelve None).
- Verificamos que el composition root comparte UNA sola instancia entre los dos
  ports (el IMX500 es un solo dispositivo fisico).
"""

from __future__ import annotations

import numpy as np

from bmo import main as composition_root
from bmo.adapters.camera.ai_camera_imx500 import (
    AiCameraImx500,
    build_perception,
    prepare_boxes,
)
from bmo.config import BrainConfig, CameraConfig, Config, ScreenConfig, VisionConfig
from bmo.domain.models import BoundingBox, Detection
from bmo.ports.camera import CameraSourcePort
from bmo.ports.vision import VisionPort


def _identity_box(coords) -> BoundingBox:
    x, y, w, h = coords
    return BoundingBox(int(x), int(y), int(w), int(h))


# --- logica pura: tensores -> Perception ------------------------------------


def test_build_perception_maps_class_index_to_label() -> None:
    perception = build_perception(
        boxes=[(0, 0, 10, 10)],
        scores=[0.9],
        classes=[0],
        labels=["person", "chair"],
        threshold=0.5,
        to_box=_identity_box,
    )

    assert perception.detections == [Detection("person", BoundingBox(0, 0, 10, 10))]


def test_build_perception_filters_scores_at_or_below_threshold() -> None:
    perception = build_perception(
        boxes=[(0, 0, 10, 10), (1, 1, 2, 2)],
        scores=[0.9, 0.2],
        classes=[0, 1],
        labels=["person", "chair"],
        threshold=0.5,
        to_box=_identity_box,
    )

    assert perception.labels() == ["person"]


def test_build_perception_uses_converter_to_scale_the_box() -> None:
    def scale(coords) -> BoundingBox:
        x, y, w, h = coords
        return BoundingBox(x * 2, y * 2, w * 2, h * 2)

    perception = build_perception(
        boxes=[(1, 1, 3, 3)],
        scores=[0.8],
        classes=[0],
        labels=["cup"],
        threshold=0.5,
        to_box=scale,
    )

    assert perception.detections[0].box == BoundingBox(2, 2, 6, 6)


def test_build_perception_out_of_range_class_falls_back_to_index() -> None:
    perception = build_perception(
        boxes=[(0, 0, 1, 1)],
        scores=[0.9],
        classes=[7],
        labels=["person"],
        threshold=0.5,
        to_box=_identity_box,
    )

    assert perception.labels() == ["7"]


def test_build_perception_empty_gives_nothing_summary() -> None:
    perception = build_perception(
        boxes=[],
        scores=[],
        classes=[],
        labels=[],
        threshold=0.5,
        to_box=_identity_box,
    )

    assert perception.summary() == "veo: nada"


# --- prepare_boxes: normalizacion y reorden segun los intrinsics del modelo ---


def test_prepare_boxes_normalizes_by_input_height() -> None:
    boxes = np.array([[0.0, 0.0, 100.0, 200.0]])

    out = prepare_boxes(boxes, bbox_normalization=True, bbox_order="yx", input_h=100)

    assert out.tolist() == [[0.0, 0.0, 1.0, 2.0]]


def test_prepare_boxes_reorders_xy_to_yx() -> None:
    boxes = np.array([[1.0, 2.0, 3.0, 4.0]])  # x0, y0, x1, y1

    out = prepare_boxes(boxes, bbox_normalization=False, bbox_order="xy", input_h=1)

    assert out.tolist() == [[2.0, 1.0, 4.0, 3.0]]  # -> y0, x0, y1, x1


def test_prepare_boxes_leaves_yx_untouched() -> None:
    boxes = np.array([[1.0, 2.0, 3.0, 4.0]])

    out = prepare_boxes(boxes, bbox_normalization=False, bbox_order="yx", input_h=1)

    assert out.tolist() == [[1.0, 2.0, 3.0, 4.0]]


# --- el adapter cumple los dos ports y degrada con gracia --------------------


def test_adapter_implements_both_ports() -> None:
    adapter = AiCameraImx500(size=(640, 480), rpk_path="x.rpk")

    assert isinstance(adapter, CameraSourcePort)
    assert isinstance(adapter, VisionPort)


def test_start_does_not_raise_when_hardware_unavailable() -> None:
    adapter = AiCameraImx500(size=(640, 480), rpk_path="x.rpk")

    adapter.start()

    assert adapter.capture() is None


def test_analyze_returns_empty_perception_when_not_started() -> None:
    adapter = AiCameraImx500(size=(640, 480), rpk_path="x.rpk")

    assert adapter.analyze("frame").detections == []


def test_stop_is_safe_when_never_started() -> None:
    adapter = AiCameraImx500(size=(640, 480), rpk_path="x.rpk")

    assert adapter.stop() is None


def test_from_config_builds_the_adapter() -> None:
    camera = CameraConfig(adapter="ai_camera_imx500")
    vision = VisionConfig(adapter="ai_camera_imx500", rpk_path="model.rpk", threshold=0.6)

    adapter = AiCameraImx500.from_config(camera, vision)

    assert isinstance(adapter, AiCameraImx500)


# --- composition root: un solo dispositivo, dos ports -----------------------


def _imx500_config() -> Config:
    return Config(
        camera=CameraConfig(adapter="ai_camera_imx500"),
        vision=VisionConfig(adapter="ai_camera_imx500", rpk_path="x.rpk"),
        brain=BrainConfig(),
        screen=ScreenConfig(enabled=False),
    )


def test_build_devices_shares_one_instance_when_both_are_imx500() -> None:
    camera, vision = composition_root.build_devices(_imx500_config())

    assert camera is vision
    assert isinstance(camera, AiCameraImx500)
