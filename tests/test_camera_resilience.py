"""Tests de resiliencia de la camara: si no hay hardware, BMO no se muere.

En Windows `picamera2` ni siquiera esta instalado, asi que `start()` dispara un
ImportError; en la Pi sin camara conectada dispara un IndexError. En los dos
casos el adapter tiene que degradar con gracia: no explota y `capture()`
devuelve None (el contrato del puerto lo permite).
"""

from __future__ import annotations

from bmo.adapters.camera.picamera2_source import Picamera2Source
from bmo.config import CameraConfig


def test_start_does_not_raise_when_camera_unavailable() -> None:
    camera = Picamera2Source(CameraConfig())

    camera.start()

    assert camera.capture() is None


def test_capture_returns_none_before_start() -> None:
    camera = Picamera2Source(CameraConfig())

    assert camera.capture() is None


def test_stop_is_safe_when_never_started() -> None:
    camera = Picamera2Source(CameraConfig())

    assert camera.stop() is None
