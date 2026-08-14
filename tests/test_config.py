"""Tests de la carga de config: robustez ante drift entre config.yaml y el código."""

from __future__ import annotations

from bmo.config import Config


def test_from_dict_ignores_unknown_keys() -> None:
    # Si config.yaml trae una clave que el dataclass no conoce (p. ej. tras
    # actualizar el YAML pero no el código), NO debe reventar: se ignora.
    data = {
        "brain": {
            "adapter": "hailo",
            "model": "qwen3:1.7b",
            "host": "http://localhost:8000",
            "future_field": 123,  # clave desconocida
        }
    }

    config = Config.from_dict(data)

    assert config.brain.adapter == "hailo"
    assert config.brain.model == "qwen3:1.7b"


def test_from_dict_keeps_known_keys() -> None:
    data = {"brain": {"adapter": "hailo", "max_history": 2}}

    config = Config.from_dict(data)

    assert config.brain.adapter == "hailo"
    assert config.brain.max_history == 2
