"""Tests del cerebro OllamaBrain (foco: warm-up del modelo al arrancar)."""

from __future__ import annotations

from typing import Any, List

from bmo.adapters.brain.ollama_brain import OllamaBrain


class FakeOllamaClient:
    """Cliente ollama falso: registra cada llamada a chat."""

    def __init__(self) -> None:
        self.calls: List[dict[str, Any]] = []

    def chat(self, model: str, messages: list, **kwargs: Any) -> dict:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return {"message": {"content": "hi"}}


def test_warm_up_pings_the_model_once() -> None:
    # warm_up carga el modelo en RAM haciendo UNA consulta minima al arrancar,
    # asi la primera pregunta real no paga el cold-start (~64s).
    client = FakeOllamaClient()
    brain = OllamaBrain(model="llama3.2:3b", host="x", client=client)

    brain.warm_up()

    assert len(client.calls) == 1
    assert client.calls[0]["model"] == "llama3.2:3b"


def test_warm_up_swallows_client_errors() -> None:
    # Si Ollama no esta listo, el warm-up NO debe tumbar el arranque de BMO.
    class BoomClient:
        def chat(self, *args: Any, **kwargs: Any) -> dict:
            raise RuntimeError("ollama not ready")

    brain = OllamaBrain(model="llama3.2:3b", host="x", client=BoomClient())

    brain.warm_up()  # no debe lanzar
