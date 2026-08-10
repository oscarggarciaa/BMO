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


def test_clean_reply_strips_trailing_emoji() -> None:
    # Un modelo pequeno ignora "never use emojis" en el prompt, asi que el
    # filtro tiene que vivir en el codigo. El emoji sale y el texto queda limpio.
    assert OllamaBrain._clean_reply("I see a book! \U0001F4DA") == "I see a book!"


def test_clean_reply_strips_inline_emoticon() -> None:
    assert (
        OllamaBrain._clean_reply("could you clarify? \U0001F60A")
        == "could you clarify?"
    )


def test_clean_reply_keeps_plain_text_intact() -> None:
    assert OllamaBrain._clean_reply("hello there, how are you?") == (
        "hello there, how are you?"
    )


def test_clean_reply_falls_back_when_only_an_emoji() -> None:
    # Si el modelo responde SOLO con un emoji, tras filtrarlo no queda nada:
    # debe caer en el mensaje de respaldo, no devolver cadena vacia.
    assert OllamaBrain._clean_reply("\U0001F44D") != ""
