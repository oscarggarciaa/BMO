"""Tests del prompt de tool-calling por texto (`_build_action_protocol`).

Contexto: BMO no usa function-calling nativo (hailo-ollama lo rechaza con un
500). Usa un protocolo ReAct por prompt: describe las acciones en el system
prompt y espera un JSON {"action": "..."}. Un modelo pequeño como
qwen2.5-instruct:1.5b sigue ese protocolo peor que llama3.2:3b, asi que el
prompt tiene que ser MUY explicito con los disparadores de vision.

Estos tests blindan la ESTRUCTURA del prompt (no el comportamiento del LLM,
que solo se valida con el modelo real en la Pi).
"""

from __future__ import annotations

from bmo.adapters.brain.ollama_brain import OllamaBrain


class FakeTool:
    name = "look"
    description = "see the world in front of you"


def _protocol() -> str:
    return OllamaBrain._build_action_protocol([FakeTool()])


def test_protocol_is_empty_without_tools() -> None:
    assert OllamaBrain._build_action_protocol([]) == ""


def test_protocol_lists_tool_name_and_description() -> None:
    protocol = _protocol()

    assert "look" in protocol
    assert "see the world in front of you" in protocol


def test_protocol_documents_the_exact_json_action_format() -> None:
    protocol = _protocol()

    assert '{"action": "look"}' in protocol


def test_protocol_includes_the_vision_phrases_that_failed_with_qwen() -> None:
    # qwen2.5-instruct:1.5b cayo en CHAT MODE ante "what are you looking at".
    # El prompt debe dar ejemplos ricos de vision para que un modelo pequeno
    # dispare la accion en esas variantes.
    lowered = _protocol().lower()

    assert "looking at" in lowered
    assert "can you see" in lowered
    assert "in front of you" in lowered


def test_protocol_distinguishes_vision_from_identity_questions() -> None:
    # La confusion del 1.5B: trato "what are you looking at" (vision -> ACTION)
    # como si fuera "who are you" (identidad -> CHAT). El prompt debe separar
    # explicitamente ambos casos.
    lowered = _protocol().lower()

    assert "who you are" in lowered
    assert "what you see" in lowered


def test_protocol_includes_vision_verb_synonyms() -> None:
    # El usuario puede pedir la vision con verbos distintos ("watching",
    # "staring", "observing"...). Un modelo pequeno necesita verlos listados
    # para mapearlos todos a la misma accion de vision.
    lowered = _protocol().lower()

    assert "watching" in lowered
    assert "staring" in lowered
    assert "observing" in lowered
