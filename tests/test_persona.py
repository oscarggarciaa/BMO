"""Tests del system prompt de la persona de BMO (`BMO_SYSTEM_PROMPT`).

BMO habla por un TTS (Piper) con audio limitado: las respuestas largas
DESBORDAN el buffer de audio ("overrun!!!" en los logs de la Pi). El prompt
debe exigir respuestas CORTAS de forma explicita e inequivoca, no solo un
"short" suelto que un modelo de 1.5B ignora.
"""

from __future__ import annotations

from bmo.domain.persona import BMO_SYSTEM_PROMPT


def test_prompt_demands_short_answers_with_a_hard_limit() -> None:
    lowered = BMO_SYSTEM_PROMPT.lower()

    # Debe pedir brevedad con un limite CONCRETO (una o dos frases), no vago.
    assert "one or two sentences" in lowered
    assert "never write long" in lowered


def test_prompt_still_describes_the_look_tool_contract() -> None:
    # El afinado de brevedad NO debe borrar el contrato de la tool 'look'.
    assert "look" in BMO_SYSTEM_PROMPT
    assert "veo:" in BMO_SYSTEM_PROMPT
