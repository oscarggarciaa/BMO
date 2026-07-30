"""Agente: orquesta la conversacion entre el humano, el brain y las tools."""

from __future__ import annotations

import logging
from typing import List, Optional

from bmo.domain.models import Expression, Message, ToolCall, Utterance
from bmo.ports.face import FacePort, NullFace
from bmo.ports.voice import NullVoice, VoicePort
from bmo.tools.tool import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    """Orquesta la conversacion entre el humano, el brain y las tools."""

    def __init__(
        self,
        brain,
        tools: ToolRegistry,
        system_prompt: str = "",
        max_steps: int = 5,
        face: Optional[FacePort] = None,
        voice: Optional[VoicePort] = None,
    ) -> None:
        self._brain = brain
        self._tools = tools
        self._max_steps = max_steps
        self._face = face or NullFace()
        self._voice = voice or NullVoice()
        self._history: List[Message] = []
        if system_prompt:
            self._history.append(Message(role="system", content=system_prompt))

    @property
    def history(self) -> List[Message]:
        """Copia de la memoria de conversacion (solo lectura)."""
        return list(self._history)

    def ask(self, text: str) -> Utterance:
        """Le pregunta algo a BMO y devuelve su respuesta final.

        Corre el loop decide -> ejecuta tools -> decide, hasta que el brain
        responde sin pedir mas tools (o hasta agotar max_steps).
        """
        self._history.append(Message(role="user", content=text))
        logger.info("USER pregunto: %s", text)
        self._face.show(Expression.THINKING)

        for step in range(self._max_steps):
            logger.info("paso %d/%d: BMO esta pensando...", step + 1, self._max_steps)
            decision = self._brain.decide(self._history, self._tools.all())

            if not decision.wants_tools:
                reply = decision.reply or Utterance(text="", speaker="bmo")
                logger.info("BMO decidio responder: %s", reply.text)
                self._voice.speak(
                    reply.text,
                    on_audio_start=lambda: self._face.show(Expression.TALKING),
                )
                self._history.append(Message(role="assistant", content=reply.text))
                return reply

            logger.info(
                "BMO decidio usar %d tool(s): %s",
                len(decision.tool_calls),
                ", ".join(c.name for c in decision.tool_calls),
            )
            partial = decision.reply.text if decision.reply else ""
            self._history.append(
                Message(
                    role="assistant",
                    content=partial,
                    tool_calls=decision.tool_calls,
                )
            )
            for call in decision.tool_calls:
                logger.info("ejecutando tool '%s' con args=%s", call.name, call.arguments)
                result = self._execute(call, question=text)
                logger.info("tool '%s' devolvio: %s", call.name, result)
                self._history.append(
                    Message(role="tool", content=result, name=call.name)
                )
                if self._tool_is_direct(call.name):
                    logger.info("tool '%s' es directa: su resultado es la respuesta", call.name)
                    self._voice.speak(
                        result,
                        on_audio_start=lambda: self._face.show(Expression.TALKING),
                    )
                    return Utterance(text=result, speaker="bmo")

        logger.warning(
            "corte de seguridad: se agotaron los %d pasos sin respuesta final",
            self._max_steps,
        )
        self._face.show(Expression.SAD)
        fallback = Utterance(
            text="(me colgue pidiendo tools, corte por seguridad)", speaker="bmo"
        )
        self._history.append(Message(role="assistant", content=fallback.text))
        return fallback

    def _tool_is_direct(self, name: str) -> bool:
        """True si la tool devuelve su resultado directo al usuario (sin releerlo el brain)."""
        try:
            return self._tools.get(name).direct
        except KeyError:
            return False

    def _execute(self, call: ToolCall, question: str = "") -> str:
        """Ejecuta una tool y devuelve su resultado como texto para el brain.

        Los errores NO se propagan: se devuelven como texto para que el brain los
        vea y decida que hacer (reintentar, disculparse, usar otra tool...).
        """
        try:
            tool = self._tools.get(call.name)
        except KeyError:
            logger.error("la tool '%s' no existe en el registro", call.name)
            return f"error: la tool '{call.name}' no existe"

        try:
            return str(tool.run(question=question, **call.arguments))
        except Exception as exc:
            logger.exception("fallo ejecutando la tool '%s'", call.name)
            return f"error ejecutando '{call.name}': {exc}"
