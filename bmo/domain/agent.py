"""El Agente: el orquestador de BMO (el 'conductor').

Responsabilidades (lo que el brain NO hace):
- Lleva la memoria de la conversacion (lista de Message).
- Corre el loop: pregunta al brain -> ejecuta las tools que pida -> vuelve a
  preguntar -> hasta que el brain da una respuesta final.
- Ejecuta las tools de verdad (via ToolRegistry) y le devuelve el resultado al
  brain. Si una tool falla, le pasa el error al brain para que se recupere.
- Se protege de loops infinitos con un tope de pasos (max_steps).

El agente no conoce ningun LLM concreto: solo le pide `.decide(...)` al brain que
le inyectan. Cambiar de Ollama al Hailo no toca ni una linea de este archivo.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from bmo.domain.models import Expression, Message, ToolCall, Utterance
from bmo.ports.face import FacePort, NullFace
from bmo.tools.tool import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    """Orquesta la conversacion entre el humano, el brain y las tools."""

    def __init__(
        self,
        brain,  # cualquier objeto con decide(messages, tools) -> BrainDecision
        tools: ToolRegistry,
        system_prompt: str = "",
        max_steps: int = 5,
        face: Optional[FacePort] = None,
    ) -> None:
        self._brain = brain
        self._tools = tools
        self._max_steps = max_steps
        # Null Object: si no hay pantalla, `face.show(...)` no hace nada y el
        # loop de abajo no necesita chequear `if face is not None`.
        self._face = face or NullFace()
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
        # BMO se pone a pensar en cuanto recibe el mensaje.
        self._face.show(Expression.THINKING)

        for step in range(self._max_steps):
            logger.info("paso %d/%d: BMO esta pensando...", step + 1, self._max_steps)
            decision = self._brain.decide(self._history, self._tools.all())

            if not decision.wants_tools:
                reply = decision.reply or Utterance(text="", speaker="bmo")
                logger.info("BMO decidio responder: %s", reply.text)
                # Tiene respuesta final: pasa a 'hablando'.
                self._face.show(Expression.TALKING)
                self._history.append(Message(role="assistant", content=reply.text))
                return reply

            # El brain pidio tools: primero registramos SU turno (con los
            # tool_calls), y despues los resultados. Ese orden es el que Ollama
            # espera; saltearlo confunde al modelo.
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
                result = self._execute(call)
                logger.info("tool '%s' devolvio: %s", call.name, result)
                self._history.append(
                    Message(role="tool", content=result, name=call.name)
                )

        # Corte de seguridad: el brain siguio pidiendo tools sin cerrar.
        logger.warning(
            "corte de seguridad: se agotaron los %d pasos sin respuesta final",
            self._max_steps,
        )
        # Algo salio mal: cara triste.
        self._face.show(Expression.SAD)
        fallback = Utterance(
            text="(me colgue pidiendo tools, corte por seguridad)", speaker="bmo"
        )
        self._history.append(Message(role="assistant", content=fallback.text))
        return fallback

    def _execute(self, call: ToolCall) -> str:
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
            return str(tool.run(**call.arguments))
        except Exception as exc:  # el resultado (error incluido) vuelve al brain
            logger.exception("fallo ejecutando la tool '%s'", call.name)
            return f"error ejecutando '{call.name}': {exc}"
