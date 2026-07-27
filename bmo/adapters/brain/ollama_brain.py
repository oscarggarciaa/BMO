"""Cerebro de BMO sobre un servidor Ollama local (LLM offline)."""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Set

from bmo.config import BrainConfig
from bmo.domain.models import BrainDecision, Message, ToolCall, Utterance
from bmo.tools.tool import Tool


class OllamaBrain:
    """El cerebro de BMO sobre un servidor Ollama local."""

    def __init__(self, model: str, host: str, client: Optional[Any] = None) -> None:
        self._model = model
        if client is None:
            import ollama

            client = ollama.Client(host=host)
        self._client = client

    @classmethod
    def from_config(cls, config: BrainConfig) -> "OllamaBrain":
        return cls(model=config.model, host=config.host)

    def decide(self, messages: List[Message], tools: List[Tool]) -> BrainDecision:
        ollama_messages = [self._to_ollama(m) for m in messages]
        protocol = self._build_action_protocol(tools)
        if protocol:
            if ollama_messages and ollama_messages[0]["role"] == "system":
                ollama_messages[0] = {
                    "role": "system",
                    "content": ollama_messages[0]["content"] + "\n\n" + protocol,
                }
            else:
                ollama_messages.insert(0, {"role": "system", "content": protocol})

        response = self._client.chat(
            model=self._model,
            messages=ollama_messages,
            options={
                "temperature": 0.2,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        )

        text = response["message"].get("content") or ""
        action = self._extract_action(text, {t.name for t in tools})
        if action is not None:
            return BrainDecision(tool_calls=(ToolCall(name=action),))
        return BrainDecision(reply=Utterance(text=self._clean_reply(text), speaker="bmo"))

    @staticmethod
    def _build_action_protocol(tools: List[Tool]) -> str:
        """Arma las instrucciones de tool-calling por prompt desde las tools.

        Dinamico: si maniana agregas una tool nueva, el protocolo se actualiza
        solo. El modelo lee esto para saber que puede hacer y como pedirlo.
        """
        if not tools:
            return ""
        lines = [
            "# COMO RESPONDER",
            "Tenes DOS modos y en cada mensaje elegis EXACTAMENTE UNO:",
            "",
            "- MODO CHARLA (por defecto): responde con TEXTO normal, corto y con "
            "tu personalidad. Es lo que usas casi siempre.",
            "- MODO ACCION: ejecuta UNA herramienta. Solo cuando es "
            "IMPRESCINDIBLE para poder responder.",
            "",
            "## Cuando usar MODO ACCION",
            "Usa una accion SOLO si se cumplen TODAS estas condiciones:",
            "1. El usuario pide algo que NO podes saber ni responder sin la "
            "herramienta (por ejemplo, ver el mundo real ahora mismo).",
            "2. Existe abajo una accion que hace EXACTAMENTE lo que pide.",
            "3. Estas SEGURO. Si dudas aunque sea un poco, elegis MODO CHARLA.",
            "",
            "REGLA DE ORO: ante la duda, CHARLA. Saludos, preguntas sobre vos, "
            "opiniones, chistes y cualquier cosa que puedas responder de memoria "
            "van SIEMPRE en MODO CHARLA, NUNCA con una accion.",
            "",
            "## Formato exacto de una accion",
            "Para ejecutar una accion responde UNICAMENTE con este JSON, en una "
            "sola linea y sin absolutamente nada mas antes ni despues:",
            '{"action": "nombre_exacto_de_la_accion"}',
            "Nada de explicaciones, saludos ni texto extra: SOLO el JSON.",
            "",
            "## Despues de una accion",
            "Cuando recibas un mensaje que empieza con [resultado de ...], ese es "
            "el resultado de tu accion. Usalo para responder al usuario en MODO "
            "CHARLA (texto normal). NO vuelvas a pedir la misma accion.",
            "",
            "## Acciones disponibles",
        ]
        for tool in tools:
            lines.append(f'- "{tool.name}": {tool.description}')
        first = tools[0].name
        lines += [
            "",
            "## Ejemplos",
            "Estos van en MODO CHARLA (texto, NUNCA una accion):",
            "  Usuario: hola -> Hola! Como andas?",
            "  Usuario: como estas? -> Muy bien, con ganas de jugar!",
            "  Usuario: quien sos? -> Soy BMO, tu amiguito robot!",
            "  Usuario: contame un chiste -> Que hace una compu en la playa? "
            "Toma sol-uciones!",
            "  Usuario: que dia es hoy? -> Uy, eso no lo se, pero podemos jugar!",
            "Estos van en MODO ACCION (solo el JSON, nada mas):",
            f'  Usuario: que ves? -> {{"action": "{first}"}}',
            f'  Usuario: mira y deci que hay adelante -> {{"action": "{first}"}}',
            f'  Usuario: sacame una foto -> {{"action": "{first}"}}',
        ]
        return "\n".join(lines)

    @staticmethod
    def _extract_action(text: str, valid_names: Set[str]) -> Optional[str]:
        """Busca un JSON {"action": "..."} en el texto y valida la accion."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        action = str(data.get("action", "")).strip()
        return action if action in valid_names else None

    @staticmethod
    def _clean_reply(text: str) -> str:
        """Limpia bloques JSON residuales de una respuesta de texto.

        Si el modelo tira un JSON con una accion invalida (o texto + JSON basura),
        no queremos escupirle el JSON crudo al usuario. Lo sacamos; si no queda
        nada, devolvemos una disculpa amable.
        """
        cleaned = re.sub(r"\{.*\}", "", text, flags=re.DOTALL).strip()
        return cleaned or "perdon, no te entendi. me lo repetis?"

    @staticmethod
    def _to_ollama(message: Message) -> dict:
        """Traduce un Message del dominio al formato que espera este modelo.

        Como NO usamos tool-calling nativo, traducimos el historial a texto puro:
        - el turno del assistant que pidio una accion se reconstruye como el JSON
          que "dijo", para que el historial sea coherente;
        - el resultado de una tool se pasa como mensaje de usuario etiquetado.
        """
        if message.role == "assistant" and message.tool_calls:
            call = message.tool_calls[0]
            return {"role": "assistant", "content": json.dumps({"action": call.name})}
        if message.role == "tool":
            etiqueta = message.name or "accion"
            return {
                "role": "user",
                "content": f"[resultado de {etiqueta}]: {message.content}",
            }
        return {"role": message.role, "content": message.content}
