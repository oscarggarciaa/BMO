"""Cerebro de BMO sobre un servidor Ollama local (LLM offline)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional, Set

from bmo.config import BrainConfig
from bmo.domain.models import BrainDecision, Message, ToolCall, Utterance
from bmo.tools.tool import Tool

# Rangos Unicode de emojis/pictogramas. Un modelo pequeno ignora "no emojis" en
# el prompt, asi que los quitamos aqui (el TTS de Piper los lee como ruido).
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001faff"  # simbolos, pictogramas, emoticonos, transporte
    "\U00002600-\U000027bf"  # simbolos varios + dingbats
    "\U0001f1e6-\U0001f1ff"  # indicadores regionales (banderas)
    "\U00002b00-\U00002bff"  # simbolos y flechas varios (estrellas...)
    "\U0000fe00-\U0000fe0f"  # selectores de variacion
    "\U0000200d"  # zero-width joiner (une emojis compuestos)
    "]+",
    flags=re.UNICODE,
)


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

    def warm_up(self) -> None:
        """Carga el modelo en RAM con una consulta mínima al arrancar.
        """
        try:
            self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "el warm-up del modelo falló (Ollama no está listo); continúo igual",
                exc_info=True,
            )

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
                "repeat_penalty": 1.3,
            },
        )

        text = response["message"].get("content") or ""
        action = self._extract_action(text, {t.name for t in tools})
        if action is not None:
            return BrainDecision(tool_calls=(ToolCall(name=action),))
        reply = self._clean_reply(self._sanitize_content(text))
        return BrainDecision(reply=Utterance(text=reply, speaker="bmo"))

    def _sanitize_content(self, text: str) -> str:
        """Hook para post-procesar el content crudo del modelo.

        En OllamaBrain no hace nada (Ollama ya filtra los tokens de plantilla).
        HailoBrain lo sobrescribe: el servidor hailo-ollama NO los filtra.
        """
        return text

    @staticmethod
    def _build_action_protocol(tools: List[Tool]) -> str:
        """Arma las instrucciones de tool-calling por prompt desde las tools.
        """
        if not tools:
            return ""
        lines = [
            "# HOW TO RESPOND",
            "You have TWO modes and in every message you pick EXACTLY ONE:",
            "",
            "- CHAT MODE (default): reply with normal TEXT, short and with your "
            "personality. This is what you use almost always.",
            "- ACTION MODE: run ONE tool. Only when it is ESSENTIAL to be able "
            "to answer.",
            "",
            "## When to use ACTION MODE",
            "Use an action ONLY if ALL of these are true:",
            "1. The user asks for something you CANNOT know or answer without the "
            "tool (for example, seeing the real world right now).",
            "2. There is an action below that does EXACTLY what is asked.",
            "3. You are SURE. If you doubt even a little, pick CHAT MODE.",
            "",
            "## VISION vs IDENTITY (read this carefully)",
            "Do NOT confuse these two:",
            "- Questions about WHAT YOU SEE, the real world in front of you, or "
            "the camera (for example 'what do you see', 'what are you looking "
            "at', 'can you see me', 'what is in front of you') -> ACTION MODE.",
            "- Questions about WHO YOU ARE, how you feel or your opinions (for "
            "example 'who are you', 'how are you', 'what can you do') -> CHAT "
            "MODE.",
            "",
            "These VISION VERBS all mean the same thing and ALL trigger ACTION "
            "MODE: looking, watching, seeing, staring, gazing, observing. If the "
            "user asks what you are looking/watching/staring at, or what you can "
            "see, ALWAYS use ACTION.",
            "",
            "GOLDEN RULE: when in doubt, CHAT. Greetings, questions about you, "
            "opinions, jokes and anything you can answer from memory ALWAYS go "
            "in CHAT MODE, NEVER with an action. The ONLY exception is looking at "
            "the real world: if the user asks what you see, ALWAYS use ACTION.",
            "",
            "## Exact format of an action",
            "To run an action reply ONLY with this JSON, on a single line and "
            "with absolutely nothing else before or after:",
            '{"action": "exact_action_name"}',
            "No explanations, greetings or extra text: ONLY the JSON.",
            "",
            "## After an action",
            "When you receive a message starting with [result of ...], that is "
            "the result of your action. Use it to answer the user in CHAT MODE "
            "(normal text). Do NOT ask for the same action again.",
            "A vision result is a ONE-TIME snapshot: it answers ONLY the "
            "question that triggered the look. For any NEW or different "
            "question, do NOT repeat a previous vision result; answer the new "
            "question on its own.",
            "",
            "## KNOWLEDGE questions",
            "Questions about facts, places, history or general knowledge (for "
            "example 'what do you know about the united states', 'tell me about "
            "dogs') are CHAT MODE. Try to answer them briefly from your own "
            "knowledge in one or two cheerful sentences. Only say you are not "
            "sure if you truly do not know. NEVER use an action for these and "
            "NEVER answer them with what you saw.",
            "",
            "## Available actions",
        ]
        for tool in tools:
            lines.append(f'- "{tool.name}": {tool.description}')
        first = tools[0].name
        lines += [
            "",
            "## STYLE",
            "Answer ONLY what the user asked, nothing more. Do NOT introduce "
            "yourself or repeat who you are unless the user literally asks 'who "
            "are you'. NEVER start an answer with your name or 'I'm BMO' — "
            "especially a vision answer: just say what you see.",
            "",
            "## Examples",
            "These go in CHAT MODE (text, NEVER an action):",
            "  User: hi -> Hi! How are you?",
            "  User: how are you? -> Great, ready to play!",
            "  User: tell me a joke -> Why did the computer go to the beach? "
            "To surf the web!",
            "  User: what day is it today? -> Oh, I don't know that, but we can "
            "play!",
            "  User: what do you know about the united states? -> It's a big "
            "country in North America with lots of cities and people!",
            "These go in ACTION MODE (only the JSON, nothing else):",
            f'  User: what do you see? -> {{"action": "{first}"}}',
            f'  User: what are you looking at? -> {{"action": "{first}"}}',
            f'  User: what are you watching? -> {{"action": "{first}"}}',
            f'  User: what are you staring at? -> {{"action": "{first}"}}',
            f'  User: can you see me? -> {{"action": "{first}"}}',
            f'  User: what is in front of you? -> {{"action": "{first}"}}',
            f'  User: look and tell me what is in front -> {{"action": "{first}"}}',
            f'  User: take a picture -> {{"action": "{first}"}}',
        ]
        return "\n".join(lines)

    @staticmethod
    def _extract_action(text: str, valid_names: Set[str]) -> Optional[str]:
        """Busca un JSON {"action": "..."} en el texto y valida la acción."""
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
        """Limpia bloques JSON residuales y emojis de una respuesta de texto.
        """
        cleaned = re.sub(r"\{.*\}", "", text, flags=re.DOTALL)
        cleaned = _EMOJI_PATTERN.sub("", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" +([.,!?])", r"\1", cleaned).strip()
        return cleaned or "sorry, I didn't get that. can you say it again?"

    @staticmethod
    def _to_ollama(message: Message) -> dict:
        """Traduce un Message del dominio al formato que espera este modelo.
        """
        if message.role == "assistant" and message.tool_calls:
            call = message.tool_calls[0]
            return {"role": "assistant", "content": json.dumps({"action": call.name})}
        if message.role == "tool":
            etiqueta = message.name or "action"
            return {
                "role": "user",
                "content": f"[result of {etiqueta}]: {message.content}",
            }
        return {"role": message.role, "content": message.content}
