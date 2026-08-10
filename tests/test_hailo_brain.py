"""Tests del HailoBrain (cerebro sobre el NPU Hailo-10H del AI HAT+ 2).

Foco: el servidor `hailo-ollama` NO filtra los tokens especiales de la
plantilla de Llama 3 (<|start_header_id|>, <|eot_id|>, ...) y encima el modelo
suele alucinar un turno nuevo detrás de ellos. BMO no debe decir esa basura.
"""

from __future__ import annotations

from typing import Any, List

from bmo.adapters.brain.hailo_brain import HailoBrain, HailoOllamaClient
from bmo.domain.models import Message


class FakeClient:
    """Cliente ollama falso: devuelve un content fijo y registra las llamadas."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: List[dict[str, Any]] = []

    def chat(self, model: str, messages: list, **kwargs: Any) -> dict:
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return {"message": {"content": self._content}}


class FakeResponse:
    """Respuesta httpx falsa."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


class FakeHttp:
    """Cliente httpx falso: captura el último POST hecho."""

    def __init__(self, data: dict) -> None:
        self._data = data
        self.last: dict[str, Any] = {}

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.last = {"url": url, "json": json, "timeout": timeout}
        return FakeResponse(self._data)


class FakeTool:
    """Tool mínima para ejercitar el ACTION MODE."""

    name = "look"
    description = "see the world in front of you"


class FakeNoteTool:
    """Tool con argumentos: ejercita el ACTION MODE con campos extra."""

    name = "save_note"
    description = "write down a note to remember it"
    parameters = {"content": {"type": "string", "description": "the note text"}}


def _user(content: str) -> Message:
    return Message(role="user", content=content)


def test_sanitize_content_cuts_from_first_special_token() -> None:
    brain = HailoBrain(model="m", host="x", client=FakeClient(""))

    assert brain._sanitize_content("Hi there<|eot_id|>junk") == "Hi there"
    assert brain._sanitize_content("no tokens here") == "no tokens here"


def test_sanitize_content_removes_qwen3_think_block() -> None:
    # qwen3 es un thinking model: razona entre <think>...</think> ANTES de la
    # respuesta real. hailo-ollama NO filtra nada, asi que BMO diria su
    # monologo interno en voz alta. Solo debe quedar la respuesta final.
    brain = HailoBrain(model="qwen3:1.7b", host="x", client=FakeClient(""))

    dirty = "<think>The user greets me, I should be friendly.</think>Hi there!"
    assert brain._sanitize_content(dirty) == "Hi there!"


def test_sanitize_content_removes_multiline_think_block() -> None:
    brain = HailoBrain(model="qwen3:1.7b", host="x", client=FakeClient(""))

    dirty = "<think>\nline one\nline two\n</think>\nHello!"
    assert brain._sanitize_content(dirty) == "Hello!"


def test_sanitize_content_drops_unclosed_think_block() -> None:
    # Si la respuesta se corta a mitad del razonamiento (sin </think>), no
    # debe quedar nada del monologo: se descarta desde <think> en adelante.
    brain = HailoBrain(model="qwen3:1.7b", host="x", client=FakeClient(""))

    dirty = "Ready!<think>still thinking and never closed..."
    assert brain._sanitize_content(dirty) == "Ready!"


def test_sanitize_content_removes_think_and_llama_tokens_together() -> None:
    brain = HailoBrain(model="qwen3:1.7b", host="x", client=FakeClient(""))

    dirty = "<think>reasoning</think>Real answer<|eot_id|>hallucinated turn"
    assert brain._sanitize_content(dirty) == "Real answer"


def test_chat_reply_strips_llama_special_tokens() -> None:
    # hailo-ollama deja pasar <|start_header_id|>assistant... y el modelo
    # alucina "How can I help?": todo eso se descarta, queda solo "Hello!".
    dirty = "Hello!<|start_header_id|>assistant<|end_header_id|>\nHow can I help?"
    brain = HailoBrain(model="llama3.2:3b", host="x", client=FakeClient(dirty))

    decision = brain.decide([_user("hi")], tools=[])

    assert decision.reply is not None
    assert decision.reply.text == "Hello!"


def test_action_mode_extracts_action_despite_trailing_tokens() -> None:
    # En ACTION MODE el modelo responde el JSON; aunque venga con tokens
    # especiales pegados detrás, la accion se extrae igual.
    client = FakeClient('{"action": "look"}<|eot_id|>assistant')
    brain = HailoBrain(model="llama3.2:3b", host="x", client=client)

    decision = brain.decide([_user("what do you see?")], tools=[FakeTool()])

    assert decision.tool_calls
    assert decision.tool_calls[0].name == "look"


def test_action_mode_captures_arguments() -> None:
    # El modelo devuelve el nombre de la accion Y el contenido: ambos deben
    # viajar en el ToolCall. Sin esto, save_note se ejecutaria sin texto.
    client = FakeClient('{"action": "save_note", "content": "buy milk"}')
    brain = HailoBrain(model="qwen3:1.7b", host="x", client=client)

    decision = brain.decide([_user("remember to buy milk")], tools=[FakeNoteTool()])

    assert decision.tool_calls
    call = decision.tool_calls[0]
    assert call.name == "save_note"
    assert call.arguments == {"content": "buy milk"}


def test_warm_up_is_inherited_from_ollama_brain() -> None:
    # HailoBrain reusa el warm-up de OllamaBrain sin cambios.
    client = FakeClient("hi")
    brain = HailoBrain(model="llama3.2:3b", host="x", client=client)

    brain.warm_up()

    assert len(client.calls) == 1


def test_hailo_client_sends_only_minimal_fields() -> None:
    # El parser oatpp de hailo-ollama explota (500 'Node is NOT a STRING') si el
    # request trae `tools` u `options`. El cliente debe mandar SOLO lo minimo.
    http = FakeHttp({"message": {"content": "hola"}})
    client = HailoOllamaClient("http://localhost:8000", http=http)

    out = client.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": "hi"}],
        options={"temperature": 0.2},   # debe descartarse
        tools=[{"whatever": 1}],         # debe descartarse
    )

    assert out == {"message": {"content": "hola"}}
    body = http.last["json"]
    assert set(body.keys()) == {"model", "messages", "stream"}
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_hailo_client_targets_api_chat_endpoint() -> None:
    http = FakeHttp({"message": {"content": "x"}})
    client = HailoOllamaClient("http://localhost:8000", http=http)

    client.chat(model="m", messages=[{"role": "user", "content": "hi"}])

    assert http.last["url"] == "http://localhost:8000/api/chat"


def test_hailo_client_normalizes_messages_to_role_and_content() -> None:
    # Aunque un message traiga campos extra, solo role y content viajan.
    http = FakeHttp({"message": {"content": "x"}})
    client = HailoOllamaClient("http://localhost:8000", http=http)

    client.chat(
        model="m",
        messages=[{"role": "system", "content": "be nice", "images": []}],
    )

    assert http.last["json"]["messages"] == [{"role": "system", "content": "be nice"}]


def test_hailo_client_normalizes_newlines_in_content() -> None:
    # El renderer interno de hailo-ollama re-serializa los mensajes a JSON para
    # su template de chat y NO escapa los saltos de linea: un system prompt
    # multilinea rompe el parseo (control character U+000A must be escaped) y
    # devuelve 500 HAILO_INTERNAL_FAILURE. El cliente debe colapsar los control
    # chars (\n, \r, \t) a un solo espacio antes de enviar.
    http = FakeHttp({"message": {"content": "x"}})
    client = HailoOllamaClient("http://localhost:8000", http=http)

    client.chat(
        model="qwen3:1.7b",
        messages=[
            {"role": "system", "content": "You are BMO.\nBe cute.\r\nBe short.\tOk"}
        ],
    )

    sent = http.last["json"]["messages"][0]["content"]
    assert "\n" not in sent
    assert "\r" not in sent
    assert "\t" not in sent
    assert sent == "You are BMO. Be cute. Be short. Ok"

