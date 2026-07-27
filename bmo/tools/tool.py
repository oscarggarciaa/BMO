"""Tools invocables por el cerebro y su registro."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    """Una capacidad invocable por el cerebro.

    - name: identificador que usa el LLM para llamarla.
    - description: que hace (el LLM lo lee para decidir cuando usarla).
    - parameters: JSON-schema de los argumentos (formato tool-calling de Ollama).
    - func: la funcion Python real que ejecuta la accion.
    - direct: si es True, el resultado de la tool ES la respuesta final para el
      usuario y NO vuelve a pasar por el cerebro (evita el 'telefono descompuesto').
    """

    name: str
    description: str
    func: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    direct: bool = False

    def run(self, **kwargs: Any) -> Any:
        sig = inspect.signature(self.func)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return self.func(**kwargs)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return self.func(**allowed)

    def to_schema(self) -> Dict[str, Any]:
        """Formato que espera Ollama/OpenAI para tool calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }


class ToolRegistry:
    """Catalogo de tools disponibles para el cerebro."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]
