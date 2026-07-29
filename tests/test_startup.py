"""Tests del arranque de BMO: el warm-up debe terminar ANTES de atender."""

from __future__ import annotations

from typing import List

from bmo.main import warm_up_then_serve


def test_serve_runs_only_after_warm_up() -> None:
    # Garantia: no se atiende ninguna consulta hasta que el modelo termino de
    # calentarse. El orden SIEMPRE es warm-up primero, servir despues.
    order: List[str] = []

    class Brain:
        def warm_up(self) -> None:
            order.append("warm")

    warm_up_then_serve(Brain(), serve=lambda: order.append("serve"))

    assert order == ["warm", "serve"]


def test_on_warmup_start_fires_before_warm_up() -> None:
    # El hook de inicio (cara durmiendo / aviso) se dispara ANTES del warm-up,
    # asi el usuario ve que BMO esta cargando mientras el modelo entra a RAM.
    order: List[str] = []

    class Brain:
        def warm_up(self) -> None:
            order.append("warm")

    warm_up_then_serve(
        Brain(),
        serve=lambda: order.append("serve"),
        on_warmup_start=lambda: order.append("sleep-face"),
    )

    assert order == ["sleep-face", "warm", "serve"]
