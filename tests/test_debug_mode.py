"""Tests de los modos de ejecucion: normal vs debug.

Contrato:
- MODO NORMAL (debug=False): silencio TOTAL de logs. La entrada (USER>) y la
  salida (BMO>) se muestran con print en la capa de presentacion, no por log.
  Si algo falla, se arranca con `bmo debug`.
- MODO DEBUG (debug=True): el log muestra TODO (BMO interno + terceros).

En normal la raiz sube a CRITICAL (calla a todos, incluido BMO y terceros).
En debug baja a DEBUG. libcamera es C: se controla por env var.
"""

from __future__ import annotations

import logging
import os

from bmo.config import Config
from bmo.domain.agent import Agent
from bmo.domain.models import BrainDecision, Utterance
from bmo.main import resolve_debug, setup_logging
from bmo.tools.tool import ToolRegistry


class _ReplyBrain:
    """Brain falso: responde texto directo, sin pedir tools."""

    def decide(self, messages, tools) -> BrainDecision:
        return BrainDecision(reply=Utterance(text="hi there", speaker="bmo"))


# --- config: el flag debug --------------------------------------------------


def test_config_debug_defaults_to_false() -> None:
    assert Config.from_dict({}).debug is False


def test_config_reads_debug_flag() -> None:
    assert Config.from_dict({"debug": True}).debug is True


# --- setup_logging: normal calla TODO, debug muestra todo -------------------


def test_normal_mode_is_fully_silent() -> None:
    # En normal NO sale NADA por log: ni BMO ni terceros (ni sus warnings).
    # La entrada/salida se muestra con print aparte.
    setup_logging(debug=False)

    assert logging.getLogger("bmo.domain.agent").isEnabledFor(logging.INFO) is False
    assert logging.getLogger("picamera2").isEnabledFor(logging.WARNING) is False


def test_debug_mode_shows_everything() -> None:
    # En debug se ve TODO: BMO interno y terceros.
    setup_logging(debug=True)

    assert logging.getLogger("bmo.domain.agent").isEnabledFor(logging.DEBUG) is True
    assert logging.getLogger("picamera2").isEnabledFor(logging.INFO) is True


def test_libcamera_env_quiet_in_normal_verbose_in_debug() -> None:
    # libcamera es C: su verbosidad se fija por variable de entorno.
    setup_logging(debug=False)
    assert os.environ["LIBCAMERA_LOG_LEVELS"] == "*:FATAL"

    setup_logging(debug=True)
    assert os.environ["LIBCAMERA_LOG_LEVELS"] == "*:INFO"


# --- resolve_debug: 'bmo debug' fuerza debug; 'bmo' a secas usa la config ---


def test_cli_debug_forces_debug_mode() -> None:
    # 'bmo debug' fuerza modo debug aunque la config diga lo contrario.
    assert resolve_debug(["debug"], config_debug=False) is True


def test_no_cli_arg_falls_back_to_config() -> None:
    # Sin argumento ('bmo' a secas), manda lo que diga la config.
    assert resolve_debug([], config_debug=True) is True
    assert resolve_debug([], config_debug=False) is False


def test_unknown_cli_arg_falls_back_to_config() -> None:
    # Un argumento desconocido no rompe: se ignora y manda la config.
    assert resolve_debug(["banana"], config_debug=True) is True


def test_cli_arg_is_case_insensitive() -> None:
    assert resolve_debug(["DEBUG"], config_debug=False) is True


# --- agent: TODO su log es DEBUG (diagnostico); en normal es silencioso -----


def test_agent_logs_everything_at_debug_never_info() -> None:
    agent = Agent(_ReplyBrain(), ToolRegistry())

    logger = logging.getLogger("bmo.domain.agent")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        agent.ask("hello bmo")
    finally:
        logger.removeHandler(handler)

    # NADA del agente sale a INFO+: en modo normal el agente es mudo por log.
    assert [r for r in records if r.levelno >= logging.INFO] == []

    # El detalle (entrada, salida y pasos) vive en DEBUG, para el modo debug.
    debugs = [r.getMessage() for r in records if r.levelno == logging.DEBUG]
    assert any("hello bmo" in m for m in debugs)
    assert any("hi there" in m for m in debugs)
    assert any("paso" in m for m in debugs)
