"""Builder de BMO: construye los adapters concretos según config.yaml.

Es el ÚNICO lugar que decide qué adapter se conecta a cada port. Su única
responsabilidad es CONSTRUIR: no arranca nada, no orquesta hilos. De eso se
encarga `run`. El código del dominio no necesita conocerlo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from bmo.config import Config
from bmo.domain.agent import Agent
from bmo.domain.persona import BMO_SYSTEM_PROMPT
from bmo.ports.camera import CameraSourcePort
from bmo.ports.face import FacePort
from bmo.ports.hearing import HearingPort
from bmo.ports.notes import NotesPort
from bmo.ports.vision import VisionPort
from bmo.ports.voice import VoicePort
from bmo.tools.look import build_look_tool
from bmo.tools.notes import build_read_notes_tool, build_save_note_tool
from bmo.tools.tool import ToolRegistry

if TYPE_CHECKING:
    from bmo.adapters.brain.ollama_brain import OllamaBrain


def build_devices(config: Config) -> tuple[CameraSourcePort, VisionPort]:
    """Arma cámara y visión: el IMX500 es UN solo dispositivo que hace ambas.
    """
    from bmo.adapters.camera.ai_camera_imx500 import AiCameraImx500

    shared = AiCameraImx500.from_config(config.camera, config.vision)
    return shared, shared


def build_brain(config: Config) -> "OllamaBrain":
    """Factory del cerebro: elige el adapter según config.
    """
    if config.brain.adapter == "ollama":
        from bmo.adapters.brain.ollama_brain import OllamaBrain

        return OllamaBrain.from_config(config.brain)
    if config.brain.adapter == "hailo":
        from bmo.adapters.brain.hailo_brain import HailoBrain

        return HailoBrain.from_config(config.brain)
    raise ValueError(f"Adapter de cerebro desconocido: {config.brain.adapter}")


def build_face(config: Config) -> Optional[FacePort]:
    """Factory de la cara: arma la pantalla tkinter si está habilitada.

    """
    if not config.screen.enabled:
        return None
    from bmo.interfaces.screen.tk_face import TkFace

    return TkFace(fullscreen=config.screen.fullscreen, fps=config.screen.fps)


def build_voice(config: Config) -> Optional[VoicePort]:
    """Factory de la voz: arma Piper si está habilitada (import lazy).
    """
    if config.voice.adapter == "piper":
        from bmo.adapters.voice.piper_voice import PiperVoice

        return PiperVoice.from_config(config.voice)
    return None


def build_hearing(config: Config) -> Optional[HearingPort]:
    """Factory del oído: arma Vosk si está habilitado (import lazy).
    """
    if config.hearing.adapter == "vosk":
        from bmo.adapters.hearing.vosk_hearing import VoskHearing

        return VoskHearing.from_config(config.hearing)
    return None


def build_notes(config: Config) -> Optional[NotesPort]:
    """Factory de las notas: arma el almacén markdown si está habilitado (lazy).
    """
    if not config.notes.enabled:
        return None
    from pathlib import Path

    from bmo.adapters.memory.markdown_notes import MarkdownNotes

    return MarkdownNotes(Path(config.notes.path))


def build_agent(
    brain,
    camera: CameraSourcePort,
    vision: VisionPort,
    face: Optional[FacePort] = None,
    voice: Optional[VoicePort] = None,
    notes: Optional[NotesPort] = None,
    max_history: int = 0,
) -> Agent:
    """Ensambla el Agente: registra las tools y le inyecta el cerebro.
    """
    registry = ToolRegistry()
    registry.register(build_look_tool(camera, vision))
    if notes is not None:
        registry.register(build_save_note_tool(notes))
        registry.register(build_read_notes_tool(notes))
    return Agent(
        brain=brain,
        tools=registry,
        system_prompt=BMO_SYSTEM_PROMPT,
        face=face,
        voice=voice,
        max_history=max_history,
    )
