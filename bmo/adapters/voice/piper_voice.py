"""piper -> audio PCM -> aplay -> ALSA -> altavoz"""

from __future__ import annotations

import logging
import subprocess
from typing import Callable, List, Optional

from bmo.config import VoiceConfig
from bmo.ports.voice import VoicePort

_LOG = logging.getLogger(__name__)

# run(text, model_path, device, sample_rate, on_audio_start) -> None
RunPipeline = Callable[[str, str, str, int, Optional[Callable[[], None]]], None]


def _piper_command(model_path: str) -> List[str]:
    """Comando de Piper: lee texto por stdin y emite PCM crudo por stdout."""
    return ["piper", "--model", model_path, "--output-raw"]


def _aplay_command(device: str, sample_rate: int) -> List[str]:
    # aplay raw
    cmd = ["aplay", "-q"]
    if device:
        cmd += ["-D", device]
    cmd += ["-r", str(sample_rate), "-f", "S16_LE", "-c", "1", "-t", "raw", "-"] # información del formato para aplay
    return cmd


def _run_pipeline(
    text: str,
    model_path: str,
    device: str,
    sample_rate: int,
    on_audio_start: Optional[Callable[[], None]] = None,
) -> None:
    """Sintetiza con piper, avisa que arranca el audio, y reproduce con aplay.
    """

    # 1. Sintetizar texto (mayor parte de la latencia)
    piper = subprocess.Popen(
        _piper_command(model_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    audio, _ = piper.communicate(text.encode("utf-8"))

    # 2. El sonido está sintetizado y listo para reproducirse: avisamos para sincronizar la cara.
    if on_audio_start is not None:
        on_audio_start()

    # 3. Reproducir el audio ya sintetizado
    aplay = subprocess.Popen(
        _aplay_command(device, sample_rate),
        stdin=subprocess.PIPE,
    )
    aplay.communicate(audio)


class PiperVoice(VoicePort):
    """Voz de BMO con Piper: sintetiza el texto y lo reproduce por ALSA."""

    def __init__(
        self,
        model_path: str,
        device: str = "",
        sample_rate: int = 22050,
        run: Optional[RunPipeline] = None,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._sample_rate = sample_rate
        self._run = run or _run_pipeline

    @classmethod
    def from_config(cls, config: VoiceConfig) -> "PiperVoice":
        return cls(
            model_path=config.model_path,
            device=config.device,
            sample_rate=config.sample_rate,
        )

    def speak(
        self, text: str, on_audio_start: Optional[Callable[[], None]] = None
    ) -> None:
        text = (text or "").strip()
        if not text:
            return
        try:
            self._run(text, self._model_path, self._device, self._sample_rate, on_audio_start)
        except Exception:
            _LOG.warning("no se pudo reproducir la voz (piper/aplay)", exc_info=True)
