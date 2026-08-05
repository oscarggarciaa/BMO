"""Adapter de voz con Piper (TTS neuronal offline) hacia una salida ALSA.

Piper genera audio PCM crudo (s16le, mono) y lo emite por stdout con
`--output-raw`. Ese stream se conecta directo a `aplay` (ALSA), que lo
reproduce por la tarjeta elegida (`-D`, p. ej. la USB). Todo local, sin nube.

    piper --model VOZ.onnx --output-raw | aplay -D plughw:1,0 -r 22050 ...

El comando se arma en funciones puras (`_piper_command`, `_aplay_command`) para
poder testearlo, y la ejecución real (`_run_pipeline`) es inyectable, así los
tests no necesitan piper ni una placa de sonido real.
"""

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
    """Comando de aplay para reproducir PCM crudo (s16le mono) por ALSA.

    Si `device` viene vacío, usa la salida por defecto del sistema; si trae
    algo (p. ej. `plughw:1,0`), fuerza esa tarjeta con `-D`.
    """
    cmd = ["aplay", "-q"]
    if device:
        cmd += ["-D", device]
    cmd += ["-r", str(sample_rate), "-f", "S16_LE", "-c", "1", "-t", "raw", "-"]
    return cmd


def _run_pipeline(
    text: str,
    model_path: str,
    device: str,
    sample_rate: int,
    on_audio_start: Optional[Callable[[], None]] = None,
) -> None:
    """Sintetiza con piper, avisa que arranca el audio, y reproduce con aplay.

    A diferencia de un pipe en streaming, aquí se sintetiza TODO primero. Toda la
    latencia (cargar el modelo onnx + generar) queda ANTES de reproducir, así el
    callback `on_audio_start` cae justo cuando el sonido empieza realmente y la
    cara se sincroniza con la voz (no habla en silencio mientras piper carga).
    """
    # 1. Sintetizar: aquí vive la latencia (carga del modelo + inferencia).
    #    stderr -> DEVNULL: piper carga onnxruntime, que escupe warnings de
    #    descubrimiento de GPU ("Failed to detect devices under /sys/class/drm")
    #    irrelevantes en la Pi. Se silencian para no ensuciar la consola.
    piper = subprocess.Popen(
        _piper_command(model_path),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    audio, _ = piper.communicate(text.encode("utf-8"))

    # 2. El sonido está por salir: avisamos para sincronizar la cara.
    if on_audio_start is not None:
        on_audio_start()

    # 3. Reproducir el audio ya sintetizado (arranca casi al instante).
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
        except Exception:  # noqa: BLE001 - un fallo de audio no debe interrumpir la conversación
            _LOG.warning("no se pudo reproducir la voz (piper/aplay)", exc_info=True)
