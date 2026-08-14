from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Callable, List, Optional

from bmo.config import HearingConfig
from bmo.ports.hearing import HearingPort

_LOG = logging.getLogger(__name__)
Session = Callable[[], str]

_FRAMES_PER_READ = 4000  # frames por lectura, valor recomendado por Vosk
_BYTES_PER_SAMPLE = 2  # S16_LE = 16 bits = 2 bytes por muestra (mono)


def _arecord_command(device: str, sample_rate: int) -> List[str]:
    """Comando arecord para capturar PCM crudo (raw) del micrófono."""
    cmd = ["arecord", "-q"]
    if device:
        cmd += ["-D", device]
    cmd += ["-f", "S16_LE", "-r", str(sample_rate), "-c", "1", "-t", "raw", "-"]
    return cmd


def _build_vosk_session(model_path: str, device: str, sample_rate: int) -> Session:
    """Carga el modelo Vosk UNA vez; captura audio FRESCO en cada escucha.

    El microfono (arecord) NO se deja corriendo de fondo: se abre al empezar a
    escuchar y se cierra al terminar la frase. Asi, mientras BMO piensa o habla,
    no hay stream acumulando audio rancio ni su propia voz (lo que antes
    provocaba overruns y transcripciones basura).
    """
    from vosk import KaldiRecognizer, Model, SetLogLevel

    SetLogLevel(0 if logging.getLogger("bmo").isEnabledFor(logging.DEBUG) else -1)
    model = Model(model_path)

    def listen_one() -> str:
        # recognizer nuevo por turno: estado de decodificacion limpio.
        recognizer = KaldiRecognizer(model, sample_rate)
        # abrir arecord para capturar audio
        mic = subprocess.Popen(
            _arecord_command(device, sample_rate),
            stdout=subprocess.PIPE,
        )
        try:
            if mic.stdout is None:
                return ""
            return _capture_phrase(
                mic.stdout, recognizer, _FRAMES_PER_READ * _BYTES_PER_SAMPLE
            )
        finally:
            # cerrar arecord: sin stream de fondo que acumule audio entre turnos.
            mic.terminate()
            try:
                mic.wait(timeout=1)
            except subprocess.TimeoutExpired:
                mic.kill()

    return listen_one


def _capture_phrase(stream: Any, recognizer: Any, chunk_bytes: int) -> str:
    """Lee bloques de audio de `stream` hasta que Vosk detecta fin de frase.

    Devuelve el texto transcrito, o "" si el stream se agota antes.
    """
    while True:
        data = stream.read(chunk_bytes)
        if not data:
            return ""
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            return str(result.get("text", "")).strip()



class VoskHearing(HearingPort):
    """Oído de BMO: transcribe el micrófono con Vosk (offline)."""

    def __init__(
        self,
        model_path: str,
        device: str = "",
        sample_rate: int = 16000,
        session: Optional[Session] = None,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._sample_rate = sample_rate
        self._session = session

    @classmethod
    def from_config(cls, config: HearingConfig) -> "VoskHearing":
        return cls(
            model_path=config.model_path,
            device=config.device,
            sample_rate=config.sample_rate,
        )

    def listen(self) -> str:
        try:
            return self._ensure_session()()
        except Exception:
            _LOG.warning("no se pudo escuchar (vosk/arecord)", exc_info=True)
            return ""

    @property
    def available(self) -> bool:
        """True si BMO puede escuchar: el modelo Vosk debe existir en disco."""
        if self._session is not None:
            return True
        return os.path.isdir(self._model_path)

    def _ensure_session(self) -> Session:
        """Arma la sesión Vosk la primera vez y la reutiliza después."""
        if self._session is None:
            self._session = _build_vosk_session(
                self._model_path, self._device, self._sample_rate
            )
        return self._session
