"""Adapter de oído con Vosk: micrófono -> texto, offline y liviano.

Vosk es un STT offline que funciona bien en la Pi (modelo small ~40MB). Un solo
motor sirve para DETECTAR 'hello' y para transcribir el comando: el audio se
captura con `arecord` (ya probado con el conjunto USB) y se transcribe en
streaming hasta que Vosk detecta el fin de la frase (silencio).

Igual que PiperVoice, tiene una costura inyectable (`session`) para poder
testear la lógica sin tener vosk ni micrófono: los tests pasan una sesión
falsa; en la Pi se arma la real (import lazy de vosk).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Callable, List, Optional

from bmo.config import HearingConfig
from bmo.ports.hearing import HearingPort

_LOG = logging.getLogger(__name__)

# Una sesión de escucha: cada llamada bloquea y devuelve UNA frase transcrita.
# Encapsula el modelo Vosk + el stream de arecord, ya inicializados una vez.
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
    """Carga el modelo Vosk y abre el micrófono UNA vez; devuelve la sesión.
    """
    from vosk import KaldiRecognizer, Model  # lazy: solo en la Pi

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, sample_rate)
    mic = subprocess.Popen(
        _arecord_command(device, sample_rate),
        stdout=subprocess.PIPE,
    )
    # modulo principal para escucha
    def listen_one() -> str:
        if mic.stdout is None:
            return ""
        while True:
            # lee un bloque de audio
            data = mic.stdout.read(_FRAMES_PER_READ * _BYTES_PER_SAMPLE)
            if not data:
                return ""
            # devuelve texto transcrito en fin de frase
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                return str(result.get("text", "")).strip()

    return listen_one


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
        except Exception:  # noqa: BLE001 - un fallo de audio no debe interrumpir a BMO
            _LOG.warning("no se pudo escuchar (vosk/arecord)", exc_info=True)
            return ""

    @property
    def available(self) -> bool:
        """True si BMO puede escuchar: el modelo Vosk debe existir en disco.

        Con una sesión inyectada (tests) se asume disponible. Si el modelo no
        está descargado, devuelve False para que el arranque recurra al teclado
        en vez de entrar en un bucle de fallos.
        """
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
