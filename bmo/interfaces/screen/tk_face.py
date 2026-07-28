"""Adapter de la cara: pinta a BMO directo en la pantalla fisica con tkinter.

Diseño de hilos (clave):
- tkinter EXIGE que la ventana y su mainloop vivan en el hilo principal.
- El REPL/Agent corre en un hilo aparte y solo llama a `show(...)`, que se
  limita a guardar la expresion deseada bajo un lock. NO toca tkinter.
- Un tick periodico (`root.after`) lee esa expresion en el hilo principal y
  actualiza la imagen. Asi respetamos que tkinter no es thread-safe.

Los frames son PNGs en `faces/<expresion>/NN.png`. Varias imagenes en una
carpeta = animacion; una sola = cara fija.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

from bmo.domain.models import Expression
from bmo.ports.face import FacePort

_LOG = logging.getLogger(__name__)
_IMG_EXT = {".png", ".gif", ".jpg", ".jpeg"}
_DEFAULT_FACES_DIR = Path(__file__).parent / "faces"


class TkFace(FacePort):
    """Cara de BMO renderizada en pantalla completa con tkinter."""

    def __init__(
        self,
        faces_dir: Optional[Path] = None,
        fullscreen: bool = True,
        fps: int = 4,
    ) -> None:
        self._faces_dir = Path(faces_dir) if faces_dir else _DEFAULT_FACES_DIR
        self._fullscreen = fullscreen
        self._delay_ms = max(1, int(1000 / max(fps, 1)))
        self._lock = threading.Lock()
        self._current = Expression.WARMUP
        self._rendered: Optional[Expression] = None
        self._frame_index = 0
        self._available = False
        self._root = None
        self._label = None
        self._photos: Dict[Expression, List[object]] = {}

    @property
    def available(self) -> bool:
        """True si la ventana pudo inicializarse (hay display y frames)."""
        return self._available

    @property
    def current(self) -> Expression:
        """Expresion que se quiere mostrar ahora (thread-safe)."""
        with self._lock:
            return self._current

    def show(self, expression: Expression) -> None:
        with self._lock:
            self._current = expression

    def _frames_for(self, expression: Expression) -> List[Path]:
        """Lista los frames de una expresion, ordenados por nombre."""
        folder = self._faces_dir / expression.value
        if not folder.is_dir():
            return []
        return sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXT
        )

    def start(self) -> None:
        try:
            import tkinter as tk

            self._root = tk.Tk()
            self._root.title("BMO")
            self._root.configure(bg="black")
            self._root.protocol("WM_DELETE_WINDOW", self.stop)
            self._root.bind("<Escape>", lambda _event: self.stop())
            if self._fullscreen:
                self._root.attributes("-fullscreen", True)
            self._root.config(cursor="none")

            self._label = tk.Label(
                self._root, bg="black", borderwidth=0, highlightthickness=0
            )
            self._label.pack(expand=True, fill="both")

            self._preload()
            self._tick()
            self._available = True
        except Exception:  # noqa: BLE001 - sin display, BMO cae a modo consola
            self._available = False
            self._root = None
            _LOG.warning(
                "no se pudo inicializar la pantalla; BMO sigue solo por consola"
            )

    def run(self) -> None:
        if self._root is not None:
            self._root.mainloop()

    def stop(self) -> None:
        root = self._root
        if root is None:
            return
        try:
            root.after(0, root.destroy)
        except Exception:  # noqa: BLE001 - la ventana ya pudo cerrarse
            pass

    def _preload(self) -> None:
        """Carga y escala todos los frames una sola vez (en el hilo principal)."""
        from PIL import Image, ImageTk

        self._root.update_idletasks()
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()

        for expression in Expression:
            photos: List[object] = []
            for path in self._frames_for(expression):
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:  # noqa: BLE001 - un frame roto no tumba la cara
                    _LOG.warning("no pude cargar el frame %s", path)
                    continue
                scale = min(screen_w / img.width, screen_h / img.height)
                size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                img = img.resize(size, Image.LANCZOS)
                photos.append(ImageTk.PhotoImage(img))
            if photos:
                self._photos[expression] = photos

        if not self._photos:
            _LOG.warning(
                "no encontre ningun frame en %s; la cara va a salir negra. "
                "Verifica que las carpetas de expresiones (neutral, happy, ...) "
                "esten sincronizadas en la Pi",
                self._faces_dir,
            )

    def _tick(self) -> None:
        """Avanza la animacion. Se re-agenda a si mismo con `after`."""
        if self._root is None:
            return

        expression = self.current
        if expression != self._rendered:
            self._rendered = expression
            self._frame_index = 0

        photos = self._photos.get(expression) or self._photos.get(Expression.NEUTRAL)
        if photos:
            frame = photos[self._frame_index % len(photos)]
            self._label.configure(image=frame)
            self._frame_index += 1

        self._root.after(self._delay_ms, self._tick)
