"""Adapter de la cara: pinta a BMO directo en la pantalla física con tkinter.

Diseño de hilos (clave):
- tkinter EXIGE que la ventana y su mainloop vivan en el hilo principal.
- El REPL/Agent corre en un hilo aparte y solo llama a `show(...)`, que se
  limita a guardar la expresión deseada bajo un lock. NO toca tkinter.
- Un tick periódico (`root.after`) lee esa expresión en el hilo principal y
  actualiza la imagen. Así respetamos que tkinter no es thread-safe.

Los frames son PNGs en `faces/<expresion>/NN.png`. Varias imágenes en una
carpeta = animación; una sola = cara fija.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from bmo.domain.models import Expression, Note
from bmo.ports.face import FacePort

_LOG = logging.getLogger(__name__)
_IMG_EXT = {".png", ".gif", ".jpg", ".jpeg"}
_DEFAULT_FACES_DIR = Path(__file__).parent / "faces"
_PREVIEW_LEN = 240


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
        self._notes_provider: Optional[Callable[[], List[Note]]] = None
        self._menu: Optional[object] = None

    @property
    def available(self) -> bool:
        """True si la ventana pudo inicializarse (hay display y frames)."""
        return self._available

    @property
    def current(self) -> Expression:
        """Expresión que se quiere mostrar ahora (thread-safe)."""
        with self._lock:
            return self._current

    def show(self, expression: Expression) -> None:
        with self._lock:
            self._current = expression

    def set_notes_provider(
        self, provider: Optional[Callable[[], List[Note]]]
    ) -> None:
        """Registra la fuente de notas a listar cuando toquen la pantalla."""
        self._notes_provider = provider

    @property
    def menu_open(self) -> bool:
        """True mientras el menú de notas está visible sobre la cara."""
        return self._menu is not None

    def _notes_to_show(self) -> List[Note]:
        """Pide las notas al proveedor. Sin proveedor (o si falla), lista vacía."""
        if self._notes_provider is None:
            return []
        try:
            return list(self._notes_provider())
        except Exception:  # noqa: BLE001 - un fallo leyendo notas no rompe la cara
            _LOG.warning("no se pudieron leer las notas para el menú", exc_info=True)
            return []

    def _handle_touch(self, _event: object = None) -> None:
        """Al tocar la pantalla, abre el menú de notas (o no hace nada sin ventana).

        El evento de tkinter llega en el hilo principal, así que construir los
        widgets del menú aquí es seguro. Si el menú ya está abierto, se ignora
        (se cierra con su propio botón).
        """
        if self._root is None or self._menu is not None:
            return
        self._open_menu()

    def _frames_for(self, expression: Expression) -> List[Path]:
        """Lista los frames de una expresión, ordenados por nombre."""
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
            self._root.bind("<Button-1>", self._handle_touch)
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
        except Exception:  # noqa: BLE001 - sin display, BMO recurre al modo consola
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
                except Exception:  # noqa: BLE001 - un frame roto no interrumpe la cara
                    _LOG.warning("no se pudo cargar el frame %s", path)
                    continue
                scale = min(screen_w / img.width, screen_h / img.height)
                size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                img = img.resize(size, Image.LANCZOS)
                photos.append(ImageTk.PhotoImage(img))
            if photos:
                self._photos[expression] = photos

        if not self._photos:
            _LOG.warning(
                "no se encontró ningún frame en %s; la cara saldrá negra. "
                "Verifica que las carpetas de expresiones (neutral, happy, ...) "
                "estén sincronizadas en la Pi",
                self._faces_dir,
            )

    def _tick(self) -> None:
        """Avanza la animación. Se re-agenda a sí mismo con `after`."""
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

    def _open_menu(self) -> None:
        """Dibuja el menú de notas: un panel a pantalla completa con scroll."""
        import tkinter as tk

        notes = self._notes_to_show()
        menu = tk.Frame(self._root, bg="black")
        menu.place(relx=0, rely=0, relwidth=1, relheight=1)

        header = tk.Frame(menu, bg="black")
        header.pack(fill="x", padx=16, pady=12)
        tk.Label(
            header, text="Notes", bg="black", fg="white",
            font=("DejaVu Sans", 22, "bold"),
        ).pack(side="left")
        tk.Button(
            header, text="✕", command=self._close_menu,
            bg="#222", fg="white", activebackground="#444", activeforeground="white",
            relief="flat", font=("DejaVu Sans", 18, "bold"), width=3, borderwidth=0,
        ).pack(side="right")

        if not notes:
            tk.Label(
                menu, text="No notes yet.", bg="black", fg="#888",
                font=("DejaVu Sans", 16),
            ).pack(expand=True)
        else:
            self._fill_notes(menu, notes)

        self._menu = menu

    def _fill_notes(self, menu: object, notes: List[Note]) -> None:
        """Rellena el panel con las notas en un texto desplazable (arrastre táctil)."""
        import tkinter as tk

        body = tk.Frame(menu, bg="black")
        body.pack(expand=True, fill="both", padx=16, pady=(0, 16))

        scrollbar = tk.Scrollbar(body)
        scrollbar.pack(side="right", fill="y")
        text = tk.Text(
            body, bg="black", fg="white", bd=0, highlightthickness=0,
            wrap="word", font=("DejaVu Sans", 15), padx=8, pady=8,
            yscrollcommand=scrollbar.set, cursor="none", spacing3=6,
        )
        text.pack(side="left", expand=True, fill="both")
        scrollbar.config(command=text.yview)

        text.tag_configure("title", font=("DejaVu Sans", 17, "bold"), foreground="#7fd1ff")
        for note in notes:
            text.insert("end", note.title + "\n", "title")
            text.insert("end", format_note_preview(note) + "\n\n")
        text.config(state="disabled")

        # scroll por arrastre: en la pantalla táctil no hay rueda de ratón
        text.bind("<ButtonPress-1>", lambda e: text.scan_mark(e.x, e.y))
        text.bind("<B1-Motion>", lambda e: text.scan_dragto(e.x, e.y, gain=1))

    def _close_menu(self) -> None:
        """Cierra el menú de notas y vuelve a mostrar la cara."""
        menu = self._menu
        self._menu = None
        if menu is not None:
            menu.destroy()


def format_note_preview(note: Note, max_len: int = _PREVIEW_LEN) -> str:
    """Aplana el cuerpo de la nota a una línea recortada para el listado."""
    body = " ".join(note.content.split())
    if len(body) <= max_len:
        return body
    return body[:max_len].rstrip() + "…"

