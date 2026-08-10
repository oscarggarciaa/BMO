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

# Colores tomados de la cara real de BMO (muestreados de faces/neutral/01.png).
_BMO_BG = "#bdffcb"       # verde menta del fondo de la cara
_BMO_DARK = "#2f5a41"     # verde oscuro de ojos/boca: texto y acentos
_CARD_BG = "#ffffff"      # tarjeta de cada nota
_CARD_LINE = "#8fe0a3"    # borde suave de la tarjeta
_DANGER = "#c0533f"       # rojo apagado para el borrar


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
        self._notes_deleter: Optional[Callable[[Note], None]] = None
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

    def set_notes_deleter(
        self, deleter: Optional[Callable[[Note], None]]
    ) -> None:
        """Registra la acción de borrar una nota desde el menú táctil."""
        self._notes_deleter = deleter

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
        """Dibuja el menú de notas: panel a pantalla completa con los colores de BMO."""
        import tkinter as tk

        menu = tk.Frame(self._root, bg=_BMO_BG)
        menu.place(relx=0, rely=0, relwidth=1, relheight=1)

        header = tk.Frame(menu, bg=_BMO_BG)
        header.pack(fill="x", padx=22, pady=(18, 6))
        tk.Button(
            header, text="✕", command=self._close_menu,
            bg=_BMO_BG, fg=_BMO_DARK, activebackground=_BMO_BG,
            activeforeground=_BMO_DARK, relief="flat", borderwidth=0,
            font=("DejaVu Sans", 22, "bold"), cursor="none",
        ).pack(side="right")

        self._render_notes(menu)
        self._menu = menu

    def _render_notes(self, menu: object) -> None:
        """Rellena el panel con una tarjeta por nota (o un aviso si no hay)."""
        import tkinter as tk

        notes = self._notes_to_show()
        if not notes:
            tk.Label(
                menu, text="No notes yet", bg=_BMO_BG, fg=_BMO_DARK,
                font=("DejaVu Sans", 18),
            ).pack(expand=True)
            return

        # canvas + frame interior = lista desplazable de tarjetas
        canvas = tk.Canvas(menu, bg=_BMO_BG, highlightthickness=0, bd=0)
        canvas.pack(side="left", expand=True, fill="both", padx=(22, 22), pady=(0, 22))
        inner = tk.Frame(canvas, bg=_BMO_BG)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfigure(window, width=e.width)
        )
        # scroll por arrastre (la pantalla táctil no tiene rueda)
        for widget in (canvas, inner):
            widget.bind("<ButtonPress-1>", lambda e: canvas.scan_mark(e.x, e.y))
            widget.bind("<B1-Motion>", lambda e: canvas.scan_dragto(e.x, e.y, gain=1))

        for note in notes:
            self._render_card(inner, note)

    def _render_card(self, parent: object, note: Note) -> None:
        """Una tarjeta: solo la frase de la nota y un botón para borrarla."""
        import tkinter as tk

        card = tk.Frame(
            parent, bg=_CARD_BG, highlightbackground=_CARD_LINE,
            highlightthickness=1, bd=0,
        )
        card.pack(fill="x", pady=7)

        delete = tk.Button(
            card, text="✕", command=lambda n=note: self._delete_note(n),
            bg=_CARD_BG, fg=_DANGER, activebackground=_CARD_BG,
            activeforeground=_DANGER, relief="flat", borderwidth=0,
            font=("DejaVu Sans", 15, "bold"), cursor="none",
        )
        delete.pack(side="right", anchor="n", padx=(0, 10), pady=8)

        tk.Label(
            card, text=format_note_preview(note), bg=_CARD_BG, fg=_BMO_DARK,
            font=("DejaVu Sans", 16), justify="left", anchor="w",
            wraplength=self._root.winfo_screenwidth() - 140,
        ).pack(side="left", fill="x", expand=True, padx=(16, 6), pady=12)

    def _delete_note(self, note: Note) -> None:
        """Borra una nota y refresca el menú para reflejarlo al instante."""
        deleter = self._notes_deleter
        if deleter is not None:
            try:
                deleter(note)
            except Exception:  # noqa: BLE001 - un fallo al borrar no rompe la cara
                _LOG.warning("no se pudo borrar la nota", exc_info=True)
        if self._root is not None and self._menu is not None:
            self._refresh_menu()

    def _refresh_menu(self) -> None:
        """Reconstruye el menú (tras borrar) sin cerrarlo del todo."""
        if self._menu is None:
            return
        self._close_menu()
        self._open_menu()

    def _close_menu(self) -> None:
        """Cierra el menú de notas y vuelve a mostrar la cara."""
        menu = self._menu
        self._menu = None
        if menu is not None:
            menu.destroy()


def format_note_preview(note: Note, max_len: int = _PREVIEW_LEN) -> str:
    """Aplana el contenido de la nota a una frase recortada para la tarjeta."""
    body = " ".join(note.content.split())
    if len(body) <= max_len:
        return body
    return body[:max_len].rstrip() + "…"

