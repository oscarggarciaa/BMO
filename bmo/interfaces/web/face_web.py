"""Adapter web de la cara de BMO (Flask + SSE): implementa FacePort."""

from __future__ import annotations

import logging
import queue
import re
import threading
from pathlib import Path
from typing import List

from flask import Flask, Response, abort, jsonify, render_template, send_file

from bmo.domain.models import Expression
from bmo.ports.face import FacePort

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 4
_FRAME_RE = re.compile(r"^\d+\.(png|gif|webp|jpg|jpeg)$", re.IGNORECASE)


class FaceWebServer(FacePort):
    """Muestra la cara de BMO en un navegador y la actualiza en vivo (SSE)."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self._host = host
        self._port = port
        self._current = Expression.NEUTRAL
        self._lock = threading.Lock()
        self._subscribers: List["queue.Queue[str]"] = []
        self._faces_dir = Path(__file__).parent / "static" / "faces"
        self.app = self._build_app()

    def show(self, expression: Expression) -> None:
        """Actualiza la expresion actual y la empuja a todos los navegadores."""
        with self._lock:
            self._current = expression
            subscribers = list(self._subscribers)
        for q in subscribers:
            q.put(expression.value)

    def start(self) -> None:
        """Arranca Flask en un hilo daemon (no bloquea el REPL)."""
        thread = threading.Thread(target=self._run, name="bmo-face-web", daemon=True)
        thread.start()
        logger.info("cara de BMO en http://%s:%d", self._host, self._port)

    def _run(self) -> None:
        self.app.run(
            host=self._host,
            port=self._port,
            threaded=True,
            use_reloader=False,
            debug=False,
        )

    def _build_app(self) -> Flask:
        app = Flask(__name__)

        @app.route("/")
        def index() -> str:
            return render_template("face.html")

        @app.route("/state")
        def state():
            with self._lock:
                return jsonify({"expression": self._current.value})

        @app.route("/events")
        def events() -> Response:
            return Response(self._event_stream(), mimetype="text/event-stream")

        @app.route("/set/<expression>", methods=["GET", "POST"])
        def set_expression(expression: str):
            try:
                expr = Expression(expression)
            except ValueError:
                return jsonify({"error": f"expresion invalida: {expression}"}), 400
            self.show(expr)
            return jsonify({"ok": True, "expression": expr.value})

        @app.route("/faces/<expression>")
        def face_manifest(expression: str):
            try:
                Expression(expression)
            except ValueError:
                abort(404)
            expr_dir = self._faces_dir / expression
            frames: List[str] = []
            if expr_dir.is_dir():
                for frame in sorted(expr_dir.iterdir()):
                    if frame.is_file() and _FRAME_RE.match(frame.name):
                        frames.append(f"/faces/{expression}/{frame.name}")
            return jsonify({"frames": frames, "fps": _DEFAULT_FPS})

        @app.route("/faces/<expression>/<filename>")
        def face_frame(expression: str, filename: str):
            try:
                Expression(expression)
            except ValueError:
                abort(404)
            if not _FRAME_RE.match(filename):
                abort(404)
            candidate = self._faces_dir / expression / filename
            if candidate.is_file():
                return send_file(str(candidate))
            abort(404)

        return app

    def _event_stream(self):
        """Generador SSE: manda la expresion actual y luego cada cambio."""
        q: "queue.Queue[str]" = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
            current = self._current.value
        yield f"data: {current}\n\n"
        try:
            while True:
                expr = q.get()
                yield f"data: {expr}\n\n"
        finally:
            with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)
