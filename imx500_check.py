"""Visor web de debug del IMX500: muestra en vivo lo que ve la AI Camera.

Corre en la Raspberry Pi (headless) y sirve un stream MJPEG con las cajas de
deteccion dibujadas encima y su confianza. Sirve para distinguir dos cosas:

  * si el sensor NO detecta nada, o
  * si detecta pero por debajo del umbral de produccion (0.55).

Por eso usa un umbral BAJO (DEBUG_THRESHOLD) y dibuja el score de cada caja.
Reusa el adapter real de BMO (mismo parseo de tensores), pero accede a sus
internals para poder mostrar los scores (que Perception ya descarta).

    python imx500_check.py

Despues abri en el navegador de otra maquina:  http://bmo.local:8080
"""

from __future__ import annotations

import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from bmo.adapters.camera.ai_camera_imx500 import AiCameraImx500, prepare_boxes
from bmo.config import Config

DEBUG_THRESHOLD = 0.20
PORT = 8080

# Ultimas detecciones dibujables (label, score, x, y, w, h). Se reusan en los
# frames que vienen SIN tensor de inferencia (get_outputs -> None), igual que el
# `last_detections` del demo oficial, para que el video no parpadee.
_LAST_DETS: list[tuple[str, float, int, int, int, int]] = []

_PAGE = (
    b"<!doctype html><html><head><meta charset='utf-8'>"
    b"<title>BMO ve</title></head>"
    b"<body style='margin:0;background:#111;text-align:center'>"
    b"<img src='/stream' style='max-width:100%;height:auto'>"
    b"</body></html>"
)


def render_frame(cam: AiCameraImx500) -> bytes:
    """Captura un frame, dibuja las detecciones y devuelve un JPEG en bytes."""
    global _LAST_DETS
    picam2 = cam._picam2
    imx500 = cam._imx500
    frame = picam2.capture_array()
    metadata = picam2.capture_metadata()

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        # Frame sin tensor: reusamos las ultimas detecciones (no parpadea).
        status = f"sin tensor (cache: {len(_LAST_DETS)})"
    else:
        boxes = np_outputs[0][0]
        scores = np_outputs[1][0]
        classes = np_outputs[2][0]

        _, input_h = imx500.get_input_size()
        bbox_normalization = getattr(cam._intrinsics, "bbox_normalization", False)
        bbox_order = getattr(cam._intrinsics, "bbox_order", "yx")
        boxes = prepare_boxes(boxes, bbox_normalization, bbox_order, input_h)

        max_score = float(max(scores)) if len(scores) else 0.0
        dets: list[tuple[str, float, int, int, int, int]] = []
        for coords, score, cls in zip(boxes, scores, classes):
            if float(score) <= DEBUG_THRESHOLD:
                continue
            x, y, w, h = imx500.convert_inference_coords(coords, metadata, picam2)
            idx = int(cls)
            label = cam._labels[idx] if 0 <= idx < len(cam._labels) else str(idx)
            dets.append((label, float(score), int(x), int(y), int(w), int(h)))
        _LAST_DETS = dets
        status = f"raw: {len(scores)}  max: {max_score:.2f}  dibujadas: {len(dets)}"

    for label, score, x, y, w, h in _LAST_DETS:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        frame,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    ok, jpeg = cv2.imencode(".jpg", frame)
    return jpeg.tobytes()


def make_handler(cam: AiCameraImx500) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:  # silenciar el log ruidoso
            pass

        def do_GET(self) -> None:  # noqa: N802 - lo pide la stdlib
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_PAGE)
                return

            if self.path == "/stream":
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()
                try:
                    while True:
                        jpeg = render_frame(cam)
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # el navegador cerro la pestania
                return

            self.send_error(404)

    return Handler


def main() -> None:
    config = Config.load("config.yaml")
    cam = AiCameraImx500.from_config(config.camera, config.vision)

    print("Arrancando la AI Camera IMX500... (la primera vez sube el firmware)")
    cam.start()

    if cam.capture() is None:
        print("ERROR: no se pudo iniciar la AI Camera. Mira el warning de arriba.")
        return

    print(f"Labels cargados: {len(cam._labels)}")
    print(f"Umbral de debug: {DEBUG_THRESHOLD} (produccion usa {cam._threshold})")
    print(f"\nAbri en el navegador:  http://bmo.local:{PORT}")
    print("(o usa la IP de la Pi si bmo.local no resuelve). Ctrl+C para salir.\n")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), make_handler(cam))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nchau")
    finally:
        server.shutdown()
        cam.stop()


if __name__ == "__main__":
    main()
