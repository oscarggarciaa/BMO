"""Tests del adapter web de la cara (Flask): estado, SSE trigger y sprites."""

from __future__ import annotations

import pytest

from bmo.domain.models import Expression
from bmo.interfaces.web.face_web import FaceWebServer


@pytest.fixture()
def server_and_client():
    server = FaceWebServer(host="127.0.0.1", port=5000)
    server.app.testing = True
    return server, server.app.test_client()


def test_initial_state_is_neutral(server_and_client) -> None:
    _, client = server_and_client
    resp = client.get("/state")
    assert resp.get_json() == {"expression": "neutral"}


def test_show_updates_current_state(server_and_client) -> None:
    server, client = server_and_client
    server.show(Expression.HAPPY)
    assert client.get("/state").get_json() == {"expression": "happy"}


def test_set_endpoint_changes_expression(server_and_client) -> None:
    _, client = server_and_client
    resp = client.post("/set/talking")
    assert resp.status_code == 200
    assert client.get("/state").get_json() == {"expression": "talking"}


def test_set_endpoint_rejects_invalid_expression(server_and_client) -> None:
    _, client = server_and_client
    resp = client.post("/set/banana")
    assert resp.status_code == 400


def test_index_renders_the_face_page(server_and_client) -> None:
    _, client = server_and_client
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"face" in resp.data.lower()


def test_faces_manifest_lists_frames(server_and_client) -> None:
    _, client = server_and_client
    resp = client.get("/faces/neutral")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["frames"]  # al menos un frame
    assert all(f.startswith("/faces/neutral/") for f in data["frames"])
    assert data["fps"] > 0


def test_faces_manifest_animated_expression_has_multiple_frames(server_and_client) -> None:
    _, client = server_and_client
    # talking viene de 'speaking' (3 frames) -> es una animacion.
    data = client.get("/faces/talking").get_json()
    assert len(data["frames"]) >= 2


def test_faces_manifest_rejects_unknown_expression(server_and_client) -> None:
    _, client = server_and_client
    resp = client.get("/faces/banana")
    assert resp.status_code == 404


def test_frame_is_served_as_image(server_and_client) -> None:
    _, client = server_and_client
    frames = client.get("/faces/neutral").get_json()["frames"]
    resp = client.get(frames[0])
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"


def test_frame_rejects_unknown_expression(server_and_client) -> None:
    _, client = server_and_client
    resp = client.get("/faces/banana/01.png")
    assert resp.status_code == 404


def test_frame_rejects_bad_filename(server_and_client) -> None:
    _, client = server_and_client
    # Nombre fuera del patron numerico -> 404 (bloquea path traversal).
    resp = client.get("/faces/neutral/secretos.txt")
    assert resp.status_code == 404
