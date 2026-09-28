from fastapi.testclient import TestClient

from app.components.c3_misinformation.dev_app import app


def test_ping():
    r = TestClient(app).get("/api/v1/c3/ping")
    assert r.status_code == 200
