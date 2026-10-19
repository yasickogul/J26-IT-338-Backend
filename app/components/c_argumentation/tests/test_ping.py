from fastapi.testclient import TestClient

from app.components.c_argumentation.dev_app import app


def test_ping():
    r = TestClient(app).get("/api/v1/c_argumentation/ping")
    assert r.status_code == 200
