from fastapi.testclient import TestClient

from app.components.c1_document_understanding.dev_app import app


def test_ping():
    r = TestClient(app).get("/api/v1/c1/ping")
    assert r.status_code == 200
