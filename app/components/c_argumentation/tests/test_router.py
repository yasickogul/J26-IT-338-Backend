from fastapi.testclient import TestClient

from app.components.c_argumentation.dev_app import app


def test_router_ping():
    assert TestClient(app).get("/api/v1/c_argumentation/ping").status_code == 200
