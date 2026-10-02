from fastapi.testclient import TestClient

from selene.api.app import create_app


def test_health():
    client = TestClient(create_app())
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ephemeris"] == "de440s.bsp"
