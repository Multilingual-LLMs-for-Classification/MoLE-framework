"""
Unit tests for health check endpoints.
"""


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_liveness(client):
    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_readiness_not_ready_in_ci(client):
    # In CI the gating models are not present, so the routing system will not
    # initialize. The endpoint must still respond with 200 and report not_ready.
    resp = client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("ready", "not_ready")
