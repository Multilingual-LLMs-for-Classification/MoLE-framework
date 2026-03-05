"""
Unit tests for authentication endpoints.

These tests run without ML dependencies — only SQLAlchemy, passlib, and
python-jose are required. The routing system will fail to initialize (no
gating models in CI), but auth endpoints are independent of it.
"""


def test_register_user(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "testuser", "password": "testpass123"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "testuser"
    assert data["disabled"] is False


def test_register_duplicate_user(client):
    client.post(
        "/api/v1/auth/register",
        json={"username": "dupuser", "password": "testpass123"},
    )
    resp = client.post(
        "/api/v1/auth/register",
        json={"username": "dupuser", "password": "testpass123"},
    )
    assert resp.status_code == 400


def test_login_returns_token(client):
    client.post(
        "/api/v1/auth/register",
        json={"username": "loginuser", "password": "testpass123"},
    )
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "loginuser", "password": "testpass123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(client):
    client.post(
        "/api/v1/auth/register",
        json={"username": "badpassuser", "password": "testpass123"},
    )
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "badpassuser", "password": "wrongpassword"},
    )
    assert resp.status_code == 401


def test_login_unknown_user(client):
    resp = client.post(
        "/api/v1/auth/token",
        data={"username": "nobody", "password": "doesnotmatter"},
    )
    assert resp.status_code == 401


def test_classify_requires_auth(client):
    resp = client.post(
        "/api/v1/classify",
        json={"text": "hello", "description": "test"},
    )
    assert resp.status_code == 401
