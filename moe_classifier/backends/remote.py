"""
Remote backend — HTTP client to a running MoLE coordinator / monolithic service.

Instead of loading any ML models, this backend sends classification requests
to an already-deployed MoLE API over HTTP.  Ideal for application developers
who consume MoLE as a service.
"""

import time
from typing import Any, Dict, Optional

import httpx

from moe_classifier.types import ClassificationResult
from .base import ClassifierBackend


class RemoteBackend(ClassifierBackend):
    """
    HTTP client backend that connects to a running MoLE service.

    Parameters
    ----------
    coordinator_url : str
        Base URL of the MoLE service (e.g. ``"http://10.8.100.21:8000"``).
    credentials : dict, optional
        ``{"username": "...", "password": "..."}`` — the backend will
        call ``/api/v1/auth/token`` during ``initialize()`` to obtain a
        JWT automatically.
    token : str, optional
        A pre-existing JWT access token.  If provided, ``credentials``
        is ignored and no login call is made.
    timeout : float, optional
        HTTP request timeout in seconds (default: 300).
    """

    def __init__(
        self,
        coordinator_url: str,
        credentials: Optional[Dict[str, str]] = None,
        token: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        # Strip trailing slash for consistency
        self._base_url = coordinator_url.rstrip("/")
        self._credentials = credentials
        self._token = token
        self._timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=self._timeout,
                write=30.0,
                pool=5.0,
            ),
        )

        # Authenticate if needed
        if self._token:
            # User already has a token — use it directly
            self._client.headers["Authorization"] = f"Bearer {self._token}"
        elif self._credentials:
            # Auto-login to get a JWT
            self._token = self._login(
                self._credentials["username"],
                self._credentials["password"],
            )
            self._client.headers["Authorization"] = f"Bearer {self._token}"
        # else: unauthenticated (will 401 on protected endpoints)

        # Verify connectivity
        self._health_check()
        self._initialized = True

    def _login(self, username: str, password: str) -> str:
        """Call the /api/v1/auth/token endpoint and return the JWT."""
        response = self._client.post(
            "/api/v1/auth/token",
            data={"username": username, "password": password},
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Authentication failed (HTTP {response.status_code}): "
                f"{response.text}"
            )
        data = response.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(
                "Authentication response did not contain 'access_token'."
            )
        return token

    def _health_check(self) -> None:
        """Verify the remote service is reachable and ready."""
        try:
            response = self._client.get("/api/v1/health/ready")
            if response.status_code != 200:
                print(
                    f"[RemoteBackend] Warning: health check returned "
                    f"HTTP {response.status_code}"
                )
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot reach MoLE service at {self._base_url}: {exc}"
            ) from exc

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._client is not None

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        description: str = "",
        *,
        return_domain_probabilities: bool = False,
        return_raw_response: bool = False,
    ) -> ClassificationResult:
        t0 = time.perf_counter()

        payload = {
            "text": text,
            "description": description,
            "options": {
                "return_probabilities": return_domain_probabilities,
                "return_raw_response": return_raw_response,
            },
        }

        response = self._client.post("/api/v1/classify", json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"Classification failed (HTTP {response.status_code}): "
                f"{response.text}"
            )

        data = response.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return ClassificationResult(
            language=data.get("language", "unknown"),
            domain=data.get("domain", "unknown"),
            task=data.get("task", "unknown"),
            result=str(data.get("result", "")),
            routing_path=data.get("routing_path", ""),
            confidence=data.get("confidence"),
            domain_probabilities=data.get("domain_probabilities"),
            raw_response=data.get("raw_response"),
            processing_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        response = self._client.get("/api/v1/classify/stats")
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        return response.json()
