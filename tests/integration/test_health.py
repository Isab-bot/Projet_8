"""Tests d'intégration pour GET /health."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


class TestHealthEndpoint:
    """Healthcheck : statut de l'API + état du modèle."""

    def test_returns_200(self, client: TestClient) -> None:
        """GET /health répond 200 même si dégradé."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_ok_when_model_loaded(self, client: TestClient) -> None:
        """Quand le predictor est chargé, status='ok' et model_loaded=True."""
        # La fixture client charge le vrai Predictor via le lifespan
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_status_degraded_when_model_not_loaded(
        self, client: TestClient
    ) -> None:
        """Quand le predictor est None, status='degraded' et model_loaded=False."""
        # On force temporairement l'état "modèle non chargé"
        original = app.state.predictor
        app.state.predictor = None
        try:
            response = client.get("/health")
            body = response.json()
            assert body["status"] == "degraded"
            assert body["model_loaded"] is False
        finally:
            app.state.predictor = original

    def test_returns_api_version(self, client: TestClient) -> None:
        """GET /health expose la version de l'API."""
        from api.config import API_VERSION

        response = client.get("/health")
        body = response.json()
        assert body["api_version"] == API_VERSION
