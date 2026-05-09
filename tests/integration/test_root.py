"""Tests d'intégration pour GET /."""
from __future__ import annotations

from fastapi.testclient import TestClient


class TestRootEndpoint:
    """Page d'accueil minimaliste de l'API."""

    def test_returns_200(self, client: TestClient) -> None:
        """GET / répond 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_service_metadata(self, client: TestClient) -> None:
        """GET / retourne les métadonnées du service."""
        response = client.get("/")
        body = response.json()
        assert "service" in body
        assert "version" in body
        assert body["docs"] == "/docs"
        assert body["health"] == "/health"