"""Tests d'intégration pour POST /predict.

Ces tests utilisent le vrai modèle XGBoost (chargé une fois via le lifespan
dans la fixture client) et un vrai payload à 326 features. C'est le test
"bout en bout" qui valide :
- chargement modèle + validation Pydantic + prédiction + persistance DB
- comportements d'erreur : 422 (validation), 503 (modèle non chargé)
- politique fail-open en cas d'erreur DB
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.main import app
from api.models import Prediction


class TestPredictHappyPath:
    """Cas nominal : payload valide, modèle chargé, DB OK."""

    def test_returns_200(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """POST /predict avec un payload valide répond 200."""
        response = client.post("/predict", json=sample_payload_dict)
        assert response.status_code == 200, response.text

    def test_response_structure(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """La réponse contient request_id, probability, decision, threshold."""
        response = client.post("/predict", json=sample_payload_dict)
        body = response.json()
        assert "request_id" in body
        assert "probability" in body
        assert "decision" in body
        assert "threshold" in body

    def test_response_types(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """Les types des champs sont conformes au contrat."""
        response = client.post("/predict", json=sample_payload_dict)
        body = response.json()
        assert isinstance(body["probability"], float)
        assert 0.0 <= body["probability"] <= 1.0
        assert body["decision"] in (0, 1)
        assert isinstance(body["threshold"], float)

    def test_request_id_is_valid_uuid(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """request_id est un UUID4 valide (parseable par uuid.UUID)."""
        response = client.post("/predict", json=sample_payload_dict)
        request_id = response.json()["request_id"]
        # uuid.UUID lève ValueError si le format est invalide
        parsed = uuid.UUID(request_id)
        assert str(parsed) == request_id


class TestPredictPersistence:
    """Vérifie que la prédiction est persistée en DB de test."""

    def test_prediction_is_persisted(
        self,
        client: TestClient,
        db_session: Session,
        sample_payload_dict: dict,
    ) -> None:
        """Après POST /predict 200, la ligne existe en DB avec le bon request_id."""
        response = client.post("/predict", json=sample_payload_dict)
        request_id = response.json()["request_id"]

        # Lire en DB de test (la fixture client override get_db vers la même)
        row = (
            db_session.query(Prediction)
            .filter_by(request_id=request_id)
            .one_or_none()
        )
        assert row is not None, "La prédiction n'a pas été persistée"
        assert row.prediction_proba == pytest.approx(
            response.json()["probability"]
        )
        assert row.prediction == response.json()["decision"]


class TestPredictValidation:
    """Erreurs de validation Pydantic (HTTP 422)."""

    def test_missing_field_returns_422(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """Un payload privé d'un champ requis répond 422."""
        bad_payload = dict(sample_payload_dict)
        # On retire le premier champ
        first_key = next(iter(bad_payload.keys()))
        del bad_payload[first_key]

        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_wrong_type_returns_422(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """Un payload avec un type incorrect (string au lieu de float) répond 422."""
        bad_payload = dict(sample_payload_dict)
        # On force une valeur string sur un champ numérique
        # AMT_INCOME_TOTAL est un float dans le schéma
        bad_payload["AMT_INCOME_TOTAL"] = "not_a_number"

        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_validation_error_has_detail(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """La réponse 422 contient un champ 'detail' explicatif."""
        bad_payload = dict(sample_payload_dict)
        first_key = next(iter(bad_payload.keys()))
        del bad_payload[first_key]

        response = client.post("/predict", json=bad_payload)
        body = response.json()
        assert "detail" in body


class TestPredictModelNotLoaded:
    """Erreur 503 quand le predictor n'est pas chargé."""

    def test_returns_503_when_predictor_is_none(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """Si app.state.predictor=None, POST /predict répond 503."""
        original = app.state.predictor
        app.state.predictor = None
        try:
            response = client.post("/predict", json=sample_payload_dict)
            assert response.status_code == 503
        finally:
            app.state.predictor = original


class TestPredictFailOpen:
    """Politique fail-open : erreur DB ne casse pas la réponse client."""

    def test_db_error_returns_200_anyway(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """Si log_prediction lève SQLAlchemyError, l'API répond 200 quand même."""
        # On patche la fonction log_prediction telle qu'importée dans api.main
        with patch(
            "api.main.log_prediction",
            side_effect=SQLAlchemyError("DB down"),
        ):
            response = client.post("/predict", json=sample_payload_dict)

        assert response.status_code == 200, (
            f"Le fail-open n'est pas respecté : {response.status_code} {response.text}"
        )
        # La réponse contient quand même la prédiction
        body = response.json()
        assert "probability" in body
        assert "decision" in body

class TestPredictInternalError:
    """Erreur 500 quand une exception générique non gérée est levée."""

    def test_unexpected_exception_returns_500(
        self, client: TestClient, sample_payload_dict: dict
    ) -> None:
        """Si predictor.predict lève une exception générique, l'API répond 500.

        Note : on construit un TestClient dédié avec raise_server_exceptions=False
        pour que les exceptions soient gérées par les handlers FastAPI au lieu
        de remonter dans le test (comportement par défaut de TestClient).
        La fixture `client` est demandée pour s'assurer que le lifespan a déjà
        chargé app.state.predictor.
        """
        def broken_predict(_):
            raise RuntimeError("Unexpected failure")

        original_predict = app.state.predictor.predict
        app.state.predictor.predict = broken_predict
        try:
            test_client = TestClient(app, raise_server_exceptions=False)
            response = test_client.post("/predict", json=sample_payload_dict)
            assert response.status_code == 500
            body = response.json()
            assert "detail" in body
        finally:
            app.state.predictor.predict = original_predict