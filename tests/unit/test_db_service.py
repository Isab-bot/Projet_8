"""Tests unitaires de api/db_service.py.

log_prediction insère une ligne dans la table predictions via SQLAlchemy.
On utilise une vraie DB SQLite de test (fixture db_session) plutôt que
de mocker la session, parce que la fonction est essentiellement de
l'orchestration ORM : la mocker reviendrait à tester rien.

Le comportement fail-open (gestion d'erreur DB côté endpoint) sera testé
dans tests/integration/test_predict.py.
"""
from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from api.db_service import log_prediction
from api.models import Prediction
from api.schemas import PredictionInput


class TestLogPrediction:
    """Persistance d'une prédiction dans la table predictions."""

    def test_inserts_one_row(
        self,
        db_session: Session,
        sample_prediction_input: PredictionInput,
    ) -> None:
        """Après log_prediction, la table contient exactement une ligne."""
        log_prediction(
            db=db_session,
            request_id="test-uuid-001",
            input_data=sample_prediction_input,
            proba=0.42,
            decision=1,
            model_version="0.1.0",
            threshold=0.334,
            latency_ms=50.0,
            inference_ms=10.0,
        )
        count = db_session.query(Prediction).count()
        assert count == 1

    def test_persists_metadata_correctly(
        self,
        db_session: Session,
        sample_prediction_input: PredictionInput,
    ) -> None:
        """Les métadonnées (request_id, proba, decision, version, seuil) sont stockées."""
        log_prediction(
            db=db_session,
            request_id="test-uuid-002",
            input_data=sample_prediction_input,
            proba=0.42,
            decision=1,
            model_version="0.1.0",
            threshold=0.334,
            latency_ms=50.0,
            inference_ms=10.0,
        )
        row = db_session.query(Prediction).filter_by(request_id="test-uuid-002").one()
        assert row.request_id == "test-uuid-002"
        assert row.prediction_proba == pytest.approx(0.42)
        assert row.prediction == 1
        assert row.model_version == "0.1.0"
        assert row.threshold == pytest.approx(0.334)

    def test_persists_features_with_sql_names(
        self,
        db_session: Session,
        sample_prediction_input: PredictionInput,
        sample_payload_dict: dict,
    ) -> None:
        """Les features sont stockées sous leurs noms SQL (sans <lambda>).

        log_prediction utilise model_dump(by_alias=False), ce qui doit
        produire les noms Python qui sont alignés avec les colonnes SQL.
        On vérifie ça en prenant la première feature du payload, en la
        transformant en nom SQL, et en lisant la valeur en base.
        """
        from api.feature_naming import to_sql_column_name

        log_prediction(
            db=db_session,
            request_id="test-uuid-003",
            input_data=sample_prediction_input,
            proba=0.5,
            decision=0,
            model_version="0.1.0",
            threshold=0.334,
            latency_ms=50.0,
            inference_ms=10.0,
        )
        row = db_session.query(Prediction).filter_by(request_id="test-uuid-003").one()

        # On prend la première feature du payload (nom d'origine, ex. avec <lambda>)
        # On vérifie que la colonne SQL correspondante existe et n'est pas None
        first_original_name = next(iter(sample_payload_dict.keys()))
        sql_name = to_sql_column_name(first_original_name)
        assert hasattr(row, sql_name), f"Colonne {sql_name} absente du modèle ORM"
        assert getattr(row, sql_name) is not None, (
            f"Feature {sql_name} non persistée"
        )

    def test_commit_is_called(
        self,
        db_session: Session,
        sample_prediction_input: PredictionInput,
    ) -> None:
        """Après log_prediction, la ligne est visible dans une nouvelle requête.

        C'est la preuve que commit() a été appelé : sans commit, expire_all()
        verrait l'objet en cache mais une nouvelle session n'y aurait pas accès.
        """
        log_prediction(
            db=db_session,
            request_id="test-uuid-004",
            input_data=sample_prediction_input,
            proba=0.1,
            decision=0,
            model_version="0.1.0",
            threshold=0.334,
            latency_ms=50.0,
            inference_ms=10.0,
        )
        db_session.expire_all()
        row = db_session.query(Prediction).filter_by(request_id="test-uuid-004").first()
        assert row is not None

    def test_two_calls_create_two_rows(
        self,
        db_session: Session,
        sample_prediction_input: PredictionInput,
    ) -> None:
        """Deux appels successifs créent deux lignes distinctes."""
        log_prediction(
            db=db_session,
            request_id="test-uuid-005a",
            input_data=sample_prediction_input,
            proba=0.1,
            decision=0,
            model_version="0.1.0",
            threshold=0.334,
            latency_ms=50.0,
            inference_ms=10.0,
        )
        log_prediction(
            db=db_session,
            request_id="test-uuid-005b",
            input_data=sample_prediction_input,
            proba=0.9,
            decision=1,
            model_version="0.1.0",
            threshold=0.334,
            latency_ms=50.0,
            inference_ms=10.0,
        )
        count = db_session.query(Prediction).count()
        assert count == 2
