"""Tests d'intégration du backend ONNX du Predictor.

Ces tests utilisent les VRAIS artefacts ONNX (models/preprocessor.pkl +
models/xgboost_classifier.onnx) et le vrai pipeline joblib, pour vérifier :

1. Le backend ONNX se charge et prédit sans erreur.
2. L'équivalence numérique entre les backends joblib et onnx sur le même
   input (tolérance 1e-5 sur la proba, décision identique). C'est la
   garantie automatisée que la conversion ONNX (branche feature/onnx-conversion)
   reste valide à chaque exécution du CI.

Contrairement à test_predict.py (qui teste l'API via le lifespan en backend
par défaut), ces tests instancient directement deux Predictor avec des
backends différents, sans passer par FastAPI.
"""
from __future__ import annotations

import pytest

from api.predictor import ModelNotLoadedError, Predictor
from api.schemas import PredictionInput

# Tolérance alignée sur Q5 du design-doc étape 9.
PROBA_TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def predictor_joblib() -> Predictor:
    """Predictor backend joblib (pipeline complet)."""
    return Predictor(backend="joblib")


@pytest.fixture(scope="module")
def predictor_onnx() -> Predictor:
    """Predictor backend ONNX (preprocessor sklearn + ONNX Runtime)."""
    return Predictor(backend="onnx")


class TestOnnxBackendLoads:
    """Chargement du backend ONNX."""

    def test_onnx_predictor_loads(self, predictor_onnx: Predictor) -> None:
        """Le backend ONNX charge le preprocessor et la session ONNX."""
        assert predictor_onnx.backend == "onnx"
        assert predictor_onnx.preprocessor is not None
        assert predictor_onnx.onnx_session is not None
        # Le pipeline joblib n'est PAS chargé en mode onnx
        assert predictor_onnx.pipeline is None

    def test_onnx_raises_when_session_none(
        self, predictor_onnx: Predictor, sample_prediction_input: PredictionInput
    ) -> None:
        """Si la session ONNX est None, predict lève ModelNotLoadedError."""
        # On sauvegarde puis on neutralise la session
        original = predictor_onnx.onnx_session
        predictor_onnx.onnx_session = None
        try:
            with pytest.raises(ModelNotLoadedError):
                predictor_onnx.predict(sample_prediction_input)
        finally:
            predictor_onnx.onnx_session = original


class TestOnnxBackendPredict:
    """Prédiction via le backend ONNX et équivalence avec joblib."""

    def test_onnx_predict_returns_valid_output(
        self, predictor_onnx: Predictor, sample_prediction_input: PredictionInput
    ) -> None:
        """Le backend ONNX retourne un PredictionOutput valide."""
        result = predictor_onnx.predict(sample_prediction_input)
        assert 0.0 <= result.probability <= 1.0
        assert result.decision in (0, 1)
        assert result.threshold == predictor_onnx.threshold

    def test_onnx_matches_joblib_probability(
        self,
        predictor_joblib: Predictor,
        predictor_onnx: Predictor,
        sample_prediction_input: PredictionInput,
    ) -> None:
        """La proba ONNX est égale à la proba joblib à 1e-5 près (Q5)."""
        proba_joblib = predictor_joblib.predict(sample_prediction_input).probability
        proba_onnx = predictor_onnx.predict(sample_prediction_input).probability
        assert abs(proba_joblib - proba_onnx) <= PROBA_TOLERANCE, (
            f"Écart proba trop grand : joblib={proba_joblib}, onnx={proba_onnx}, "
            f"diff={abs(proba_joblib - proba_onnx):.2e} > {PROBA_TOLERANCE:.0e}"
        )

    def test_onnx_matches_joblib_decision(
        self,
        predictor_joblib: Predictor,
        predictor_onnx: Predictor,
        sample_prediction_input: PredictionInput,
    ) -> None:
        """La décision ONNX est identique à la décision joblib (Q5 : 100%)."""
        decision_joblib = predictor_joblib.predict(sample_prediction_input).decision
        decision_onnx = predictor_onnx.predict(sample_prediction_input).decision
        assert decision_joblib == decision_onnx
