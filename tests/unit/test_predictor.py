"""Tests unitaires de api/predictor.py.

Le pipeline scikit-learn est mocké pour isoler la logique de Predictor
(application du seuil, format de sortie, gestion d'erreurs). Le test
bout en bout avec le vrai modèle est dans tests/integration/test_predict.py.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.predictor import ModelNotLoadedError, Predictor


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Pipeline scikit-learn mocké, predict_proba retourne [[0.3, 0.7]]."""
    pipeline = MagicMock()
    pipeline.predict_proba.return_value = np.array([[0.3, 0.7]])
    return pipeline


@pytest.fixture
def predictor(mock_pipeline: MagicMock) -> Predictor:
    """Predictor instancié avec un pipeline mocké (pas de chargement disque)."""
    with patch.object(Predictor, "_load_pipeline", return_value=mock_pipeline):
        return Predictor()


class TestPredictorInit:
    """Construction du Predictor et chargement du pipeline."""

    def test_init_loads_pipeline(self, mock_pipeline: MagicMock) -> None:
        """À l'instanciation, _load_pipeline est appelé et stocké."""
        with patch.object(Predictor, "_load_pipeline", return_value=mock_pipeline):
            p = Predictor()
        assert p.pipeline is mock_pipeline

    def test_init_uses_default_threshold(self, mock_pipeline: MagicMock) -> None:
        """Sans argument, le seuil par défaut DECISION_THRESHOLD est utilisé."""
        from api.config import DECISION_THRESHOLD

        with patch.object(Predictor, "_load_pipeline", return_value=mock_pipeline):
            p = Predictor()
        assert p.threshold == DECISION_THRESHOLD

    def test_init_accepts_custom_threshold(self, mock_pipeline: MagicMock) -> None:
        """Un seuil custom passé en argument est respecté."""
        with patch.object(Predictor, "_load_pipeline", return_value=mock_pipeline):
            p = Predictor(threshold=0.5)
        assert p.threshold == 0.5

    def test_load_pipeline_raises_if_file_missing(self) -> None:
        """_load_pipeline lève FileNotFoundError si le .pkl n'existe pas."""
        fake_path = Path("models/this_does_not_exist.pkl")
        with pytest.raises(FileNotFoundError, match="Modèle introuvable"):
            Predictor(model_path=fake_path)


class TestPredictorPredict:
    """Logique de prédiction : extraction proba, application seuil, format sortie."""

    def test_predict_returns_probability_from_class_1(
        self, predictor: Predictor
    ) -> None:
        """La probabilité retournée est celle de la classe 1 (défaut), pas la 0."""
        # mock_pipeline retourne [[0.3, 0.7]] → on doit voir 0.7
        input_mock = MagicMock()
        input_mock.model_dump.return_value = {"feature": 1.0}
        result = predictor.predict(input_mock)
        assert result.probability == 0.7

    def test_predict_decision_above_threshold(self, predictor: Predictor) -> None:
        """Probabilité >= seuil → décision = 1 (refus)."""
        # mock retourne 0.7, threshold par défaut ≈ 0.334 → 0.7 > 0.334
        input_mock = MagicMock()
        input_mock.model_dump.return_value = {"feature": 1.0}
        result = predictor.predict(input_mock)
        assert result.decision == 1

    def test_predict_decision_below_threshold(
        self, mock_pipeline: MagicMock
    ) -> None:
        """Probabilité < seuil → décision = 0 (acceptation)."""
        mock_pipeline.predict_proba.return_value = np.array([[0.9, 0.1]])
        with patch.object(Predictor, "_load_pipeline", return_value=mock_pipeline):
            p = Predictor()
        input_mock = MagicMock()
        input_mock.model_dump.return_value = {"feature": 1.0}
        result = p.predict(input_mock)
        assert result.decision == 0
        assert result.probability == 0.1

    def test_predict_decision_equal_threshold(
        self, mock_pipeline: MagicMock
    ) -> None:
        """Probabilité exactement égale au seuil → décision = 1 (>=)."""
        threshold = 0.5
        mock_pipeline.predict_proba.return_value = np.array([[0.5, 0.5]])
        with patch.object(Predictor, "_load_pipeline", return_value=mock_pipeline):
            p = Predictor(threshold=threshold)
        input_mock = MagicMock()
        input_mock.model_dump.return_value = {"feature": 1.0}
        result = p.predict(input_mock)
        assert result.decision == 1

    def test_predict_uses_alias_for_dataframe(self, predictor: Predictor) -> None:
        """model_dump est appelé avec by_alias=True (noms originaux pandas)."""
        input_mock = MagicMock()
        input_mock.model_dump.return_value = {"feature": 1.0}
        predictor.predict(input_mock)
        input_mock.model_dump.assert_called_once_with(by_alias=True)

    def test_predict_returns_threshold_in_output(
        self, mock_pipeline: MagicMock
    ) -> None:
        """Le seuil utilisé est inclus dans PredictionOutput."""
        with patch.object(Predictor, "_load_pipeline", return_value=mock_pipeline):
            p = Predictor(threshold=0.42)
        input_mock = MagicMock()
        input_mock.model_dump.return_value = {"feature": 1.0}
        result = p.predict(input_mock)
        assert result.threshold == 0.42

    def test_predict_raises_if_pipeline_none(self, predictor: Predictor) -> None:
        """Si le pipeline est None, predict lève ModelNotLoadedError."""
        predictor.pipeline = None
        input_mock = MagicMock()
        input_mock.model_dump.return_value = {"feature": 1.0}
        with pytest.raises(ModelNotLoadedError):
            predictor.predict(input_mock)
            