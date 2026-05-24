"""Tests unitaires pour monitoring/core/drift.py.

Ne teste pas les 3 fonctions build_*_drift_report (dépendent d'Evidently,
validées par smoke test manuel). Teste les helpers de configuration et
de parsing : make_data_definition, extract_drift_summary, extract_column_drift.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from monitoring.core.drift import (
    extract_column_drift,
    extract_drift_summary,
    make_data_definition,
)


class TestMakeDataDefinition:
    """Tests sur la détection automatique des types par dtype pandas."""

    def test_detecte_les_colonnes_numeriques(self):
        df = pd.DataFrame(
            {
                "a_float": [1.0, 2.0, 3.0],
                "b_int": [1, 2, 3],
            }
        )
        data_def = make_data_definition(df)
        assert "a_float" in data_def.numerical_columns
        assert "b_int" in data_def.numerical_columns

    def test_detecte_les_colonnes_categorielles(self):
        df = pd.DataFrame(
            {
                "a_str": ["x", "y", "z"],
                "b_obj": pd.Series(["A", "B", "C"], dtype="object"),
            }
        )
        data_def = make_data_definition(df)
        assert "a_str" in data_def.categorical_columns
        assert "b_obj" in data_def.categorical_columns

    def test_exclut_bool_du_numerique(self):
        # Décision technique : les booléens vont en catégoriel pour éviter
        # le crash quantile d'Evidently (voir docstring de make_data_definition).
        df = pd.DataFrame({"flag": [True, False, True]})
        data_def = make_data_definition(df)
        assert "flag" not in data_def.numerical_columns
        assert "flag" in data_def.categorical_columns

    def test_mix_complet(self):
        df = pd.DataFrame(
            {
                "num": [1.0, 2.0],
                "cat": ["A", "B"],
                "flag": [True, False],
            }
        )
        data_def = make_data_definition(df)
        assert data_def.numerical_columns == ["num"]
        assert set(data_def.categorical_columns) == {"cat", "flag"}


def _build_snapshot_mock(metrics: list[dict]) -> MagicMock:
    """Construit un mock de Snapshot Evidently avec une méthode dict() configurée."""
    snap = MagicMock()
    snap.dict.return_value = {"metrics": metrics, "tests": []}
    return snap


class TestExtractDriftSummary:
    """Tests sur l'extraction du résumé global DriftedColumnsCount."""

    def test_extrait_count_et_share(self):
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                    "value": {"count": 3, "share": 0.5},
                }
            ]
        )
        result = extract_drift_summary(snap)
        assert result == {"count": 3, "share": 0.5}

    def test_count_castee_en_int(self):
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                    "value": {"count": 3.0, "share": 0.6},
                }
            ]
        )
        result = extract_drift_summary(snap)
        assert isinstance(result["count"], int)
        assert result["count"] == 3

    def test_share_castee_en_float(self):
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                    "value": {"count": 1, "share": "0.25"},
                }
            ]
        )
        result = extract_drift_summary(snap)
        assert isinstance(result["share"], float)
        assert result["share"] == pytest.approx(0.25)

    def test_retourne_none_si_aucun_drifted_columns_count(self):
        # Cas du rapport prediction_proba (ValueDrift seul, pas d'agrégat).
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "ValueDrift(column=prediction_proba,method=K-S p_value,threshold=0.05)",
                    "value": 0.001,
                }
            ]
        )
        result = extract_drift_summary(snap)
        assert result == {"count": None, "share": None}


class TestExtractColumnDrift:
    """Tests sur l'extraction du détail par colonne."""

    def test_extrait_une_seule_metric(self):
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "ValueDrift(column=EXT_SOURCE_2,method=K-S p_value,threshold=0.05)",
                    "value": 0.001,
                }
            ]
        )
        df = extract_column_drift(snap)
        assert len(df) == 1
        assert df.iloc[0]["column"] == "EXT_SOURCE_2"
        assert df.iloc[0]["method"] == "K-S p_value"
        assert df.iloc[0]["p_value"] == pytest.approx(0.001)
        assert df.iloc[0]["drift_detected"]

    def test_drift_detected_false_au_dessus_du_seuil(self):
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "ValueDrift(column=EXT_SOURCE_3,method=K-S p_value,threshold=0.05)",
                    "value": 0.5,
                }
            ]
        )
        df = extract_column_drift(snap)
        assert not df.iloc[0]["drift_detected"]

    def test_drift_detected_false_exactement_au_seuil(self):
        # Convention : p_value < 0.05 (strict) → drift_detected
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "ValueDrift(column=X,method=K-S p_value,threshold=0.05)",
                    "value": 0.05,
                }
            ]
        )
        df = extract_column_drift(snap)
        assert not df.iloc[0]["drift_detected"]

    def test_tri_par_p_value_croissante(self):
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "ValueDrift(column=A,method=K-S p_value,threshold=0.05)",
                    "value": 0.8,
                },
                {
                    "metric_name": "ValueDrift(column=B,method=K-S p_value,threshold=0.05)",
                    "value": 0.001,
                },
                {
                    "metric_name": "ValueDrift(column=C,method=K-S p_value,threshold=0.05)",
                    "value": 0.3,
                },
            ]
        )
        df = extract_column_drift(snap)
        assert df["column"].tolist() == ["B", "C", "A"]

    def test_ignore_les_metrics_non_value_drift(self):
        # DriftedColumnsCount ne doit pas apparaître dans le détail par colonne.
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                    "value": {"count": 2, "share": 0.4},
                },
                {
                    "metric_name": "ValueDrift(column=A,method=K-S p_value,threshold=0.05)",
                    "value": 0.01,
                },
            ]
        )
        df = extract_column_drift(snap)
        assert len(df) == 1
        assert df.iloc[0]["column"] == "A"

    def test_dataframe_vide_si_aucun_value_drift(self):
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                    "value": {"count": 0, "share": 0.0},
                }
            ]
        )
        df = extract_column_drift(snap)
        assert df.empty

    def test_detecte_methode_z_test_pour_categoriel(self):
        snap = _build_snapshot_mock(
            [
                {
                    "metric_name": "ValueDrift(column=CODE_GENDER,method=Z-test p_value,threshold=0.05)",
                    "value": 0.9,
                }
            ]
        )
        df = extract_column_drift(snap)
        assert df.iloc[0]["method"] == "Z-test p_value"
