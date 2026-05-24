"""Tests unitaires pour monitoring/core/shap_mapping.py."""

from __future__ import annotations

import pandas as pd

from monitoring.core.shap_mapping import (
    PARQUET_TO_SQL_RENAME,
    SHAP_TO_RAW,
    SHAP_TOP_5_FEATURES,
    align_parquet_to_sql,
)


class TestShapTop5Features:
    """Tests sur la constante SHAP_TOP_5_FEATURES."""

    def test_contient_exactement_5_features(self):
        assert len(SHAP_TOP_5_FEATURES) == 5

    def test_noms_attendus(self):
        assert SHAP_TOP_5_FEATURES == [
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "CODE_GENDER",
            "CREDIT_TERM",
            "DAYS_EMPLOYED",
        ]


class TestShapToRaw:
    """Tests sur le mapping nom SHAP transformer -> nom brut."""

    def test_contient_5_entrees(self):
        assert len(SHAP_TO_RAW) == 5

    def test_les_valeurs_sont_dans_shap_top_5(self):
        # Toutes les cibles du mapping doivent être dans SHAP_TOP_5_FEATURES.
        assert set(SHAP_TO_RAW.values()) == set(SHAP_TOP_5_FEATURES)


class TestParquetToSqlRename:
    """Tests sur le mapping des colonnes <lambda> du parquet vers la SQL."""

    def test_contient_10_entrees(self):
        assert len(PARQUET_TO_SQL_RENAME) == 10

    def test_toutes_les_cles_contiennent_lambda_entre_crochets(self):
        for key in PARQUET_TO_SQL_RENAME:
            assert "<lambda" in key

    def test_aucune_valeur_ne_contient_crochets(self):
        for value in PARQUET_TO_SQL_RENAME.values():
            assert "<" not in value and ">" not in value

    def test_la_regle_est_supprimer_les_crochets(self):
        # Le renommage doit transformer `<lambda>` en `lambda` et `<lambda_N>`
        # en `lambda_N` (suppression simple des deux crochets).
        for parquet_name, sql_name in PARQUET_TO_SQL_RENAME.items():
            expected = parquet_name.replace("<", "").replace(">", "")
            assert sql_name == expected


class TestAlignParquetToSql:
    """Tests sur la fonction align_parquet_to_sql."""

    def test_renomme_les_colonnes_lambda(self):
        df = pd.DataFrame(
            {
                "BUREAU_CREDIT_ACTIVE_<lambda>": [1, 2, 3],
                "EXT_SOURCE_2": [0.1, 0.2, 0.3],
            }
        )
        result = align_parquet_to_sql(df)
        assert "BUREAU_CREDIT_ACTIVE_lambda" in result.columns
        assert "BUREAU_CREDIT_ACTIVE_<lambda>" not in result.columns
        # Les autres colonnes restent inchangées.
        assert "EXT_SOURCE_2" in result.columns

    def test_preserve_les_donnees(self):
        df = pd.DataFrame({"BUREAU_CREDIT_ACTIVE_<lambda>": [1, 2, 3]})
        result = align_parquet_to_sql(df)
        assert result["BUREAU_CREDIT_ACTIVE_lambda"].tolist() == [1, 2, 3]

    def test_ne_modifie_pas_le_dataframe_dorigine(self):
        df = pd.DataFrame({"BUREAU_CREDIT_ACTIVE_<lambda>": [1, 2, 3]})
        original_cols = list(df.columns)
        align_parquet_to_sql(df)
        # Le DataFrame d'origine ne doit pas avoir été modifié en place.
        assert list(df.columns) == original_cols

    def test_idempotence_sur_dataframe_deja_aligne(self):
        # Si on applique deux fois la fonction, le résultat doit être identique
        # à une seule application (les noms SQL ne contiennent pas `<lambda>`).
        df = pd.DataFrame(
            {
                "BUREAU_CREDIT_ACTIVE_<lambda>": [1, 2],
                "EXT_SOURCE_2": [0.1, 0.2],
            }
        )
        once = align_parquet_to_sql(df)
        twice = align_parquet_to_sql(once)
        assert list(once.columns) == list(twice.columns)

    def test_dataframe_sans_colonne_a_renommer(self):
        # Cas où aucune colonne du mapping n'est présente : retour inchangé.
        df = pd.DataFrame({"EXT_SOURCE_2": [0.1, 0.2], "CODE_GENDER": ["M", "F"]})
        result = align_parquet_to_sql(df)
        assert list(result.columns) == ["EXT_SOURCE_2", "CODE_GENDER"]
