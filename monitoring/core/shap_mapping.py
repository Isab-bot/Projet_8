"""Mapping des noms de features entre les 3 référentiels du projet.

Le projet manipule les mêmes features sous 3 noms différents selon l'endroit :

1. **Noms SHAP / ColumnTransformer** : préfixés par le type de transformer
   (`num__EXT_SOURCE_2`, `cat__CODE_GENDER_M`, etc.). C'est la forme produite
   par le pipeline scikit-learn entraîné au Projet 6. Utile uniquement pour
   référencer les features importantes du modèle.

2. **Noms parquet** : les noms bruts tels que sortis du preprocessing du
   Projet 6 et stockés dans `data/reference_data.parquet`. Pour 10 colonnes
   d'agrégation issues de `groupby().agg(lambda)`, pandas a conservé l'alias
   d'agrégation entre crochets : `BUREAU_CREDIT_ACTIVE_<lambda>`,
   `POS_NAME_CONTRACT_STATUS_<lambda_0>_mean`, etc.

3. **Noms SQL** : les noms tels qu'ils figurent dans la table `predictions`
   de la base PostgreSQL alimentée par l'API. Pydantic a transformé
   `<lambda>` en `_lambda` (les crochets ne sont pas des caractères valides
   en attribut Python), donc les colonnes deviennent
   `BUREAU_CREDIT_ACTIVE_lambda`, `POS_NAME_CONTRACT_STATUS_lambda_0_mean`,
   etc.

**Décision d'architecture** (cf. design-doc §2.2 et notes de reprise) : la
table SQL est la source de vérité de la production, on ne la modifie pas.
Au moment de comparer référence et production via Evidently, on renomme
les colonnes du DataFrame parquet pour qu'elles matchent les noms SQL.

Ce module expose :

- ``SHAP_TOP_5_FEATURES`` : liste figée des 5 features à monitorer en
  drift détaillé (noms bruts utilisés à la fois côté parquet et côté SQL,
  car aucune des 5 ne fait partie des colonnes ``<lambda>``).
- ``SHAP_TO_RAW`` : mapping nom SHAP transformer → nom brut, pour pouvoir
  citer la provenance dans la doc/UI si besoin.
- ``PARQUET_TO_SQL_RENAME`` : mapping des 10 colonnes ``<lambda>`` du
  parquet vers leurs équivalents SQL ``_lambda``.
- ``align_parquet_to_sql(df)`` : applique le renommage sur un DataFrame
  chargé depuis le parquet de référence.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Top 5 features SHAP à monitorer en drift détaillé
# ---------------------------------------------------------------------------
# Liste figée par le design-doc §2.2. Représente ~98% de l'importance SHAP
# cumulée du top 20 du modèle XGBoost champion du Projet 6.
#
# Les 5 noms ci-dessous sont identiques côté parquet ET côté SQL (aucun ne
# contient `<lambda>`), donc une seule liste suffit.

SHAP_TOP_5_FEATURES: list[str] = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CODE_GENDER",
    "CREDIT_TERM",
    "DAYS_EMPLOYED",
]

# ---------------------------------------------------------------------------
# Mapping nom SHAP transformer -> nom brut
# ---------------------------------------------------------------------------
# Utilisé pour la traçabilité (afficher dans l'UI "Cette feature s'appelle
# `num__EXT_SOURCE_2` dans le pipeline SHAP, mais on la suit ici sous le nom
# brut `EXT_SOURCE_2`"). Pas indispensable pour le drift, mais utile pour la
# soutenance.

SHAP_TO_RAW: dict[str, str] = {
    "num__EXT_SOURCE_2": "EXT_SOURCE_2",
    "num__EXT_SOURCE_3": "EXT_SOURCE_3",
    "cat__CODE_GENDER_M": "CODE_GENDER",
    "num__CREDIT_TERM": "CREDIT_TERM",
    "num__DAYS_EMPLOYED": "DAYS_EMPLOYED",
}

# ---------------------------------------------------------------------------
# Renommage des 10 colonnes `<lambda>` du parquet vers leur forme SQL
# ---------------------------------------------------------------------------
# Source : inspection du parquet (script inspect_reference_parquet.py de la
# session précédente) et du schéma SQL de la table `predictions`.
#
# Règle : remplacer `<lambda>` par `lambda` et `<lambda_N>` par `lambda_N`.
# Les autres colonnes (321 sur 331) ne changent pas de nom entre parquet et
# SQL, donc elles ne figurent pas dans ce mapping (renommage idempotent).

PARQUET_TO_SQL_RENAME: dict[str, str] = {
    "BUREAU_CREDIT_ACTIVE_<lambda>": "BUREAU_CREDIT_ACTIVE_lambda",
    "POS_NAME_CONTRACT_STATUS_<lambda_0>_mean": "POS_NAME_CONTRACT_STATUS_lambda_0_mean",
    "POS_NAME_CONTRACT_STATUS_<lambda_0>_max": "POS_NAME_CONTRACT_STATUS_lambda_0_max",
    "POS_NAME_CONTRACT_STATUS_<lambda_0>_sum": "POS_NAME_CONTRACT_STATUS_lambda_0_sum",
    "POS_NAME_CONTRACT_STATUS_<lambda_1>_mean": "POS_NAME_CONTRACT_STATUS_lambda_1_mean",
    "POS_NAME_CONTRACT_STATUS_<lambda_1>_max": "POS_NAME_CONTRACT_STATUS_lambda_1_max",
    "POS_NAME_CONTRACT_STATUS_<lambda_1>_sum": "POS_NAME_CONTRACT_STATUS_lambda_1_sum",
    "PREV_NAME_CONTRACT_STATUS_<lambda_0>": "PREV_NAME_CONTRACT_STATUS_lambda_0",
    "PREV_NAME_CONTRACT_STATUS_<lambda_1>": "PREV_NAME_CONTRACT_STATUS_lambda_1",
    "PREV_NAME_CONTRACT_STATUS_<lambda_2>": "PREV_NAME_CONTRACT_STATUS_lambda_2",
}


def align_parquet_to_sql(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes du parquet de référence pour matcher la SQL.

    Renomme les 10 colonnes contenant ``<lambda>`` vers leur forme
    ``_lambda`` utilisée par la table SQL ``predictions``. Les autres
    colonnes sont laissées intactes.

    Le DataFrame d'entrée n'est pas modifié en place : on retourne une
    nouvelle vue avec colonnes renommées (``pandas.DataFrame.rename``
    par défaut).

    Parameters
    ----------
    df :
        DataFrame chargé depuis ``data/reference_data.parquet``.

    Returns
    -------
    pandas.DataFrame
        Même DataFrame avec les 10 colonnes ``<lambda>`` renommées en
        ``_lambda``. Si aucune des colonnes du mapping n'est présente
        (cas par exemple d'un DataFrame déjà aligné), le DataFrame est
        retourné inchangé.
    """
    return df.rename(columns=PARQUET_TO_SQL_RENAME)
