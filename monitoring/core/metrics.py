"""Calcul de métriques agrégées sur les prédictions persistées.

Fonctions pures, sans dépendance à Streamlit ni à la DB. Elles prennent
un DataFrame en entrée et renvoient des valeurs scalaires ou des dicts.

Testables unitairement avec un DataFrame factice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 95.0, 99.0)


def acceptance_rate(df: pd.DataFrame) -> float:
    """Taux d'acceptation = proportion de prédictions où prediction == 0.

    En crédit scoring, prediction == 0 signifie "client accepté" (pas de
    défaut prédit). Le complément à 1 donne le taux de refus.

    Args:
        df: DataFrame avec une colonne 'prediction' (entier 0/1).

    Returns:
        Float entre 0 et 1. Retourne NaN si df est vide.
    """
    if len(df) == 0 or "prediction" not in df.columns:
        return float("nan")
    return float((df["prediction"] == 0).mean())


def score_distribution(df: pd.DataFrame) -> pd.Series:
    """Renvoie la série des prediction_proba pour visualisation.

    Args:
        df: DataFrame avec une colonne 'prediction_proba'.

    Returns:
        Series des probabilités, NaN exclus.
    """
    if "prediction_proba" not in df.columns:
        return pd.Series([], dtype=float)
    return df["prediction_proba"].dropna()


def latency_percentiles(
    df: pd.DataFrame,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    """Percentiles de la latence API totale (en ms).

    Args:
        df: DataFrame avec une colonne 'latency_ms'.
        percentiles: Liste de percentiles à calculer (en pourcentage).

    Returns:
        Dict {'mean', 'p50', 'p95', 'p99'} avec NaN si données absentes.
    """
    return _percentiles(df, column="latency_ms", percentiles=percentiles)


def inference_percentiles(
    df: pd.DataFrame,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    """Percentiles du temps d'inférence du modèle (en ms).

    Args:
        df: DataFrame avec une colonne 'inference_ms'.
        percentiles: Liste de percentiles à calculer (en pourcentage).

    Returns:
        Dict {'mean', 'p50', 'p95', 'p99'} avec NaN si données absentes.
    """
    return _percentiles(df, column="inference_ms", percentiles=percentiles)


def _percentiles(
    df: pd.DataFrame,
    column: str,
    percentiles: tuple[float, ...],
) -> dict[str, float]:
    """Implémentation partagée des percentiles."""
    result: dict[str, float] = {"mean": float("nan")}
    for p in percentiles:
        result[f"p{int(p)}"] = float("nan")

    if column not in df.columns:
        return result

    series = df[column].dropna()
    if len(series) == 0:
        return result

    result["mean"] = float(series.mean())
    computed = np.percentile(series, percentiles)
    for p, value in zip(percentiles, computed, strict=True):
        result[f"p{int(p)}"] = float(value)

    return result
