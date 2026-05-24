"""Accès en lecture seule à la base PostgreSQL des prédictions.

Ce module isole toute la logique de connexion et de requêtage SQL.
Discipline : SELECT-only, jamais d'INSERT/UPDATE/DELETE/ALTER. Streamlit
consomme les données, l'API les écrit — séparation stricte des rôles.

Configuration :
    - Variable d'env DATABASE_URL (défaut: localhost:5433/credit_scoring)
    - L'URL doit utiliser le driver psycopg v3 (postgresql+psycopg://...)

Cache :
    Les fonctions exposées sont décorées avec @st.cache_data(ttl=60) côté
    Streamlit. Ici on garde des fonctions pures sans cache pour les tests
    unitaires. Le cache est appliqué dans les pages.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_DATABASE_URL: str = (
    "postgresql+psycopg://credit_user:credit_pass@localhost:5433/credit_scoring"
)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Retourne un Engine SQLAlchemy partagé pour toute l'application.

    Le pool de connexions est géré par SQLAlchemy. L'URL provient de la
    variable d'environnement DATABASE_URL, ou utilise le défaut local.

    Returns:
        Engine SQLAlchemy connecté à PostgreSQL.
    """
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    return create_engine(url, pool_pre_ping=True)


def get_predictions_count(engine: Engine | None = None) -> int:
    """Compte total des prédictions en base.

    Args:
        engine: Engine SQLAlchemy (optionnel, créé si non fourni).

    Returns:
        Nombre total de lignes dans la table predictions.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM predictions"))
        return int(result.scalar() or 0)


def get_instrumented_predictions_count(engine: Engine | None = None) -> int:
    """Compte des prédictions avec instrumentation (latency_ms IS NOT NULL).

    Args:
        engine: Engine SQLAlchemy (optionnel).

    Returns:
        Nombre de lignes avec latency_ms renseignée.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM predictions WHERE latency_ms IS NOT NULL")
        )
        return int(result.scalar() or 0)


def get_last_prediction_timestamp(
    engine: Engine | None = None,
) -> pd.Timestamp | None:
    """Timestamp de la prédiction la plus récente en base.

    Args:
        engine: Engine SQLAlchemy (optionnel).

    Returns:
        Timestamp UTC de la dernière prédiction, ou None si table vide.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(timestamp) FROM predictions")
        )
        ts = result.scalar()
        return pd.Timestamp(ts) if ts is not None else None


def fetch_predictions(
    limit: int = 100,
    columns: list[str] | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Récupère les N dernières prédictions en DataFrame.

    Args:
        limit: Nombre de lignes à récupérer (défaut 100).
        columns: Liste de colonnes à sélectionner (défaut: toutes).
        engine: Engine SQLAlchemy (optionnel).

    Returns:
        DataFrame ordonné par timestamp décroissant.
    """
    engine = engine or get_engine()
    cols_sql = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    query = (
        f"SELECT {cols_sql} FROM predictions "
        f"ORDER BY timestamp DESC LIMIT {int(limit)}"
    )
    return pd.read_sql(query, engine)
