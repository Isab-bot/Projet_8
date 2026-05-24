"""Simulation de trafic production vers l'API de scoring crédit.

Ce script envoie un volume contrôlé de requêtes vers /predict pour alimenter
la table predictions, sur laquelle s'appuie le dashboard de monitoring
Streamlit + Evidently de l'étape 8.

Source des données : data/raw/df_test_final.parquet (48 744 obs, 326 features).
Volume par défaut : 3 000 requêtes échantillonnées avec seed fixée.
Cadence : séquentielle, sans pause.

Drift artificiel injecté sur 3 features du top 5 SHAP du modèle Projet 6 :
    - EXT_SOURCE_2  : multiplicateur x 0.85
    - EXT_SOURCE_3  : bruit gaussien N(0, 0.05) puis clip dans [0, 1]
    - DAYS_EMPLOYED : shift additif de -1 (unité = année, équivaut à
                      un an d'ancienneté supplémentaire en moyenne)

Politique d'erreur : best-effort. Chaque requête échouée est loguée et le
script poursuit. Un récap final indique les succès et les erreurs.

Usage :
    # API en local sur le port par défaut
    uv run python scripts/simulate_production.py

    # API sur une autre URL (ex. déployée)
    $env:API_URL = "https://my-api.hf.space"
    uv run python scripts/simulate_production.py

Tests : les fonctions pures (sample_production_data, inject_drift,
row_to_payload) sont importables et testées dans tests/unit/.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes de configuration
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
N_SAMPLES: int = 3000

DEFAULT_PARQUET_PATH: str = "data/raw/df_test_final.parquet"
DEFAULT_API_URL: str = "http://localhost:8000"

# Colonnes à exclure du payload (présentes dans df_test_final mais pas
# attendues par l'API : identifiants et résidus d'export pandas)
COLS_TO_DROP: list[str] = ["SK_ID_CURR", "Unnamed: 0"]

# Configuration du drift artificiel injecté sur le top 5 SHAP
# Justification : voir design-doc étape 8 section 2.2 et discussion
# branche 2 (calibrage à l'échelle réelle des valeurs).
DRIFT_CONFIG: dict[str, dict] = {
    "EXT_SOURCE_2": {"type": "multiplier", "value": 0.85},
    "EXT_SOURCE_3": {"type": "gaussian_noise_clipped", "sigma": 0.05,
                     "clip_min": 0.0, "clip_max": 1.0},
    "DAYS_EMPLOYED": {"type": "additive_shift", "value": -1.0},
}

LOG_EVERY: int = 100
REQUEST_TIMEOUT_SEC: int = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulate_production")


# ---------------------------------------------------------------------------
# Fonctions pures (importables et testables)
# ---------------------------------------------------------------------------

def sample_production_data(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Échantillonne n lignes du DataFrame source de façon reproductible.

    Args:
        df: DataFrame source (df_test_final).
        n: Nombre de lignes à échantillonner.
        seed: Seed pour reproductibilité.

    Returns:
        DataFrame de n lignes, index réinitialisé.

    Raises:
        ValueError: si n est supérieur au nombre de lignes du DataFrame.
    """
    if n > len(df):
        raise ValueError(
            f"Volume demandé ({n}) supérieur au nombre de lignes disponibles "
            f"({len(df)})."
        )
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def inject_drift(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Applique les transformations de drift artificiel sur le DataFrame.

    Ne mute pas le DataFrame d'entrée (renvoie une copie).

    Args:
        df: DataFrame contenant les colonnes EXT_SOURCE_2, EXT_SOURCE_3,
            DAYS_EMPLOYED (les autres colonnes sont laissées intactes).
        seed: Seed pour reproductibilité du bruit gaussien.

    Returns:
        Nouveau DataFrame avec les colonnes drifted.
    """
    drifted = df.copy()
    rng = np.random.default_rng(seed)

    for column, config in DRIFT_CONFIG.items():
        if column not in drifted.columns:
            logger.warning(
                "Feature %s absente du DataFrame, drift ignoré.", column,
            )
            continue

        kind = config["type"]
        if kind == "multiplier":
            drifted[column] = drifted[column] * config["value"]
        elif kind == "additive_shift":
            drifted[column] = drifted[column] + config["value"]
        elif kind == "gaussian_noise_clipped":
            noise = rng.normal(0.0, config["sigma"], size=len(drifted))
            drifted[column] = np.clip(
                drifted[column] + noise,
                config["clip_min"],
                config["clip_max"],
            )
        else:
            raise ValueError(f"Type de drift non supporté : {kind!r}")

    return drifted


def row_to_payload(row: pd.Series) -> dict:
    """Convertit une ligne pandas en dict JSON-serializable.

    Gère la conversion des types numpy vers les types Python natifs et
    transforme les NaN en None (Pydantic accepte None mais pas NaN).

    Args:
        row: Une ligne du DataFrame production (avec uniquement les
            326 features attendues par l'API).

    Returns:
        Dictionnaire prêt à être sérialisé en JSON.
    """
    payload = {}
    for key, value in row.items():
        if value is None:
            payload[key] = None
        elif isinstance(value, float) and math.isnan(value):
            payload[key] = None
        elif isinstance(value, (np.integer,)):
            payload[key] = int(value)
        elif isinstance(value, (np.floating,)):
            if math.isnan(float(value)):
                payload[key] = None
            else:
                payload[key] = float(value)
        elif isinstance(value, np.bool_):
            payload[key] = bool(value)
        else:
            payload[key] = value
    return payload


def post_prediction(
    url: str,
    payload: dict,
    timeout: int = REQUEST_TIMEOUT_SEC,
) -> dict | None:
    """Envoie une requête POST vers /predict de l'API.

    Args:
        url: URL complète de l'endpoint (ex. "http://localhost:8000/predict").
        payload: Dict des 326 features à envoyer.
        timeout: Timeout en secondes.

    Returns:
        Le dict de la réponse JSON si succès, None si échec.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        logger.error("HTTP %s : %s", exc.code, exc.reason)
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.error("Erreur réseau : %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Erreur inattendue : %s", exc)
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    """Point d'entrée : charge le parquet, échantillonne, drift, envoie."""
    parquet_path = os.getenv("PARQUET_PATH", DEFAULT_PARQUET_PATH)
    api_base = os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")
    predict_url = f"{api_base}/predict"

    logger.info("Configuration :")
    logger.info("  parquet  : %s", parquet_path)
    logger.info("  API URL  : %s", predict_url)
    logger.info("  N        : %d", N_SAMPLES)
    logger.info("  seed     : %d", RANDOM_SEED)
    logger.info("  drift    : %s", list(DRIFT_CONFIG.keys()))

    parquet = Path(parquet_path)
    if not parquet.exists():
        logger.error("Fichier introuvable : %s", parquet)
        return 1

    logger.info("Chargement du parquet...")
    df = pd.read_parquet(parquet)
    logger.info("Parquet chargé : %d lignes x %d colonnes",
                len(df), len(df.columns))

    # Filtrer les colonnes non-features
    features_df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    logger.info("Après exclusion des non-features : %d colonnes",
                len(features_df.columns))

    if len(features_df.columns) != 326:
        logger.warning(
            "Nombre inattendu de colonnes (%d, attendu 326). "
            "Vérifier le schéma du parquet.",
            len(features_df.columns),
        )

    # Échantillonner
    sampled = sample_production_data(features_df, n=N_SAMPLES, seed=RANDOM_SEED)
    logger.info("Échantillonnage : %d lignes", len(sampled))

    # Injecter le drift
    drifted = inject_drift(sampled, seed=RANDOM_SEED)
    logger.info("Drift injecté sur : %s", list(DRIFT_CONFIG.keys()))

    # Envoyer les requêtes
    logger.info("Envoi de %d requêtes vers %s ...", N_SAMPLES, predict_url)
    started = time.perf_counter()
    n_success = 0
    n_error = 0

    for i, (_, row) in enumerate(drifted.iterrows(), start=1):
        payload = row_to_payload(row)
        result = post_prediction(predict_url, payload)
        if result is not None:
            n_success += 1
        else:
            n_error += 1

        if i % LOG_EVERY == 0:
            elapsed = time.perf_counter() - started
            rate = i / elapsed if elapsed > 0 else 0
            logger.info(
                "  %d/%d (succès=%d, erreurs=%d, %.1f req/s)",
                i, N_SAMPLES, n_success, n_error, rate,
            )

    elapsed = time.perf_counter() - started
    logger.info("Terminé en %.1f s", elapsed)
    logger.info("Récap : %d succès, %d erreurs", n_success, n_error)

    if n_error > 0:
        logger.warning(
            "%.1f%% d'échecs (%d/%d).",
            100.0 * n_error / N_SAMPLES, n_error, N_SAMPLES,
        )

    return 0 if n_error == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
