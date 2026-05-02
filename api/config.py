"""Configuration centralisée de l'API.

Toutes les constantes du projet sont définies ici pour éviter les
magic numbers dispersés dans le code et faciliter la maintenance.
"""

from pathlib import Path

# --- Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_champion.pkl"

# --- Modèle ---
# Seuil de décision optimal (maximisation F3) issu du Projet 6
# Run MLflow: a3ff1e12347c4bfc9b484ac36916eb14
DECISION_THRESHOLD = 0.33381930539322036
N_FEATURES = 326

# --- API ---
API_VERSION = "0.1.0"
API_TITLE = "Credit Scoring API"
API_DESCRIPTION = (
    "API de scoring crédit basée sur le modèle XGBoost champion "
    "(F3=0.5583) entraîné dans le cadre du Projet 6."
)