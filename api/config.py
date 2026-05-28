"""Configuration centralisée de l'API.

Toutes les constantes du projet sont définies ici pour éviter les
magic numbers dispersés dans le code et faciliter la maintenance.
"""

import os
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

# --- Base de données ---
# Stratégie : Postgres (Docker local) en dev/prod, SQLite en fallback (CI/HF Spaces).
# DATABASE_URL prend priorité si définie, sinon fallback SQLite automatique.


DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "predictions.db"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

# --- Profiling  ---
# Activation conditionnelle des endpoints /profile/start et /profile/stop.
# Désactivé par défaut : les endpoints ne sont PAS montés sur l'app si false.
# Activation locale uniquement, jamais en prod HF Spaces.
ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() == "true"

# --- Backend d'inférence  ---
# Sélectionne le moteur d'inférence :
#   "joblib" (défaut) : pipeline scikit-learn complet (preprocessor + XGBClassifier)
#   "onnx"            : preprocessor sklearn + XGBClassifier converti en ONNX Runtime
# Permet de comparer les deux backends au benchmark (branche feature/benchmark-comparison)
# sans changer de code, et de basculer en prod via variable d'env.
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "joblib")

# Chemins des artefacts ONNX (utilisés seulement si INFERENCE_BACKEND == "onnx")
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_classifier.onnx"

# Répertoire de sortie pour les fichiers .prof
PROFILE_OUTPUT_DIR = PROJECT_ROOT / "reports"
