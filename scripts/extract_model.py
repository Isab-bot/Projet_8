"""
Extraction du modèle XGBoost champion du Projet 6 vers le Projet 8.

Source : MLflow run a3ff1e12347c4bfc9b484ac36916eb14 (P6, FINAL_TEST_XGBoost)
Cible  : models/xgboost_champion.pkl + models/model_metadata.json
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

# --- Configuration ---
SOURCE_PKL = Path(
    r"C:\Users\renar\Documents\Alternance\Projet_6\MLOps_1\mlruns\9"
    r"\models\m-30fce0016ff849fe863eba9abcdce510\artifacts\model.pkl"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_PKL = MODELS_DIR / "xgboost_champion.pkl"
METADATA_JSON = MODELS_DIR / "model_metadata.json"

METADATA = {
    "model_name": "xgboost_champion",
    "source_project": "Projet_6",
    "mlflow_run_uuid": "a3ff1e12347c4bfc9b484ac36916eb14",
    "mlflow_run_name": "FINAL_TEST_XGBoost",
    "mlflow_model_id": "m-30fce0016ff849fe863eba9abcdce510",
    "mlflow_experiment": "Credit_Risk_04_Final_Evaluation_Test_Set",
    "mlflow_experiment_id": 9,
    "tracking_uri": "sqlite:///C:/MLflow_tracking/credit_risk/mlflow.db",
    "metrics": {
        "test_f3_score": 0.5583,
    },
    "decision_threshold": 0.33381930539322036,
    "extracted_at": datetime.now(timezone.utc).isoformat(),
}


def main() -> None:
    if not SOURCE_PKL.exists():
        raise FileNotFoundError(f"Modèle source introuvable : {SOURCE_PKL}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SOURCE_PKL, TARGET_PKL)
    size_mb = TARGET_PKL.stat().st_size / (1024 * 1024)
    print(f"[OK] Modèle copié : {TARGET_PKL} ({size_mb:.2f} Mo)")

    METADATA_JSON.write_text(
        json.dumps(METADATA, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OK] Métadonnées : {METADATA_JSON}")

    # Vérification
    import joblib
    model = joblib.load(TARGET_PKL)
    print(f"[OK] Modèle chargé : {type(model).__name__}")
    if hasattr(model, "n_features_in_"):
        print(f"     n_features_in_ = {model.n_features_in_}")


if __name__ == "__main__":
    main()