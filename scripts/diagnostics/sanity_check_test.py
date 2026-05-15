"""
Sanity check : reproduit les prédictions du notebook P6 sur df_test_final.
Doit donner 38.8% de prédictions positives (notebook cellule 4).
"""
import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_champion.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
TEST_PARQUET = Path(
    r"C:\Users\renar\Documents\Alternance\Projet_6\MLOps_1\data\df_test_final.parquet"
)


def main() -> None:
    model = joblib.load(MODEL_PATH)
    threshold = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["decision_threshold"]
    print(f"Seuil utilisé : {threshold:.6f}")

    df = pd.read_parquet(TEST_PARQUET)
    print(f"Shape df_test_final : {df.shape}")

    NON_FEATURE_COLS = {"TARGET", "SK_ID_CURR", "Unnamed: 0"}
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]

    if hasattr(model, "feature_names_in_"):
        X = X[list(model.feature_names_in_)]

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    print(f"\n--- Résultats sur df_test_final ({len(df)} obs) ---")
    print(f"Taux prédictions positives : {pred.mean():.4f}")
    print("Attendu (notebook P6)      : 0.3880")
    print(f"Moyenne probas             : {proba.mean():.4f}")
    print("Attendu (notebook P6)      : 0.3119")


if __name__ == "__main__":
    main()
