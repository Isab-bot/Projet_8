"""
Construction du reference dataset Evidently à partir de df_train_final (P6).

- Sample stratifié 10k sur TARGET
- Ajout des colonnes 'prediction_proba' et 'prediction' (seuil 0.3999)
- TARGET conservée pour le target drift Evidently
- Export parquet versionnable dans data/reference_data.parquet
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
SOURCE_PARQUET = Path(
    r"C:\Users\renar\Documents\Alternance\Projet_6\MLOps_1\data\df_train_final.parquet"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_champion.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "reference_data.parquet"

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42
TARGET_COL = "TARGET"


def main() -> None:
    # --- 1. Chargement ---
    if not SOURCE_PARQUET.exists():
        raise FileNotFoundError(f"Dataset source introuvable : {SOURCE_PARQUET}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. Lance d'abord extract_model.py."
        )

    print(f"[1/5] Chargement de {SOURCE_PARQUET.name}...")
    df = pd.read_parquet(SOURCE_PARQUET)
    print(f"      shape source : {df.shape}")

    if TARGET_COL not in df.columns:
        raise KeyError(f"Colonne '{TARGET_COL}' absente du dataset source.")

    # --- 2. Sampling stratifié ---
    print(f"[2/5] Sampling stratifié {SAMPLE_SIZE} sur '{TARGET_COL}'...")
    sample, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    sample = sample.reset_index(drop=True)
    print(f"      shape sample : {sample.shape}")
    print(f"      distribution TARGET : {sample[TARGET_COL].value_counts().to_dict()}")

    # --- 3. Chargement modèle + seuil ---
    print(f"[3/5] Chargement modèle + seuil...")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    threshold = metadata["decision_threshold"]
    print(f"      seuil = {threshold:.6f}")

    # --- 4. Prédictions ---
    print(f"[4/5] Calcul des prédictions...")
    # Colonnes à exclure : TARGET (label) + colonnes parasites du parquet P6
    NON_FEATURE_COLS = {TARGET_COL, "SK_ID_CURR", "Unnamed: 0"}
    feature_cols = [c for c in sample.columns if c not in NON_FEATURE_COLS]
    X = sample[feature_cols]

    # Sécurité : aligner l'ordre des colonnes sur celui attendu par le modèle
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        X = X[expected_cols]  # KeyError explicite si une colonne manque

    # Sécurité : vérifier le nombre de features attendu
    if hasattr(model, "n_features_in_") and model.n_features_in_ != X.shape[1]:
        raise ValueError(
            f"Mismatch features : modèle attend {model.n_features_in_}, "
            f"dataset fournit {X.shape[1]}"
        )

    proba = model.predict_proba(X)[:, 1]
    sample["prediction_proba"] = proba
    sample["prediction"] = (proba >= threshold).astype(int)
    print(f"      taux de prédictions positives : {sample['prediction'].mean():.4f}")
    print(f"      taux réel TARGET=1            : {sample[TARGET_COL].mean():.4f}")

    # --- 5. Export ---
    print(f"[5/5] Export parquet...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUTPUT_PATH, index=False, engine="pyarrow", compression="snappy")
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] {OUTPUT_PATH} ({size_mb:.2f} Mo, {len(sample)} lignes)")


if __name__ == "__main__":
    main()
    