"""Convertit le pipeline XGBoost (Projet 6) en deux artefacts séparés.

Stratégie alignée sur Q1 du design-doc étape 9 :
- Le ColumnTransformer (preprocessing sklearn) reste en sklearn.
- Le XGBClassifier final est converti en ONNX pour inférence via ONNX Runtime.

Le pipeline source `models/xgboost_champion.pkl` est un Pipeline scikit-learn
avec deux étapes :
  - 'preprocessor' : ColumnTransformer (SimpleImputer + StandardScaler)
  - 'classifier'   : XGBClassifier

Ce script produit deux fichiers dans models/ :
  - preprocessor.pkl          : ColumnTransformer extrait
  - xgboost_classifier.onnx   : XGBClassifier converti en ONNX

Usage :
  uv run --group onnx python scripts/convert_to_onnx.py

Note : --group onnx active le groupe de dépendances onnx (onnxmltools,
onnxruntime, onnx) défini dans pyproject.toml.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Suppress XGBoost serialization warning (compat ascendante, inoffensif)
warnings.filterwarnings(
    "ignore",
    message=".*If you are loading a serialized model.*",
    category=UserWarning,
)

# --- Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PIPELINE = PROJECT_ROOT / "models" / "xgboost_champion.pkl"
TARGET_PREPROCESSOR = PROJECT_ROOT / "models" / "preprocessor.pkl"
TARGET_ONNX = PROJECT_ROOT / "models" / "xgboost_classifier.onnx"

# Pour mesurer la shape du tenseur d'entrée du XGBClassifier, on a besoin
# d'un échantillon de données BRUTES à passer dans le preprocessor.
SAMPLE_PARQUET = PROJECT_ROOT / "data" / "raw" / "df_test_final.parquet"
COLUMNS_TO_EXCLUDE = ["SK_ID_CURR", "Unnamed: 0", "TARGET"]

# --- Configuration ONNX ---
# target_opset 15 est stable, supporté par onnxruntime 1.20+, et compatible
# avec onnxmltools 1.13+.
TARGET_OPSET = 15
ONNX_MODEL_NAME = "xgboost_credit_scoring_classifier"


def load_pipeline() -> object:
    """Charge le pipeline source depuis disque."""
    if not SOURCE_PIPELINE.exists():
        print(f"[ERREUR] Pipeline source introuvable : {SOURCE_PIPELINE}")
        print("        Lance d'abord scripts/extract_model.py.")
        sys.exit(1)
    print(f"[1/5] Chargement du pipeline depuis {SOURCE_PIPELINE}...")
    pipeline = joblib.load(SOURCE_PIPELINE)
    print(f"      OK. Type : {type(pipeline).__name__}")
    print(f"      Steps   : {list(pipeline.named_steps.keys())}")
    return pipeline


def extract_and_save_preprocessor(pipeline) -> object:
    """Extrait le ColumnTransformer du pipeline et le sérialise."""
    print("[2/5] Extraction du preprocessor...")
    preprocessor = pipeline.named_steps["preprocessor"]
    print(f"      Type : {type(preprocessor).__name__}")

    TARGET_PREPROCESSOR.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, TARGET_PREPROCESSOR)
    size_kb = TARGET_PREPROCESSOR.stat().st_size / 1024
    print(f"      Sauvegardé : {TARGET_PREPROCESSOR} ({size_kb:.1f} KB)")
    return preprocessor


def measure_input_shape(preprocessor) -> int:
    """Mesure la dimensionnalité de sortie du preprocessor.

    Le XGBClassifier reçoit la matrice transformée par le ColumnTransformer.
    On charge un échantillon brut, on le passe dans le preprocessor, et on
    récupère le nombre de colonnes du résultat. C'est le N qui sera la shape
    d'entrée du modèle ONNX.
    """
    print("[3/5] Mesure de la shape d'entrée du XGBClassifier...")
    if not SAMPLE_PARQUET.exists():
        print(f"[ERREUR] Échantillon introuvable : {SAMPLE_PARQUET}")
        sys.exit(1)

    df = pd.read_parquet(SAMPLE_PARQUET).head(1)
    cols_to_drop = [c for c in COLUMNS_TO_EXCLUDE if c in df.columns]
    df_features = df.drop(columns=cols_to_drop)
    print(f"      Features brutes : {df_features.shape[1]} colonnes")

    X_transformed = preprocessor.transform(df_features)
    n_features = X_transformed.shape[1]
    print(f"      Features transformées : {n_features} colonnes")
    return n_features


def convert_xgb_to_onnx(pipeline, n_features: int) -> bytes:
    """Convertit le XGBClassifier en ONNX."""
    print(f"[4/5] Conversion XGBClassifier -> ONNX (opset {TARGET_OPSET})...")
    xgb_classifier = pipeline.named_steps["classifier"]
    print(f"      Type : {type(xgb_classifier).__name__}")
    print(f"      n_estimators : {xgb_classifier.n_estimators}")
    print(f"      max_depth    : {xgb_classifier.max_depth}")

    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(
        xgb_classifier,
        initial_types=initial_types,
        target_opset=TARGET_OPSET,
        name=ONNX_MODEL_NAME,
    )
    print(f"      Conversion OK. Opset producteur : {onnx_model.opset_import[0].version}")
    return onnx_model


def save_onnx(onnx_model) -> None:
    """Sérialise le modèle ONNX sur disque."""
    print("[5/5] Sauvegarde du modèle ONNX...")
    TARGET_ONNX.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_ONNX.open("wb") as f:
        f.write(onnx_model.SerializeToString())
    size_kb = TARGET_ONNX.stat().st_size / 1024
    print(f"      Sauvegardé : {TARGET_ONNX} ({size_kb:.1f} KB)")


def main() -> int:
    print("=" * 70)
    print("Conversion XGBoost -> ONNX (étape 9, branche feature/onnx-conversion)")
    print("=" * 70)

    pipeline = load_pipeline()
    preprocessor = extract_and_save_preprocessor(pipeline)
    n_features = measure_input_shape(preprocessor)
    onnx_model = convert_xgb_to_onnx(pipeline, n_features)
    save_onnx(onnx_model)

    print()
    print("=" * 70)
    print("✅ Conversion terminée. Artefacts produits :")
    print(f"   - {TARGET_PREPROCESSOR}")
    print(f"   - {TARGET_ONNX}")
    print()
    print("   Validation : uv run --group onnx python scripts/validate_onnx.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
