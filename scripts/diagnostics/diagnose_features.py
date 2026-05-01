"""Diagnostic : identifier les colonnes en trop entre le dataset et le modèle."""

import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_champion.pkl"
PARQUET_PATH = Path(
    r"C:\Users\renar\Documents\Alternance\Projet_6\MLOps_1\data\df_train_final.parquet"
)

model = joblib.load(MODEL_PATH)
df = pd.read_parquet(PARQUET_PATH)

print("Type du modèle :", type(model).__name__)
print("Steps du Pipeline :", [name for name, _ in model.steps])
print()

# feature_names_in_ existe sur les estimateurs sklearn fittés
expected = None
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    print(f"feature_names_in_ trouvé sur le Pipeline : {len(expected)} features")
else:
    first_step_name, first_step = model.steps[0]
    if hasattr(first_step, "feature_names_in_"):
        expected = list(first_step.feature_names_in_)
        print(f"feature_names_in_ sur step '{first_step_name}' : {len(expected)} features")
    else:
        print("feature_names_in_ introuvable sur le Pipeline et son premier step.")
        # Fallback : on inspecte tous les steps
        for name, step in model.steps:
            print(f"  Step '{name}' : {type(step).__name__}")
            if hasattr(step, "feature_names_in_"):
                print(f"    -> feature_names_in_ : {len(step.feature_names_in_)} features")

dataset_cols = [c for c in df.columns if c != "TARGET"]
print(f"\nColonnes dataset (hors TARGET) : {len(dataset_cols)}")

if expected is not None:
    extra = set(dataset_cols) - set(expected)
    missing = set(expected) - set(dataset_cols)
    print(f"\nColonnes EN TROP dans le dataset ({len(extra)}) :")
    for col in sorted(extra):
        print(f"  - {col}")
    print(f"\nColonnes MANQUANTES dans le dataset ({len(missing)}) :")
    for col in sorted(missing):
        print(f"  - {col}")