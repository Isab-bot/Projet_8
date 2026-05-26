"""Génère un payload JSON valide pour les tests et le profiling de l'API.

Extrait une ligne du fichier df_test_final.parquet, la convertit au format
attendu par le schéma Pydantic PredictionInput, et la sauvegarde en JSON.

Usage:
    uv run python scripts/generate_sample_payload.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "raw" / "df_test_final.parquet"
OUTPUT_PATH = REPO_ROOT / "tests" / "data" / "sample_payload.json"

# Colonnes à exclure du payload (identifiants techniques, pas de features)
EXCLUDED_COLUMNS = ["SK_ID_CURR", "Unnamed: 0", "TARGET"]

# Row index à utiliser (fixe pour reproductibilité)
ROW_INDEX = 0


def main() -> int:
    if not PARQUET_PATH.exists():
        print(f"ERROR: fichier {PARQUET_PATH} introuvable.", file=sys.stderr)
        return 1

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Parquet chargé : {len(df)} lignes, {len(df.columns)} colonnes")

    # Sélection de la ligne
    row = df.iloc[ROW_INDEX]

    # Conversion en dict en excluant les colonnes techniques
    payload = {
        col: row[col]
        for col in df.columns
        if col not in EXCLUDED_COLUMNS
    }

    # Nettoyage : convertir les NaN en None (JSON-compatible), les numpy types en Python natifs
    cleaned = {}
    for k, v in payload.items():
        if pd.isna(v):
            cleaned[k] = None
        elif hasattr(v, "item"):  # numpy scalar
            cleaned[k] = v.item()
        else:
            cleaned[k] = v

    # Renommage des colonnes <lambda> → _lambda (conforme schéma Pydantic)
    final = cleaned

    print(f"Payload prêt : {len(final)} features (attendu : 326)")

    # Sauvegarde
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"Sauvegardé : {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
