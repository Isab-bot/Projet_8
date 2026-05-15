"""Génère un payload JSON pour tester POST /predict.

Lit la première ligne de reference_data.parquet, en extrait les 326 features
avec leurs noms d'origine (alias parquet, ex. <lambda>) et écrit un fichier
data/sample_request.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

PARQUET = ROOT / "data" / "reference_data.parquet"
OUTPUT = ROOT / "data" / "sample_request.json"
EXCLUDED = {"Unnamed: 0", "SK_ID_CURR", "TARGET", "prediction_proba", "prediction"}


def main() -> None:
    df = pd.read_parquet(PARQUET)
    row = df.iloc[0]
    payload = {}
    for col in df.columns:
        if col in EXCLUDED:
            continue
        val = row[col]
        if pd.isna(val):
            payload[col] = None
        elif hasattr(val, "item"):
            payload[col] = val.item()
        else:
            payload[col] = val
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(payload)} features ecrites dans {OUTPUT}")


if __name__ == "__main__":
    main()
