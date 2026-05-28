"""Met à jour models/model_metadata.json avec les infos d'optimisation ONNX.

Ajoute une section 'optimization' documentant la conversion ONNX de l'étape 9
sans modifier les champs existants (origine MLflow, métriques, seuil).

Usage : uv run python scripts/update_metadata.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def main() -> int:
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Ajout de la section optimization (étape 9). N'écrase pas l'existant.
    metadata["optimization"] = {
        "stage": "Projet_8_etape_9",
        "strategy": "ONNX conversion of XGBClassifier only (preprocessor stays sklearn)",
        "artifacts": {
            "preprocessor": "models/preprocessor.pkl",
            "onnx_model": "models/xgboost_classifier.onnx",
            "source_pipeline": "models/xgboost_champion.pkl",
        },
        "onnx": {
            "converter": "onnxmltools",
            "target_opset": 15,
            "n_features_raw": 326,
            "n_features_transformed": 336,
        },
        "validation": {
            "n_samples": 100,
            "random_state": 42,
            "proba_tolerance": 1e-5,
            "proba_max_diff": 2.68e-7,
            "decision_accuracy": 1.0,
            "result": "PASS",
        },
        "backends_available": ["joblib", "onnx"],
        "default_backend": "joblib",
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"✅ {METADATA_PATH} mis à jour avec la section 'optimization'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
