"""Valide l'équivalence numérique entre le pipeline pickle et la version ONNX.

Conformément à Q5 du design-doc étape 9 :
- Tolérance numérique : 1e-5 sur les probabilités
- 100% des décisions binaires (threshold) doivent être identiques
- Échantillon : 100 lignes aléatoires du test set (random_state=42)

Cette validation est OFFLINE : elle ne touche pas à l'API. Elle compare
deux chaînes d'inférence à partir des mêmes données brutes :

  Chaîne A (référence) : pipeline.predict_proba(df)
  Chaîne B (cible)     : preprocessor.transform(df) -> onnx_session.run()

Usage :
  uv run --group onnx python scripts/validate_onnx.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=".*If you are loading a serialized model.*",
    category=UserWarning,
)

# --- Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PIPELINE = PROJECT_ROOT / "models" / "xgboost_champion.pkl"
SOURCE_PREPROCESSOR = PROJECT_ROOT / "models" / "preprocessor.pkl"
SOURCE_ONNX = PROJECT_ROOT / "models" / "xgboost_classifier.onnx"
SAMPLE_PARQUET = PROJECT_ROOT / "data" / "raw" / "df_test_final.parquet"
REPORT_DIR = PROJECT_ROOT / "reports"

COLUMNS_TO_EXCLUDE = ["SK_ID_CURR", "Unnamed: 0", "TARGET"]

# --- Configuration ---
N_SAMPLES = 100
RANDOM_STATE = 42
PROBA_TOLERANCE = 1e-5
DECISION_THRESHOLD = 0.33381930539322036  # Q5 du design-doc / Projet 6


def load_artifacts():
    """Charge les 3 artefacts nécessaires : pipeline source, preprocessor, ONNX."""
    print(f"[1/4] Chargement des artefacts...")
    for path in (SOURCE_PIPELINE, SOURCE_PREPROCESSOR, SOURCE_ONNX):
        if not path.exists():
            print(f"[ERREUR] Artefact introuvable : {path}")
            print(f"        Lance d'abord scripts/convert_to_onnx.py.")
            sys.exit(1)

    pipeline = joblib.load(SOURCE_PIPELINE)
    preprocessor = joblib.load(SOURCE_PREPROCESSOR)
    onnx_session = ort.InferenceSession(
        str(SOURCE_ONNX),
        providers=["CPUExecutionProvider"],
    )
    print(f"      Pipeline      : {type(pipeline).__name__}")
    print(f"      Preprocessor  : {type(preprocessor).__name__}")
    print(f"      ONNX session  : {onnx_session.get_providers()[0]}")
    print(f"      ONNX inputs   : {[i.name for i in onnx_session.get_inputs()]}")
    print(f"      ONNX outputs  : {[o.name for o in onnx_session.get_outputs()]}")
    return pipeline, preprocessor, onnx_session


def load_sample() -> pd.DataFrame:
    """Charge 100 lignes aléatoires du test set."""
    print(f"[2/4] Chargement de l'échantillon ({N_SAMPLES} lignes, seed={RANDOM_STATE})...")
    df = pd.read_parquet(SAMPLE_PARQUET)
    cols_to_drop = [c for c in COLUMNS_TO_EXCLUDE if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    sample = df.sample(n=N_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"      Sample shape : {sample.shape}")
    return sample


def predict_reference(pipeline, sample: pd.DataFrame) -> np.ndarray:
    """Chaîne A : pipeline complet (joblib) -> predict_proba."""
    print(f"[3/4] Prédictions de référence (pipeline pickle)...")
    proba = pipeline.predict_proba(sample)[:, 1]
    print(f"      Probabilités calculées : {len(proba)} valeurs")
    print(f"      Range : [{proba.min():.6f}, {proba.max():.6f}]")
    return proba


def predict_onnx(preprocessor, onnx_session, sample: pd.DataFrame) -> np.ndarray:
    """Chaîne B : preprocessor sklearn -> ONNX session."""
    print(f"[3/4] Prédictions ONNX (preprocessor sklearn + ONNX Runtime)...")

    # Étape 1 : transform via preprocessor sklearn
    X_transformed = preprocessor.transform(sample)
    # ColumnTransformer peut renvoyer sparse, on densifie pour ONNX
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = X_transformed.astype(np.float32)

    # Étape 2 : inférence ONNX
    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: X_transformed})

    # XGBoost ONNX renvoie typiquement [labels, probas]. On veut probas classe 1.
    # Cas 1 : output est une liste de dicts {0: p0, 1: p1}
    # Cas 2 : output est un array 2D [N, 2]
    probas_output = outputs[1]
    if isinstance(probas_output, list) and isinstance(probas_output[0], dict):
        # Format dict (label -> proba)
        proba = np.array([d[1] for d in probas_output], dtype=np.float64)
    else:
        # Format array
        proba = np.asarray(probas_output)[:, 1].astype(np.float64)

    print(f"      Probabilités calculées : {len(proba)} valeurs")
    print(f"      Range : [{proba.min():.6f}, {proba.max():.6f}]")
    return proba


def compare(proba_ref: np.ndarray, proba_onnx: np.ndarray) -> dict:
    """Compare les deux séries de probabilités et de décisions."""
    print(f"[4/4] Comparaison référence vs ONNX...")

    # Différences absolues sur les probas
    diff = np.abs(proba_ref - proba_onnx)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    n_within_tolerance = int((diff <= PROBA_TOLERANCE).sum())

    # Décisions binaires
    decisions_ref = (proba_ref >= DECISION_THRESHOLD).astype(int)
    decisions_onnx = (proba_onnx >= DECISION_THRESHOLD).astype(int)
    n_decisions_identical = int((decisions_ref == decisions_onnx).sum())
    decision_accuracy = n_decisions_identical / len(decisions_ref)

    proba_ok = max_diff <= PROBA_TOLERANCE
    decision_ok = decision_accuracy == 1.0

    results = {
        "n_samples": len(proba_ref),
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "n_within_tolerance": n_within_tolerance,
        "tolerance": PROBA_TOLERANCE,
        "proba_validation": "PASS" if proba_ok else "FAIL",
        "n_decisions_identical": n_decisions_identical,
        "decision_accuracy": decision_accuracy,
        "decision_validation": "PASS" if decision_ok else "FAIL",
        "overall_validation": "PASS" if (proba_ok and decision_ok) else "FAIL",
    }
    return results


def print_results(results: dict) -> None:
    """Affiche les résultats sur stdout."""
    print()
    print("=" * 70)
    print("RÉSULTATS DE VALIDATION")
    print("=" * 70)
    print(f"  Échantillon              : {results['n_samples']} prédictions")
    print()
    print(f"  --- Probabilités (tolérance {results['tolerance']:.0e}) ---")
    print(f"  Différence max           : {results['max_diff']:.2e}")
    print(f"  Différence moyenne       : {results['mean_diff']:.2e}")
    print(f"  Valeurs dans tolérance   : {results['n_within_tolerance']}/{results['n_samples']}")
    print(f"  Validation probas        : {results['proba_validation']}")
    print()
    print(f"  --- Décisions binaires (threshold {DECISION_THRESHOLD:.6f}) ---")
    print(f"  Décisions identiques     : {results['n_decisions_identical']}/{results['n_samples']}")
    print(f"  Précision décisions      : {results['decision_accuracy']:.4%}")
    print(f"  Validation décisions     : {results['decision_validation']}")
    print()
    print(f"  ===> VALIDATION GLOBALE  : {results['overall_validation']}")
    print("=" * 70)


def save_report(results: dict) -> Path:
    """Sauvegarde un rapport texte dans reports/."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"validation_onnx_{timestamp}.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Validation ONNX vs pickle - {timestamp}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Échantillon       : {results['n_samples']} prédictions\n")
        f.write(f"Source            : {SAMPLE_PARQUET.name} (random_state={RANDOM_STATE})\n")
        f.write(f"Tolérance probas  : {results['tolerance']:.0e}\n")
        f.write(f"Threshold décision: {DECISION_THRESHOLD}\n\n")
        f.write(f"Différence max    : {results['max_diff']:.2e}\n")
        f.write(f"Différence moyenne: {results['mean_diff']:.2e}\n")
        f.write(f"In tolerance      : {results['n_within_tolerance']}/{results['n_samples']}\n")
        f.write(f"Decisions match   : {results['n_decisions_identical']}/{results['n_samples']}\n")
        f.write(f"Decision accuracy : {results['decision_accuracy']:.4%}\n\n")
        f.write(f"Probabilités  : {results['proba_validation']}\n")
        f.write(f"Décisions     : {results['decision_validation']}\n")
        f.write(f"Global        : {results['overall_validation']}\n")
    return report_path


def main() -> int:
    print("=" * 70)
    print("Validation ONNX (étape 9, branche feature/onnx-conversion)")
    print("=" * 70)

    pipeline, preprocessor, onnx_session = load_artifacts()
    sample = load_sample()
    proba_ref = predict_reference(pipeline, sample)
    proba_onnx = predict_onnx(preprocessor, onnx_session, sample)
    results = compare(proba_ref, proba_onnx)
    print_results(results)

    report_path = save_report(results)
    print(f"\n📄 Rapport sauvegardé : {report_path}")

    return 0 if results["overall_validation"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())