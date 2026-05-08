"""Sanity check de bout en bout du service log_prediction.

Test : prend la première ligne de reference_data.parquet, exécute la chaîne
complète (Pydantic -> Predictor -> log_prediction -> DB), puis relit la
ligne pour vérifier la cohérence.

Usage :
    uv run python scripts/sanity_check_db_service.py

Présuppose :
    - DATABASE_URL pointe vers une DB où la table predictions existe
      (alembic upgrade head appliqué).
    - models/xgboost_champion.pkl est présent.
    - data/reference_data.parquet est présent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from api.config import API_VERSION, DATABASE_URL, DECISION_THRESHOLD  # noqa: E402
from api.database import SessionLocal  # noqa: E402
from api.db_service import log_prediction  # noqa: E402
from api.models import Prediction  # noqa: E402
from api.predictor import Predictor  # noqa: E402
from api.schemas import PredictionInput  # noqa: E402

PARQUET_PATH = ROOT / "data" / "reference_data.parquet"
EXCLUDED = {"Unnamed: 0", "SK_ID_CURR", "TARGET", "prediction_proba", "prediction"}


def main() -> None:
    print(f"DATABASE_URL : {DATABASE_URL}\n")

    # --- [1/4] Lecture première ligne du parquet ---
    print("[1/4] Lecture premiere ligne du parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    row = df.iloc[0]
    sk_id = int(row["SK_ID_CURR"])
    expected_proba = float(row["prediction_proba"])
    expected_decision = int(row["prediction"])
    print(f"      OK: SK_ID_CURR={sk_id}, "
          f"proba_attendue={expected_proba:.10f}, "
          f"decision_attendue={expected_decision}\n")

    # --- [2/4] Construction PredictionInput depuis le dict ---
    print("[2/4] Construction PredictionInput (validation Pydantic)...")
    features_dict = {
        col: row[col] for col in df.columns if col not in EXCLUDED
    }
    # Pydantic v2 : on passe par by_alias pour matcher les "<lambda>"
    input_data = PredictionInput.model_validate(features_dict, by_alias=True)
    print(f"      OK: {len(features_dict)} features chargees, validation Pydantic OK\n")

    # --- [3/4] Appel Predictor.predict() ---
    print("[3/4] Appel Predictor.predict()...")
    predictor = Predictor()  # charge le pipeline pickle
    output = predictor.predict(input_data)
    print(f"      OK: proba={output.probability:.10f}, "
          f"decision={output.decision}")

    # Sanity : on doit retrouver la proba du parquet (cf. fin etape 3)
    proba_match = abs(output.probability - expected_proba) < 1e-6
    decision_match = output.decision == expected_decision
    print(f"      Proba match parquet     : {proba_match}")
    print(f"      Decision match parquet  : {decision_match}\n")
    if not (proba_match and decision_match):
        raise AssertionError(
            "Mismatch entre Predictor et parquet. "
            "Verifier le pipeline ou les features d'entree."
        )

    # --- [4/4] log_prediction() + relecture ---
    print("[4/4] Appel log_prediction() puis relecture en base...")
    request_id = str(uuid4())
    db = SessionLocal()
    try:
        log_prediction(
            db=db,
            request_id=request_id,
            input_data=input_data,
            proba=output.probability,
            decision=output.decision,
            model_version=API_VERSION,
            threshold=output.threshold,
        )
        # Relecture de la ligne qu'on vient d'écrire
        retrieved = db.query(Prediction).filter_by(request_id=request_id).one()
        print(f"      OK: ligne inseree avec id={retrieved.id}")
        print(f"          request_id    = {retrieved.request_id}")
        print(f"          timestamp     = {retrieved.timestamp}")
        print(f"          model_version = {retrieved.model_version}")
        print(f"          threshold     = {retrieved.threshold}")
        print(f"          proba         = {retrieved.prediction_proba:.10f}")
        print(f"          decision      = {retrieved.prediction}")

        # Verifier qu'une feature est correctement persistee
        sample_feature = "EXT_SOURCE_2"
        if sample_feature in df.columns:
            db_value = getattr(retrieved, sample_feature)
            parquet_value = float(row[sample_feature])
            match = abs((db_value or 0) - (parquet_value or 0)) < 1e-9
            print(f"          {sample_feature:14s}= {db_value} "
                  f"(parquet={parquet_value}, match={match})")
    finally:
        db.close()

    print("\n[OK] Sanity check de bout en bout : SUCCES")


if __name__ == "__main__":
    main()
