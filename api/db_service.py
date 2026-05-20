"""Service de persistance des prédictions en base de données.

Ce module expose la fonction log_prediction() utilisée par l'endpoint
POST /predict pour enregistrer chaque prédiction dans la table predictions.

La fonction commit silencieusement en cas de succès. En cas d'erreur DB,
elle laisse remonter l'exception SQLAlchemyError ; le caller (handler
FastAPI) est responsable de la politique fail-open (cf. décision 5 :
l'instrumentation ne doit pas casser le service observé).
"""
from sqlalchemy.orm import Session

from api.models import Prediction
from api.schemas import PredictionInput


def log_prediction(
    db: Session,
    request_id: str,
    input_data: PredictionInput,
    proba: float,
    decision: int,
    model_version: str,
    threshold: float,
    latency_ms: float,
    inference_ms: float,
) -> None:
    """Persiste une prédiction dans la table predictions.

    Args:
        db: Session SQLAlchemy ouverte (fournie par get_db).
        request_id: UUID de la requête (str de 36 caractères).
        input_data: PredictionInput Pydantic (326 features validées).
        proba: Probabilité prédite (sortie XGBoost classe 1).
        decision: Décision binaire 0 ou 1 après application du seuil.
        model_version: Version de l'API/modèle (ex. "0.1.0").
        threshold: Seuil F3 utilisé pour la décision.
        latency_ms: Latence totale de la requête mesurée côté handler
            (du middleware start_time jusqu'à juste avant l'écriture DB),
            en millisecondes.
        inference_ms: Temps d'exécution de model.predict_proba()
            uniquement, en millisecondes.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: en cas d'échec d'insertion.
            Le caller doit décider de la politique de gestion d'erreur.

    Notes:
        - Le timestamp est posé côté DB (server_default=now()), pas ici.
        - by_alias=False : on veut les noms Python (alignés avec models.py),
          pas les noms d'origine du parquet (qui contiennent "<lambda>").
        - latency_ms exclut le temps d'écriture DB (la persistance se fait
          après le calcul de latency_ms côté handler).
    """
    features_dict = input_data.model_dump(by_alias=False)

    prediction = Prediction(
        request_id=request_id,
        model_version=model_version,
        threshold=threshold,
        prediction_proba=proba,
        prediction=decision,
        latency_ms=latency_ms,
        inference_ms=inference_ms,
        **features_dict,
    )

    db.add(prediction)
    db.commit()
