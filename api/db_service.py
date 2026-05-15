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

    Raises:
        sqlalchemy.exc.SQLAlchemyError: en cas d'échec d'insertion.
            Le caller doit décider de la politique de gestion d'erreur.

    Notes:
        - Le timestamp est posé côté DB (server_default=now()), pas ici.
        - by_alias=False : on veut les noms Python (alignés avec models.py),
          pas les noms d'origine du parquet (qui contiennent "<lambda>").
    """
    features_dict = input_data.model_dump(by_alias=False)

    prediction = Prediction(
        request_id=request_id,
        model_version=model_version,
        threshold=threshold,
        prediction_proba=proba,
        prediction=decision,
        **features_dict,
    )

    db.add(prediction)
    db.commit()
