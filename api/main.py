"""Point d'entrée de l'API FastAPI.

Définit l'application, le cycle de vie (chargement du modèle au démarrage)
et les endpoints exposés (/health, /predict).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.config import API_DESCRIPTION, API_TITLE, API_VERSION
from api.database import get_db
from api.db_service import log_prediction
from api.exceptions import (
    InternalErrorResponse,
    ServiceUnavailableResponse,
    ValidationErrorResponse,
    register_exception_handlers,
)
from api.predictor import ModelNotLoadedError, Predictor
from api.schemas import HealthResponse, PredictionInput, PredictionOutput

logger = logging.getLogger(__name__)


# --- Cycle de vie de l'application -----------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge le modèle au démarrage et libère les ressources à l'arrêt.

    Le Predictor est instancié une seule fois et stocké dans app.state pour
    être réutilisé par toutes les requêtes (point de vigilance MLOps :
    pas de chargement par requête).
    """
    # STARTUP
    app.state.predictor = Predictor()
    yield
    # SHUTDOWN
    app.state.predictor = None


# --- Application FastAPI ---------------------------------------------------

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

# Enregistrement des handlers d'exceptions custom
register_exception_handlers(app)


# --- Dépendances -----------------------------------------------------------

def get_predictor(request: Request) -> Predictor:
    """Récupère l'instance unique du Predictor depuis app.state.

    Cette dépendance peut être surchargée dans les tests pour injecter
    un mock ou un Predictor configuré sur un modèle factice.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise ModelNotLoadedError(
            "Le modèle n'est pas chargé. Réessayez dans quelques instants."
        )
    return predictor


# --- Endpoints -------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> dict:
    """Page d'accueil minimaliste — redirige implicitement vers /docs."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifie l'état de l'API et du modèle",
    tags=["monitoring"],
)
async def health(request: Request) -> HealthResponse:
    """Retourne le statut de l'API et indique si le modèle est chargé."""
    model_loaded = getattr(request.app.state, "predictor", None) is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        api_version=API_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Prédit la probabilité de défaut de crédit",
    tags=["prediction"],
    responses={
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ValidationErrorResponse,
            "description": "Erreur de validation des features d'entrée.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ServiceUnavailableResponse,
            "description": "Modèle non chargé.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": InternalErrorResponse,
            "description": "Erreur interne du serveur.",
        },
    },
)
def predict(
    input_data: PredictionInput,
    predictor: Predictor = Depends(get_predictor),
    db: Session = Depends(get_db),
) -> PredictionOutput:
    """Calcule la probabilité de défaut et la décision pour un dossier crédit.

    L'input doit contenir les 326 features attendues par le pipeline.
    Les features avec lambda dans leur nom (ex: `BUREAU_CREDIT_ACTIVE_<lambda>`)
    doivent être envoyées avec leur nom d'origine.

    Le résultat est persisté en base de données pour permettre le monitoring
    de drift à l'étape ultérieure (Evidently). En cas d'échec de persistance,
    la prédiction est tout de même retournée au client (politique fail-open :
    l'instrumentation ne doit pas casser le service observé).
    """
    request_id = str(uuid4())

    # Calcul de la prédiction (peut lever : 500 si modèle planté)
    output = predictor.predict(input_data)
    output.request_id = request_id

    # Persistance fail-open : on log l'erreur mais on retourne quand même la prédiction
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
    except SQLAlchemyError:
        logger.error(
            "Échec de persistance de la prédiction en base",
            exc_info=True,
            extra={"request_id": request_id},
        )

    return output