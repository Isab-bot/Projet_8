"""Point d'entrée de l'API FastAPI.

Définit l'application, le cycle de vie (chargement du modèle au démarrage)
et les endpoints exposés (/health, /predict).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status

from api.config import API_DESCRIPTION, API_TITLE, API_VERSION
from api.predictor import ModelNotLoadedError, Predictor
from api.schemas import HealthResponse, PredictionInput, PredictionOutput


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


# --- Dépendances -----------------------------------------------------------

def get_predictor(request: Request) -> Predictor:
    """Récupère l'instance unique du Predictor depuis app.state.

    Cette dépendance peut être surchargée dans les tests pour injecter
    un mock ou un Predictor configuré sur un modèle factice.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle n'est pas chargé. Réessayez dans quelques instants.",
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
)
async def predict(
    input_data: PredictionInput,
    predictor: Predictor = Depends(get_predictor),
) -> PredictionOutput:
    """Calcule la probabilité de défaut et la décision pour un dossier crédit.

    L'input doit contenir les 326 features attendues par le pipeline.
    Les features avec lambda dans leur nom (ex: `BUREAU_CREDIT_ACTIVE_<lambda>`)
    doivent être envoyées avec leur nom d'origine.
    """
    try:
        return predictor.predict(input_data)
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Filet de sécurité : toute autre erreur côté pipeline -> 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {exc}",
        ) from exc