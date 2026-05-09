"""Gestionnaires d'erreurs custom pour l'API.

Centralise :
- Les exceptions métier (modèle non chargé, etc.)
- Le reformatage des erreurs de validation Pydantic en JSON lisible
- Les schémas de réponse d'erreur documentés dans Swagger

Les handlers définis ici sont enregistrés dans api/main.py via app.add_exception_handler().
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.predictor import ModelNotLoadedError


# --- Schémas de réponse d'erreur (documentés dans Swagger) -----------------

class ValidationErrorDetail(BaseModel):
    """Détail d'une erreur de validation pour un champ donné."""

    field: str = Field(..., description="Nom du champ en erreur (avec son chemin).")
    message: str = Field(..., description="Message d'erreur lisible.")
    type: str = Field(..., description="Type d'erreur Pydantic (ex: 'missing', 'float_parsing').")


class ValidationErrorResponse(BaseModel):
    """Réponse 422 : erreurs de validation Pydantic regroupées."""

    detail: str = Field(..., description="Message global de l'erreur.")
    errors: list[ValidationErrorDetail] = Field(
        ..., description="Liste détaillée des champs en erreur."
    )


class ServiceUnavailableResponse(BaseModel):
    """Réponse 503 : service temporairement indisponible (modèle non chargé)."""

    detail: str


class InternalErrorResponse(BaseModel):
    """Réponse 500 : erreur interne du serveur."""

    detail: str


# --- Handlers d'exceptions -------------------------------------------------

async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Reformate les erreurs Pydantic en réponse lisible.

    Au lieu de la liste brute renvoyée par FastAPI par défaut,
    on construit une structure {detail, errors[]} où chaque erreur est
    facile à lire pour le client (champ + message + type).
    """
    formatted_errors: list[dict] = []
    for err in exc.errors():
        # 'loc' est un tuple comme ('body', 'AMT_INCOME_TOTAL')
        # on retire le préfixe 'body' qui n'apporte rien au client
        loc = [str(part) for part in err["loc"] if part != "body"]
        field = ".".join(loc) if loc else "<root>"
        formatted_errors.append(
            {
                "field": field,
                "message": err["msg"],
                "type": err["type"],
            }
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": f"Erreur de validation : {len(formatted_errors)} champ(s) invalide(s).",
            "errors": formatted_errors,
        },
    )


async def model_not_loaded_handler(
    request: Request, exc: ModelNotLoadedError
) -> JSONResponse:
    """Mappe ModelNotLoadedError vers une réponse HTTP 503."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Filet de sécurité pour toute exception non prévue.

    Évite de leaker la stack trace au client tout en retournant un message
    informatif. Le détail technique reste dans les logs serveur.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Erreur interne du serveur : {type(exc).__name__}"},
    )


# --- Enregistrement des handlers -------------------------------------------

def register_exception_handlers(app: FastAPI) -> None:
    """Enregistre tous les handlers custom sur l'application FastAPI."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
    # Le handler générique est volontairement enregistré en dernier
    # (FastAPI essaie les handlers les plus spécifiques en premier)
    app.add_exception_handler(Exception, generic_exception_handler)