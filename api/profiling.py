"""Module de profiling cProfile pour l'API FastAPI.

Branche un middleware qui enregistre les appels à la route /predict
dans un cProfile.Profile partagé, contrôlable via deux endpoints :
- POST /profile/start : démarre une session de profiling
- POST /profile/stop  : arrête, dump le .prof, retourne le chemin

Ces endpoints ne sont montés que si la variable d'environnement
ENABLE_PROFILING=true au démarrage de l'app. En production, le module
est neutre : aucun overhead, aucune surface d'attaque exposée.

Référence design-doc Projet 8 étape 9, Section 7.1 (Option B).
"""

from __future__ import annotations

import cProfile
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from api.config import PROFILE_OUTPUT_DIR

logger = logging.getLogger(__name__)


class ProfilingState:
    """Conteneur de l'état de profiling, attaché à app.state.profiling.

    Attributs:
        profiler: instance cProfile active, ou None si pas de session en cours.
        request_count: nombre de requêtes /predict profilées dans la session.
        lock: protège l'accès concurrent au profiler (cProfile n'est pas thread-safe).
        started_at: timestamp ISO du démarrage de la session.
    """

    def __init__(self) -> None:
        self.profiler: Optional[cProfile.Profile] = None
        self.request_count: int = 0
        self.lock = threading.Lock()
        self.started_at: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.profiler is not None


class PredictProfilingMiddleware(BaseHTTPMiddleware):
    """Middleware qui profile uniquement les requêtes vers /predict.

    Si une session de profiling est active (app.state.profiling.is_active),
    chaque appel à /predict est enregistré dans le profiler partagé.
    Les autres routes (/health, /docs, /profile/*) sont ignorées.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Filtre : on ne profile que /predict, et seulement si session active
        state: ProfilingState = request.app.state.profiling
        should_profile = request.url.path == "/predict" and state.is_active

        if not should_profile:
            return await call_next(request)

        # cProfile.enable/disable ne sont pas thread-safe ; on sérialise
        # via le lock. Acceptable ici : le bench tourne en single-worker.
        with state.lock:
            state.profiler.enable()
        try:
            response = await call_next(request)
        finally:
            with state.lock:
                state.profiler.disable()
                state.request_count += 1

        return response


def _build_profile_path() -> Path:
    """Construit un chemin .prof horodaté dans le dossier reports/."""
    PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROFILE_OUTPUT_DIR / f"predict_profile_{timestamp}.prof"


def register_profiling(app: FastAPI) -> None:
    """Monte conditionnellement le middleware et les endpoints de profiling.

    À appeler depuis api/main.py uniquement si config.ENABLE_PROFILING == True.
    En production (ENABLE_PROFILING=false), cette fonction n'est pas appelée
    et le module est totalement neutre.
    """
    logger.warning(
        "Profiling ACTIVÉ. Les endpoints /profile/start et /profile/stop "
        "sont exposés. Ne PAS activer en production."
    )

    # État partagé attaché à l'app
    app.state.profiling = ProfilingState()

    # Middleware ASGI
    app.add_middleware(PredictProfilingMiddleware)

    @app.post(
        "/profile/start",
        tags=["profiling"],
        summary="Démarre une session de profiling cProfile sur /predict",
    )
    async def profile_start(request: Request) -> dict:
        state: ProfilingState = request.app.state.profiling
        with state.lock:
            if state.is_active:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Une session de profiling est déjà active. "
                           "Appeler /profile/stop avant d'en démarrer une nouvelle.",
                )
            state.profiler = cProfile.Profile()
            state.request_count = 0
            state.started_at = datetime.now().isoformat()
        logger.info("Profiling démarré à %s", state.started_at)
        return {
            "status": "started",
            "started_at": state.started_at,
        }

    @app.post(
        "/profile/stop",
        tags=["profiling"],
        summary="Arrête la session, dump le .prof, retourne le chemin",
    )
    async def profile_stop(request: Request) -> dict:
        state: ProfilingState = request.app.state.profiling
        with state.lock:
            if not state.is_active:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Aucune session de profiling active. "
                           "Appeler /profile/start d'abord.",
                )
            # Snapshot avant reset
            profiler = state.profiler
            request_count = state.request_count
            started_at = state.started_at

            # Dump
            output_path = _build_profile_path()
            profiler.dump_stats(str(output_path))

            # Reset
            state.profiler = None
            state.request_count = 0
            state.started_at = None

        logger.info(
            "Profiling arrêté. %d requêtes profilées. Dump: %s",
            request_count, output_path,
        )
        return {
            "status": "stopped",
            "started_at": started_at,
            "request_count": request_count,
            "output_path": str(output_path),
        }

    @app.get(
        "/profile/status",
        tags=["profiling"],
        summary="État courant de la session de profiling",
    )
    async def profile_status(request: Request) -> dict:
        state: ProfilingState = request.app.state.profiling
        return {
            "active": state.is_active,
            "started_at": state.started_at,
            "request_count": state.request_count,
        }
