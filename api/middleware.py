"""Middlewares FastAPI pour l'instrumentation transverse.

Ce module définit les middlewares ajoutés à l'application FastAPI pour
collecter des métriques transverses (latence) sans polluer la logique
métier des endpoints.

Le middleware pose un top de chrono sur `request.state` au début de chaque
requête. Le calcul effectif de la latence et sa persistance en base sont
de la responsabilité des endpoints qui le souhaitent (cf. handler /predict
qui logge la latence en base pour le monitoring de drift à l'étape
ultérieure).
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response


def register_latency_middleware(app: FastAPI) -> None:
    """Enregistre le middleware de mesure de la latence des requêtes.

    Le middleware pose `request.state.start_time` (valeur de
    `time.perf_counter()`) au début de chaque requête. Les endpoints qui
    souhaitent calculer la latence peuvent lire cette valeur juste avant
    leur `return` :

        elapsed_ms = (time.perf_counter() - request.state.start_time) * 1000

    L'horloge utilisée est `time.perf_counter()` : monotone et de haute
    résolution, non affectée par les changements d'heure système.

    Args:
        app: Instance FastAPI sur laquelle enregistrer le middleware.

    Notes:
        - Le middleware s'applique à TOUTES les routes (/, /health, /predict).
        - C'est aux endpoints de décider s'ils lisent et persistent la latence.
        - La latence mesurée par cette stratégie exclut le temps de
          sérialisation de la réponse côté FastAPI (qui se passe après le
          `return` du handler) ; ce delta est négligeable (~quelques
          centaines de microsecondes).
    """

    @app.middleware("http")
    async def measure_latency(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.start_time = time.perf_counter()
        response = await call_next(request)
        return response
