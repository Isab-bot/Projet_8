"""Lance une session de profiling sur l'API en cours d'exécution.

Workflow :
  1. POST /profile/start
  2. Envoie N_WARMUP requêtes (non profilées si /profile/start est appelé après)
     → ici simplifié : on lance start AVANT, donc warm-up inclus.
     Variante propre : warm-up AVANT start. Implémentée ci-dessous.
  3. POST /profile/start
  4. Envoie N_MEASURED requêtes
  5. POST /profile/stop → récupère le chemin du .prof

Prérequis :
  - API lancée en local sur API_URL avec ENABLE_PROFILING=true
  - Payload de référence dans tests/data/sample_payload.json

Usage :
  uv run python scripts/profile_api.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8001"
PAYLOAD_PATH = Path("tests/data/sample_payload.json")
N_WARMUP = 20
N_MEASURED = 300  # 300 pour le profiling ; 500 sera utilisé pour le benchmark final
TIMEOUT_S = 30.0


def load_payload() -> dict:
    with PAYLOAD_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_api_alive() -> None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ERREUR] API non joignable sur {API_URL}: {exc}")
        sys.exit(1)


def warmup(payload: dict) -> None:
    print(f"[1/4] Warm-up : {N_WARMUP} requêtes (non profilées)...")
    for i in range(N_WARMUP):
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT_S)
        r.raise_for_status()
    print(f"      Warm-up terminé.")


def profile_start() -> None:
    print("[2/4] Démarrage du profiling...")
    r = requests.post(f"{API_URL}/profile/start", timeout=10)
    r.raise_for_status()
    print(f"      {r.json()}")


def measure(payload: dict) -> None:
    print(f"[3/4] Mesures : {N_MEASURED} requêtes profilées...")
    t0 = time.perf_counter()
    for i in range(N_MEASURED):
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT_S)
        r.raise_for_status()
        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{N_MEASURED} requêtes effectuées")
    elapsed = time.perf_counter() - t0
    print(f"      Terminé en {elapsed:.2f}s ({N_MEASURED / elapsed:.1f} req/s)")


def profile_stop() -> str:
    print("[4/4] Arrêt du profiling et dump...")
    r = requests.post(f"{API_URL}/profile/stop", timeout=10)
    r.raise_for_status()
    data = r.json()
    print(f"      {data}")
    return data["output_path"]


def main() -> int:
    if not PAYLOAD_PATH.exists():
        print(f"[ERREUR] Payload introuvable : {PAYLOAD_PATH}")
        return 1

    check_api_alive()
    payload = load_payload()

    warmup(payload)
    profile_start()
    try:
        measure(payload)
    finally:
        # Stop systématique même si une mesure échoue, sinon session orpheline
        output_path = profile_stop()

    print(f"\n✅ Profil sauvegardé : {output_path}")
    print(f"   Analyse : uv run python scripts/analyze_profile.py {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())