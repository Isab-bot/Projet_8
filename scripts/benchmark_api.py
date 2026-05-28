"""Benchmark de latence end-to-end de l'API /predict (un backend à la fois).

Mesure la latence client de N requêtes POST /predict contre une API en cours
d'exécution. À lancer DEUX fois : une fois l'API en mode joblib, une fois en
mode onnx, avec un label différent. Les deux JSON produits sont ensuite
comparés par scripts/compare_benchmarks.py.

Méthodologie (design-doc) : 20 warm-up + 500 mesures, ENABLE_PROFILING=false,
persistance DB active (mesure réaliste de la prod).

Usage :
  # 1. Lancer l'API en mode joblib sur le port 8003, puis :
  uv run python scripts/benchmark_api.py --url http://127.0.0.1:8003 --label joblib

  # 2. Relancer l'API en mode onnx sur le port 8003, puis :
  uv run python scripts/benchmark_api.py --url http://127.0.0.1:8003 --label onnx
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_PATH = PROJECT_ROOT / "tests" / "data" / "sample_payload.json"
REPORT_DIR = PROJECT_ROOT / "reports"

N_WARMUP = 20
N_MEASURED = 500
TIMEOUT_S = 30.0


def load_payload() -> dict:
    with PAYLOAD_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def compute_stats(timings_ms: list[float]) -> dict:
    return {
        "mean": statistics.mean(timings_ms),
        "p50": percentile(timings_ms, 50),
        "p95": percentile(timings_ms, 95),
        "p99": percentile(timings_ms, 99),
        "min": min(timings_ms),
        "max": max(timings_ms),
        "std": statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "throughput_per_s": 1000.0 / statistics.mean(timings_ms),
    }


def check_api(url: str) -> None:
    try:
        r = requests.get(f"{url}/health", timeout=5)
        r.raise_for_status()
        body = r.json()
        print(f"  API OK : status={body.get('status')}, model_loaded={body.get('model_loaded')}")
    except requests.RequestException as exc:
        print(f"[ERREUR] API non joignable sur {url}: {exc}")
        sys.exit(1)


def run_benchmark(url: str, payload: dict) -> list[float]:
    predict_url = f"{url}/predict"

    print(f"  Warm-up ({N_WARMUP} requêtes)...")
    for _ in range(N_WARMUP):
        r = requests.post(predict_url, json=payload, timeout=TIMEOUT_S)
        r.raise_for_status()

    print(f"  Mesures ({N_MEASURED} requêtes)...")
    timings = []
    for i in range(N_MEASURED):
        t0 = time.perf_counter()
        r = requests.post(predict_url, json=payload, timeout=TIMEOUT_S)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        timings.append(elapsed_ms)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{N_MEASURED}")
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark API /predict")
    parser.add_argument("--url", required=True, help="Base URL de l'API (ex: http://127.0.0.1:8003)")
    parser.add_argument("--label", required=True, help="Label du backend (ex: joblib, onnx)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Benchmark API /predict — backend '{args.label}'")
    print(f"Config : {N_WARMUP} warm-up + {N_MEASURED} mesures")
    print(f"URL : {args.url}")
    print("=" * 70)

    check_api(args.url)
    payload = load_payload()

    t_start = time.perf_counter()
    timings = run_benchmark(args.url, payload)
    total_s = time.perf_counter() - t_start

    stats = compute_stats(timings)

    print("\n" + "-" * 70)
    print(f"Résultats — backend '{args.label}'")
    print("-" * 70)
    print(f"  Requêtes    : {N_MEASURED}")
    print(f"  Durée totale: {total_s:.1f}s")
    print(f"  Mean        : {stats['mean']:.2f} ms")
    print(f"  p50         : {stats['p50']:.2f} ms")
    print(f"  p95         : {stats['p95']:.2f} ms")
    print(f"  p99         : {stats['p99']:.2f} ms")
    print(f"  Min / Max   : {stats['min']:.2f} / {stats['max']:.2f} ms")
    print(f"  Débit       : {stats['throughput_per_s']:.1f} req/s")
    print("-" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"benchmark_api_{args.label}_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "label": args.label,
                "url": args.url,
                "config": {"n_warmup": N_WARMUP, "n_measured": N_MEASURED},
                "total_seconds": total_s,
                "stats": stats,
                "raw_timings_ms": timings,
            },
            f,
            indent=2,
        )
    print(f"\n📄 Rapport sauvegardé : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
