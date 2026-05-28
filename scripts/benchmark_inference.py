"""Benchmark de l'inférence isolée : joblib vs ONNX (hors API).

Compare les deux backends du Predictor sur le même input, sans FastAPI,
sans réseau, sans DB. Mesure trois granularités (décision B1 = gamma) :
  - predict() complet (ce que voit l'appelant)
  - preprocessing seul (commun aux deux backends)
  - inférence pure du classifier (la partie qui diffère)

Méthodologie (design-doc Q) : 20 warm-up + 500 mesures, répété sur 3 runs.
On reporte mean/p50/p95/p99/min/max/std + débit, et la moyenne inter-runs.

Usage :
  uv run --group onnx python scripts/benchmark_inference.py

Note : --group onnx n'est pas requis (onnxruntime est en deps principales),
mais ne gêne pas.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

from api.predictor import Predictor  # noqa: E402
from api.schemas import PredictionInput  # noqa: E402

PAYLOAD_PATH = PROJECT_ROOT / "tests" / "data" / "sample_payload.json"
REPORT_DIR = PROJECT_ROOT / "reports"

N_WARMUP = 20
N_MEASURED = 500
N_RUNS = 10


def load_input() -> PredictionInput:
    """Charge le payload de test et le valide en PredictionInput."""
    with PAYLOAD_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return PredictionInput.model_validate(payload)


def percentile(values: list[float], p: float) -> float:
    """Percentile via numpy."""
    return float(np.percentile(values, p))


def compute_stats(timings_ms: list[float]) -> dict:
    """Calcule les statistiques d'une série de mesures (en ms)."""
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


def bench_predict_full(predictor: Predictor, input_data: PredictionInput) -> list[float]:
    """Mesure predict() complet sur N_MEASURED appels (ms par appel)."""
    # Warm-up
    for _ in range(N_WARMUP):
        predictor.predict(input_data)
    # Mesures
    timings = []
    for _ in range(N_MEASURED):
        t0 = time.perf_counter()
        predictor.predict(input_data)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def bench_decomposed(predictor: Predictor, input_data: PredictionInput) -> tuple[list[float], list[float]]:
    """Décompose predict() en preprocessing vs inférence pure.

    Retourne (timings_preprocessing_ms, timings_inference_ms).

    Le preprocessor est résolu UNE SEULE FOIS hors boucle (via named_steps en
    joblib, via l'attribut en onnx) pour que la mesure du preprocessing soit
    strictement comparable entre les deux backends : dans les deux cas on
    chronomètre le même appel `preproc.transform(df)`, sans l'indirection
    named_steps dans la boucle. Seule l'inférence diffère réellement.
    """
    import numpy as _np

    df = predictor._build_dataframe(input_data)

    # Résolution du preprocessor (une seule fois, hors mesure).
    if predictor.backend == "onnx":
        preproc = predictor.preprocessor
    else:
        preproc = predictor.pipeline.named_steps["preprocessor"]

    # Warm-up
    for _ in range(N_WARMUP):
        x = preproc.transform(df)
        if hasattr(x, "toarray"):
            x = x.toarray()
        if predictor.backend == "onnx":
            x32 = x.astype(_np.float32)
            name = predictor.onnx_session.get_inputs()[0].name
            predictor.onnx_session.run(None, {name: x32})
        else:
            predictor.pipeline.named_steps["classifier"].predict_proba(x)

    timings_pre = []
    timings_inf = []

    for _ in range(N_MEASURED):
        # --- Preprocessing (identique dans les deux modes : preproc.transform) ---
        t0 = time.perf_counter()
        x = preproc.transform(df)
        if hasattr(x, "toarray"):
            x = x.toarray()
        timings_pre.append((time.perf_counter() - t0) * 1000.0)

        # --- Inférence pure (la partie qui diffère) ---
        if predictor.backend == "onnx":
            x32 = x.astype(_np.float32)
            name = predictor.onnx_session.get_inputs()[0].name
            t1 = time.perf_counter()
            predictor.onnx_session.run(None, {name: x32})
            timings_inf.append((time.perf_counter() - t1) * 1000.0)
        else:
            classifier = predictor.pipeline.named_steps["classifier"]
            t1 = time.perf_counter()
            classifier.predict_proba(x)
            timings_inf.append((time.perf_counter() - t1) * 1000.0)

    return timings_pre, timings_inf


def run_once(input_data: PredictionInput, run_idx: int) -> dict:
    """Une run complète : joblib et onnx, predict complet + décomposé."""
    print(f"\n--- Run {run_idx + 1}/{N_RUNS} ---")
    results = {}

    for backend in ("joblib", "onnx"):
        print(f"  [{backend}] chargement...")
        predictor = Predictor(backend=backend)

        print(f"  [{backend}] predict() complet ({N_MEASURED} mesures)...")
        full = bench_predict_full(predictor, input_data)

        print(f"  [{backend}] décomposition preprocessing/inférence...")
        pre, inf = bench_decomposed(predictor, input_data)

        results[backend] = {
            "predict_full": compute_stats(full),
            "preprocessing": compute_stats(pre),
            "inference": compute_stats(inf),
        }
        print(f"  [{backend}] predict() mean = {results[backend]['predict_full']['mean']:.3f} ms")

    return results


def aggregate_runs(all_runs: list[dict]) -> dict:
    """Moyenne les means de chaque run pour chaque backend/granularité."""
    agg = {}
    for backend in ("joblib", "onnx"):
        agg[backend] = {}
        for granularity in ("predict_full", "preprocessing", "inference"):
            means = [run[backend][granularity]["mean"] for run in all_runs]
            agg[backend][granularity] = {
                "mean_of_means": statistics.mean(means),
                "std_of_means": statistics.stdev(means) if len(means) > 1 else 0.0,
                "per_run_means": means,
            }
    return agg


def print_summary(agg: dict) -> None:
    """Affiche le tableau comparatif final."""
    print("\n" + "=" * 70)
    print(f"RÉSUMÉ — Inférence isolée (moyenne sur {N_RUNS} runs, ms)")
    print("=" * 70)

    for granularity, label in [
        ("predict_full", "predict() complet"),
        ("preprocessing", "Preprocessing seul"),
        ("inference", "Inférence pure (classifier)"),
    ]:
        j = agg["joblib"][granularity]["mean_of_means"]
        o = agg["onnx"][granularity]["mean_of_means"]
        gain_pct = (j - o) / j * 100 if j > 0 else 0.0
        speedup = j / o if o > 0 else 0.0
        print(f"\n  {label}")
        print(f"    joblib : {j:.4f} ms")
        print(f"    onnx   : {o:.4f} ms")
        print(f"    gain   : {gain_pct:+.1f}%  (speedup {speedup:.2f}x)")
    print("\n" + "=" * 70)


def main() -> int:
    print("=" * 70)
    print("Benchmark inférence isolée (étape 9, branche feature/benchmark-comparison)")
    print(f"Config : {N_WARMUP} warm-up + {N_MEASURED} mesures x {N_RUNS} runs")
    print("=" * 70)

    input_data = load_input()
    all_runs = [run_once(input_data, i) for i in range(N_RUNS)]
    agg = aggregate_runs(all_runs)
    print_summary(agg)

    # Sauvegarde JSON
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"benchmark_inference_{timestamp}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {"n_warmup": N_WARMUP, "n_measured": N_MEASURED, "n_runs": N_RUNS},
                "per_run": all_runs,
                "aggregated": agg,
            },
            f,
            indent=2,
        )
    print(f"\n📄 Rapport sauvegardé : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
