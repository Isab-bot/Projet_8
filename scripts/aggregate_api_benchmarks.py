"""Agrège plusieurs runs de benchmark API et compare joblib vs onnx.

Lit tous les fichiers reports/benchmark_api_<backend>_run*.json, regroupe par
backend (joblib / onnx), calcule la moyenne et l'écart-type inter-runs de
chaque métrique, et affiche un comparatif statistique.

Permet de conclure si la différence joblib/onnx est significative ou noyée
dans le bruit de mesure (chevauchement des intervalles mean ± std).

Usage :
  uv run python scripts/aggregate_api_benchmarks.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "reports"

METRICS = ["mean", "p50", "p95", "p99", "throughput_per_s"]


def collect(backend: str) -> list[dict]:
    """Charge les JSON des runs propres (_run1 à _run10) du backend donné.

    Ne prend QUE les fichiers labellisés _run* pour exclure le run unique
    initial fait dans des conditions différentes.
    """
    files = sorted(REPORT_DIR.glob(f"benchmark_api_{backend}_run*.json"))
    runs = []
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs


def aggregate_metric(runs: list[dict], metric: str) -> dict:
    """Moyenne et écart-type d'une métrique sur tous les runs."""
    values = [r["stats"][metric] for r in runs]
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
        "values": values,
    }


def main() -> int:
    joblib_runs = collect("joblib")
    onnx_runs = collect("onnx")

    if not joblib_runs or not onnx_runs:
        print(f"[ERREUR] JSON introuvables. joblib={len(joblib_runs)}, onnx={len(onnx_runs)}")
        print(f"        Cherché dans : {REPORT_DIR}")
        return 1

    print("=" * 78)
    print("AGRÉGATION BENCHMARK API — joblib vs onnx")
    print(f"Runs : joblib={len(joblib_runs)}, onnx={len(onnx_runs)} "
          f"(500 mesures chacun)")
    print("=" * 78)
    print(f"{'Métrique':<16}{'joblib (mean±std)':>22}{'onnx (mean±std)':>22}{'Gain':>14}")
    print("-" * 78)

    for metric in METRICS:
        j = aggregate_metric(joblib_runs, metric)
        o = aggregate_metric(onnx_runs, metric)

        if metric == "throughput_per_s":
            gain = (o["mean"] - j["mean"]) / j["mean"] * 100 if j["mean"] > 0 else 0.0
        else:
            gain = (j["mean"] - o["mean"]) / j["mean"] * 100 if j["mean"] > 0 else 0.0

        j_str = f"{j['mean']:.2f}±{j['std']:.2f}"
        o_str = f"{o['mean']:.2f}±{o['std']:.2f}"
        label = metric if metric != "throughput_per_s" else "throughput"
        print(f"{label:<16}{j_str:>22}{o_str:>22}{gain:>+12.1f}%")

    print("-" * 78)

    # Verdict statistique sur la moyenne : les intervalles mean±std se chevauchent-ils ?
    j_mean = aggregate_metric(joblib_runs, "mean")
    o_mean = aggregate_metric(onnx_runs, "mean")
    j_lo, j_hi = j_mean["mean"] - j_mean["std"], j_mean["mean"] + j_mean["std"]
    o_lo, o_hi = o_mean["mean"] - o_mean["std"], o_mean["mean"] + o_mean["std"]
    overlap = not (j_hi < o_lo or o_hi < j_lo)

    print()
    print("VERDICT (sur la latence moyenne) :")
    print(f"  joblib : {j_mean['mean']:.2f} ms  [{j_lo:.2f}, {j_hi:.2f}] (±1 std)")
    print(f"  onnx   : {o_mean['mean']:.2f} ms  [{o_lo:.2f}, {o_hi:.2f}] (±1 std)")
    if overlap:
        print("  -> Les intervalles ±1 std SE CHEVAUCHENT : la différence end-to-end")
        print("     n'est PAS statistiquement distinguable du bruit de mesure.")
    else:
        diff_pct = (j_mean["mean"] - o_mean["mean"]) / j_mean["mean"] * 100
        print("  -> Les intervalles NE se chevauchent PAS : différence mesurable")
        print(f"     ({diff_pct:+.1f}% en faveur de {'onnx' if diff_pct > 0 else 'joblib'}).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
