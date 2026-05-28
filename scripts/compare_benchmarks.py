"""Compare deux JSON de benchmark API (joblib vs onnx) et produit le comparatif.

Lit les deux fichiers reports/benchmark_api_{label}_*.json produits par
benchmark_api.py et affiche un tableau comparatif des latences.

Usage :
  uv run python scripts/compare_benchmarks.py <joblib.json> <onnx.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        print(f"[ERREUR] Fichier introuvable : {path}")
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage : uv run python scripts/compare_benchmarks.py <joblib.json> <onnx.json>")
        return 1

    a = load(sys.argv[1])
    b = load(sys.argv[2])

    sa, sb = a["stats"], b["stats"]
    la, lb = a["label"], b["label"]

    print("=" * 70)
    print(f"COMPARAISON LATENCE API — {la} vs {lb}")
    print("=" * 70)
    print(f"{'Métrique':<14}{la:>14}{lb:>14}{'Gain':>14}")
    print("-" * 70)

    for key, label in [
        ("mean", "Mean"),
        ("p50", "p50"),
        ("p95", "p95"),
        ("p99", "p99"),
        ("throughput_per_s", "Débit (req/s)"),
    ]:
        va, vb = sa[key], sb[key]
        if key == "throughput_per_s":
            gain = (vb - va) / va * 100 if va > 0 else 0.0
            print(f"{label:<14}{va:>14.1f}{vb:>14.1f}{gain:>+13.1f}%")
        else:
            gain = (va - vb) / va * 100 if va > 0 else 0.0
            print(f"{label:<14}{va:>13.2f}m{vb:>13.2f}m{gain:>+13.1f}%")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
