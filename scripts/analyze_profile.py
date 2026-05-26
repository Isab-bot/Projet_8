"""Analyse un fichier .prof produit par cProfile et génère un rapport .txt.

Usage :
  uv run python scripts/analyze_profile.py reports/predict_profile_YYYYMMDD_HHMMSS.prof

Produit un .txt avec :
  - Top 30 fonctions par temps cumulé (cumulative)
  - Top 30 fonctions par temps interne (tottime)
  - Stats globales (nombre d'appels, durée totale)
"""

from __future__ import annotations

import io
import pstats
import sys
from pathlib import Path


def analyze(prof_path: Path) -> Path:
    if not prof_path.exists():
        print(f"[ERREUR] Fichier introuvable : {prof_path}")
        sys.exit(1)

    output_path = prof_path.with_suffix(".txt")

    with output_path.open("w", encoding="utf-8") as f:
        # Section 1 : Top par cumulative time (où le temps EST passé, en remontant la pile)
        f.write("=" * 80 + "\n")
        f.write("TOP 30 — TEMPS CUMULÉ (cumulative)\n")
        f.write("Indique où le programme passe le plus de temps en remontant la pile.\n")
        f.write("=" * 80 + "\n\n")

        buf = io.StringIO()
        stats = pstats.Stats(str(prof_path), stream=buf)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        f.write(buf.getvalue())
        f.write("\n\n")

        # Section 2 : Top par tottime (où le temps est passé HORS appels imbriqués)
        f.write("=" * 80 + "\n")
        f.write("TOP 30 — TEMPS INTERNE (tottime)\n")
        f.write("Indique les fonctions qui consomment du temps elles-mêmes (hors sous-appels).\n")
        f.write("Ce sont les vrais hotspots à optimiser.\n")
        f.write("=" * 80 + "\n\n")

        buf = io.StringIO()
        stats = pstats.Stats(str(prof_path), stream=buf)
        stats.strip_dirs().sort_stats("tottime").print_stats(30)
        f.write(buf.getvalue())
        f.write("\n\n")

        # Section 3 : Focus prédiction — filtre sur predictor et xgboost
        f.write("=" * 80 + "\n")
        f.write("FOCUS — Fonctions liées à predictor / xgboost / sklearn\n")
        f.write("=" * 80 + "\n\n")

        buf = io.StringIO()
        stats = pstats.Stats(str(prof_path), stream=buf)
        stats.strip_dirs().sort_stats("cumulative").print_stats(
            "predictor|xgboost|sklearn|pandas", 30
        )
        f.write(buf.getvalue())

    return output_path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage : uv run python scripts/analyze_profile.py <chemin.prof>")
        return 1

    prof_path = Path(sys.argv[1])
    output_path = analyze(prof_path)
    print(f"✅ Rapport généré : {output_path}")
    print(f"   Visualisation interactive : uv run snakeviz {prof_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
