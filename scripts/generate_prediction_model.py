"""Générateur du modèle ORM Prediction (script utilitaire one-shot).

Lit la liste des features depuis data/reference_data.parquet, applique le
mapping de noms (api.feature_naming.to_sql_column_name), et génère
api/models.py contenant la classe SQLAlchemy Prediction.

Ce script est conçu pour être ré-exécuté si jamais le modèle évolue
(ajout/suppression de features). En P8 le modèle est figé donc il devrait
être lancé une seule fois.

Usage :
    uv run python scripts/generate_prediction_model.py

Sortie :
    api/models.py (écrasé si existe)

Garde-fou :
    Le script échoue si le nombre de features détectées ne correspond pas
    à api.config.N_FEATURES (326). Cela évite de générer silencieusement
    un fichier incohérent si le parquet a été altéré.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ajout du répertoire racine au PYTHONPATH pour pouvoir importer api.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.config import N_FEATURES, PROJECT_ROOT  # noqa: E402
from api.feature_naming import to_sql_column_name  # noqa: E402

# --- Configuration ---
PARQUET_PATH = PROJECT_ROOT / "data" / "reference_data.parquet"
OUTPUT_PATH = PROJECT_ROOT / "api" / "models.py"

# Colonnes du parquet à exclure (métadonnées, label, sorties du modèle)
EXCLUDED_COLUMNS = {
    "Unnamed: 0",
    "SK_ID_CURR",
    "TARGET",
    "prediction_proba",
    "prediction",
}

# --- Template du fichier généré ---
HEADER = '''"""Modèle ORM SQLAlchemy 2.0 pour la table predictions.

⚠️ FICHIER AUTO-GÉNÉRÉ — NE PAS ÉDITER À LA MAIN.
Régénérer via : uv run python scripts/generate_prediction_model.py

Source des features : data/reference_data.parquet
Mapping des noms   : api.feature_naming.to_sql_column_name
"""
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base


class Prediction(Base):
    """Une prédiction enregistrée par l'endpoint POST /predict.

    Schéma à plat (une colonne par feature) pour permettre une lecture
    directe par Evidently à l'étape de monitoring (pas de transformation
    intermédiaire de JSON vers DataFrame).
    """

    __tablename__ = "predictions"

    # --- Identifiants ---
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(
        String(36), unique=True, index=True, nullable=False,
        comment="UUID de la requête, exposé au client pour traçabilité",
    )

    # --- Métadonnées ---
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
        server_default=func.now(),
        comment="Horodatage UTC de la prédiction",
    )
    model_version: Mapped[str] = mapped_column(
        String(32), index=True, nullable=False,
        comment="Version de l'API/modèle ayant produit la prédiction",
    )
    threshold: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Seuil de décision F3 utilisé (issu du Projet 6)",
    )

    # --- Sortie du modèle ---
    prediction_proba: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Probabilité prédite par XGBoost (classe 1 = défaut)",
    )
    prediction: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="Décision binaire (0 ou 1) après application du seuil",
    )

    # --- Features (auto-générées) ---
'''

FOOTER = '''
    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} request_id={self.request_id} "
            f"proba={self.prediction_proba:.4f} pred={self.prediction}>"
        )
'''


def main() -> None:
    print(f"Lecture du parquet : {PARQUET_PATH}")
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet introuvable : {PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH)
    all_columns = df.columns.tolist()
    print(f"  Total colonnes dans le parquet : {len(all_columns)}")

    # Filtrer les colonnes-métadonnées
    feature_columns = [c for c in all_columns if c not in EXCLUDED_COLUMNS]
    print(f"  Colonnes exclues : {sorted(EXCLUDED_COLUMNS)}")
    print(f"  Features candidates : {len(feature_columns)}")

    # Garde-fou : on attend exactement N_FEATURES (326)
    if len(feature_columns) != N_FEATURES:
        raise ValueError(
            f"Nombre de features incohérent : attendu {N_FEATURES}, "
            f"trouvé {len(feature_columns)}. Vérifier le parquet ou la blacklist."
        )

    # Appliquer le mapping de noms et vérifier l'unicité
    sql_names = [to_sql_column_name(c) for c in feature_columns]
    if len(set(sql_names)) != len(sql_names):
        # Détecter les doublons pour message d'erreur explicite
        seen, dupes = set(), []
        for n in sql_names:
            if n in seen:
                dupes.append(n)
            seen.add(n)
        raise ValueError(
            f"Collision de noms SQL après mapping : {dupes}. "
            "Réviser api/feature_naming.py."
        )

    # Construire la liste des champs Mapped[float | None]
    print(f"\nGénération de {len(feature_columns)} colonnes ORM...")
    field_lines: list[str] = []
    n_renamed = 0
    for original in feature_columns:
        sql_name = to_sql_column_name(original)
        if sql_name != original:
            n_renamed += 1
            # Annoter le renommage en commentaire pour traçabilité
            field_lines.append(
                f'    {sql_name}: Mapped[float | None] = mapped_column('
                f'Float, nullable=True)  # original: {original}'
            )
        else:
            field_lines.append(
                f'    {sql_name}: Mapped[float | None] = mapped_column('
                f'Float, nullable=True)'
            )
    print(f"  Renommées (lambda) : {n_renamed}")
    print(f"  Identiques         : {len(feature_columns) - n_renamed}")

    # Assembler le fichier
    body = "\n".join(field_lines)
    file_content = HEADER + body + FOOTER

    # Écrire (UTF-8 sans BOM pour compat Linux/Docker)
    OUTPUT_PATH.write_text(file_content, encoding="utf-8")
    print(f"\n✓ Fichier généré : {OUTPUT_PATH}")
    print(f"  Taille : {OUTPUT_PATH.stat().st_size:,} octets")
    print(f"  Lignes : {file_content.count(chr(10)) + 1}")


if __name__ == "__main__":
    main()
