"""Générateur du modèle ORM Prediction (script utilitaire one-shot).

Lit la liste et les types des features depuis api.schemas.PredictionInput,
applique le mapping de noms (api.feature_naming.to_sql_column_name),
et génère api/models.py contenant la classe SQLAlchemy Prediction.

Source de vérité des types : api/schemas.py (Pydantic).
On ne lit PAS le parquet pour les types (le parquet pourrait évoluer
indépendamment de l'API ; c'est le contrat Pydantic qui fait foi).

Usage :
    uv run python scripts/generate_prediction_model.py

Sortie :
    api/models.py (écrasé si existe)

Garde-fou :
    Le script échoue si le nombre de features détectées ne correspond pas
    à api.config.N_FEATURES (326). Cela évite de générer silencieusement
    un fichier incohérent.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ajout du répertoire racine au PYTHONPATH pour pouvoir importer api.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.config import N_FEATURES, PROJECT_ROOT  # noqa: E402
from api.feature_naming import to_sql_column_name  # noqa: E402
from api.schemas import PredictionInput  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "api" / "models.py"

# Longueur réservée pour les colonnes VARCHAR
# (mesure max dans le reference dataset = 29 ; marge généreuse à 64)
STRING_LENGTH = 64


# Mapping types Python -> code SQLAlchemy (column type, import statement)
PYTHON_TO_SQLA = {
    float: ("Float", "Float"),
    int: ("Integer", "Integer"),
    str: (f"String(length={STRING_LENGTH})", "String"),
    bool: ("Boolean", "Boolean"),
}


def python_type_to_sqla(py_type: type) -> tuple[str, str]:
    """Renvoie (code SQLAlchemy column type, nom du type à importer)."""
    if py_type not in PYTHON_TO_SQLA:
        raise ValueError(
            f"Type Python non supporté : {py_type.__name__}. "
            f"Étendre PYTHON_TO_SQLA dans le générateur."
        )
    return PYTHON_TO_SQLA[py_type]


def python_type_to_mapped(py_type: type) -> str:
    """Renvoie l'annotation Mapped[...] correspondant au type Python."""
    return f"Mapped[{py_type.__name__} | None]"


def generate_header(used_sqla_types: set[str]) -> str:
    """Génère l'en-tête du fichier avec les imports adaptés."""
    sqla_types_imports = ", ".join(sorted(used_sqla_types))
    return f'''"""Modèle ORM SQLAlchemy 2.0 pour la table predictions.

"""Modèle ORM SQLAlchemy 2.0 pour la table predictions.
⚠️ FICHIER AUTO-GÉNÉRÉ — NE PAS ÉDITER À LA MAIN.
Régénérer via : uv run python scripts/generate_prediction_model.py
Source des features : api.schemas.PredictionInput
Mapping des noms   : api.feature_naming.to_sql_column_name
Les colonnes métier (id, request_id, timestamp, model_version, threshold,
prediction_proba, prediction, latency_ms, inference_ms) sont définies dans
le HEADER du script générateur — pas éditer ici, éditer le script.
"""
from datetime import datetime

from sqlalchemy import DateTime, {sqla_types_imports}, func
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
        String(length=36), unique=True, index=True, nullable=False,
        comment="UUID de la requête, exposé au client pour traçabilité",
    )

    # --- Métadonnées ---
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
        server_default=func.current_timestamp(),
        comment="Horodatage UTC de la prédiction",
    )
    model_version: Mapped[str] = mapped_column(
        String(length=32), index=True, nullable=False,
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

    # --- Features (auto-générées depuis schemas.py) ---
'''


FOOTER = '''
    def __repr__(self) -> str:
        return (
            f"<Prediction id={self.id} request_id={self.request_id} "
            f"proba={self.prediction_proba:.4f} pred={self.prediction}>"
        )
'''


def main() -> None:
    print("Lecture des types depuis api.schemas.PredictionInput...")

    fields = PredictionInput.model_fields
    print(f"  Total champs Pydantic : {len(fields)}")

    if len(fields) != N_FEATURES:
        raise ValueError(
            f"Nombre de champs Pydantic incohérent : attendu {N_FEATURES}, "
            f"trouvé {len(fields)}. Vérifier api/schemas.py."
        )

    # Extraire (nom_python, type_python, alias) pour chaque champ
    print("\nGénération de la classe ORM...")
    field_lines: list[str] = []
    used_sqla_types: set[str] = {"Float", "Integer", "String"}  # types des colonnes système
    type_counts: dict[str, int] = {}
    n_renamed = 0

    for python_name, field_info in fields.items():
        py_type = field_info.annotation
        type_name = py_type.__name__
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

        sqla_code, sqla_import = python_type_to_sqla(py_type)
        used_sqla_types.add(sqla_import)
        mapped_annotation = python_type_to_mapped(py_type)

        # Le nom python de schemas.py doit déjà être un nom SQL valide
        # (alignement validé : schemas.py utilise des "_lambda" et pas "<lambda>")
        sql_name = to_sql_column_name(python_name)

        # L'alias original (si présent) sert à documenter la généalogie
        alias = field_info.alias
        comment_part = f"  # original: {alias}" if alias and alias != python_name else ""

        if sql_name != python_name:
            n_renamed += 1
            comment_part = f"  # original: {alias or python_name}"

        line = (
            f'    {sql_name}: {mapped_annotation} = mapped_column('
            f'{sqla_code}, nullable=True){comment_part}'
        )
        field_lines.append(line)

    print(f"  Distribution des types : {dict(sorted(type_counts.items()))}")
    print(f"  Renommées (lambda)     : {n_renamed}")
    print(f"  Identiques             : {len(fields) - n_renamed}")

    # Assembler le fichier
    body = "\n".join(field_lines)
    file_content = generate_header(used_sqla_types) + body + FOOTER

    OUTPUT_PATH.write_text(file_content, encoding="utf-8")
    print(f"\n✓ Fichier généré : {OUTPUT_PATH}")
    print(f"  Taille : {OUTPUT_PATH.stat().st_size:,} octets")
    print(f"  Lignes : {file_content.count(chr(10)) + 1}")


if __name__ == "__main__":
    main()

