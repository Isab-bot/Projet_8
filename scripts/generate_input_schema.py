"""Génère api/schemas.py à partir du pipeline et du reference dataset.

Source de vérité :
- Liste des features = pipeline.named_steps['preprocessor'].feature_names_in_
- Types des features = dtypes du parquet de référence

Le fichier généré contient :
- PredictionInput : 326 champs Pydantic typés (tous obligatoires)
- PredictionOutput : sortie de /predict (manuel)
- HealthResponse : sortie de /health (manuel)

Ce script est un utilitaire one-shot. Le modèle étant figé en P8,
il ne devrait pas être relancé sauf changement de modèle.
"""

from pathlib import Path

import joblib
import pandas as pd

# --- Chemins ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_champion.pkl"
PARQUET_PATH = PROJECT_ROOT / "data" / "reference_data.parquet"
OUTPUT_PATH = PROJECT_ROOT / "api" / "schemas.py"


# --- Mapping pandas dtype -> type Python pour Pydantic ---
def dtype_to_python(dtype) -> str:
    """Convertit un dtype pandas en annotation de type Python."""
    name = str(dtype)
    if name.startswith("float"):
        return "float"
    if name.startswith("int"):
        return "int"
    if name == "bool":
        return "bool"
    if name in ("object", "str", "string"):
        return "str"
    raise ValueError(f"Type non supporté: {name}")

def sanitize_field_name(name: str) -> str:
    """Convertit un nom de feature en identifiant Python valide.
    
    Les caractères '<' et '>' issus des lambdas pandas sont supprimés.
    Si le nom est déjà valide, il est retourné tel quel.
    """
    if name.isidentifier():
        return name
    safe = name.replace("<", "").replace(">", "")
    if not safe.isidentifier():
        raise ValueError(f"Impossible de sanitizer le nom: {name!r} -> {safe!r}")
    return safe

def main() -> None:
    print(f"Chargement du pipeline depuis {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    expected_features = list(
        pipeline.named_steps["preprocessor"].feature_names_in_
    )
    print(f"  -> {len(expected_features)} features attendues")

    print(f"Chargement du parquet depuis {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  -> {df.shape[1]} colonnes")

    # On récupère les dtypes des features attendues uniquement
    dtypes = df[expected_features].dtypes
    print(f"  -> {len(dtypes)} dtypes récupérés")

    # Génération du contenu de schemas.py
    lines = [
        '"""Schémas Pydantic pour la validation des entrées/sorties de l\'API.',
        "",
        "Fichier généré automatiquement par scripts/generate_input_schema.py",
        "Source de vérité : pipeline.preprocessor.feature_names_in_ (326 features)",
        "Tous les champs sont obligatoires (le modèle a été entraîné sans NaN).",
        '"""',
        "",
        "from pydantic import BaseModel, Field",
        "",
        "",
        "class PredictionInput(BaseModel):",
        '    """Input du endpoint /predict — 326 features attendues par le modèle."""',
        "",
        "    model_config = {",
        '        "populate_by_name": True,',
        "    }",
        "",
    ]

    n_aliased = 0
    for feature_name in expected_features:
        py_type = dtype_to_python(dtypes[feature_name])
        safe_name = sanitize_field_name(feature_name)
        if safe_name == feature_name:
            # Nom déjà valide : champ simple
            lines.append(f"    {feature_name}: {py_type}")
        else:
            # Nom à aliaser : on conserve le nom original côté JSON
            lines.append(
                f'    {safe_name}: {py_type} = Field(..., alias="{feature_name}")'
            )
            n_aliased += 1

    # Output schemas (manuels)
    lines += [
        "",
        "",
        "class PredictionOutput(BaseModel):",
        '    """Sortie du endpoint /predict."""',
        "",
        "    probability: float = Field(",
        "        ..., ge=0.0, le=1.0,",
        '        description="Probabilité prédite de défaut de crédit (classe 1)."',
        "    )",
        "    decision: int = Field(",
        "        ..., ge=0, le=1,",
        '        description="Décision binaire : 0 = crédit accordé, 1 = défaut prédit."',
        "    )",
        "    threshold: float = Field(",
        "        ...,",
        '        description="Seuil de décision utilisé (0.3338, optimal F3 du Projet 6)."',
        "    )",
        "",
        "",
        "class HealthResponse(BaseModel):",
        '    """Sortie du endpoint /health."""',
        "",
        "    status: str",
        "    model_loaded: bool",
        "    api_version: str",
        "",
    ]

    content = "\n".join(lines)

     # Information : combien de champs ont été aliasés
    if n_aliased:
        print(f"  -> {n_aliased} champs avec alias (noms d'origine non-identifiants Python)")

    print(f"Écriture de {OUTPUT_PATH}")
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"  -> {len(lines)} lignes générées")
    print("\n✅ schemas.py généré avec succès")


if __name__ == "__main__":
    main()