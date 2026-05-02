# Changelog

Toutes les évolutions notables du projet sont documentées ici.

Le format suit la convention [Keep a Changelog](https://keepachangelog.com/),
et le projet adhère au versionnement sémantique [SemVer](https://semver.org/).

---

## [Unreleased]

### À venir
- Étape 4 : persistance PostgreSQL des prédictions (`feature/database-postgres`)
- Étape 5 : tests unitaires et d'intégration avec pytest
- Étape 6 : conteneurisation Docker
- Étape 7 : pipeline CI/CD GitHub Actions et déploiement Hugging Face Spaces
- Étape 8 : monitoring du drift avec Evidently et dashboard Streamlit

---

## Étape 3 — API FastAPI (`feature/api-fastapi`)

**Objectif :** exposer le modèle XGBoost figé du Projet 6 via une API REST.

### Ajouté

- Module `api/` structuré : `main.py`, `schemas.py`, `predictor.py`, `config.py`, `exceptions.py`
- Schéma Pydantic `PredictionInput` avec **326 champs typés** générés automatiquement depuis le pipeline et le reference dataset
- Gestion des 10 features avec noms non-Python valides (`<lambda>` issus du feature engineering P6) via le système d'alias Pydantic
- Classe `Predictor` qui charge le pipeline scikit-learn (ColumnTransformer + XGBoost) **une seule fois** au démarrage via le `lifespan` FastAPI
- Endpoint `POST /predict` appliquant le seuil de décision optimal (`0.33381930539322036`, F3 du Projet 6)
- Endpoint `GET /health` retournant l'état de l'API et du modèle
- Handlers d'exceptions custom :
  - Reformatage des erreurs de validation Pydantic en JSON lisible (422)
  - Mappage de `ModelNotLoadedError` vers HTTP 503
  - Filet de sécurité pour les erreurs inattendues (500)
- Documentation Swagger automatique des codes de réponse (200, 422, 500, 503)
- Script utilitaire `scripts/generate_input_schema.py` pour régénérer le schéma si besoin

### Validé

- Sanity check de bout en bout : la prédiction renvoyée par l'API est strictement identique à `prediction_proba` du reference dataset
- Test fonctionnel cas nominal → 200 OK
- Test fonctionnel champs manquants → 422 avec message lisible
- Test fonctionnel mauvais type → 422 avec message lisible

### Stack

- FastAPI 0.136.1, Pydantic, joblib, pandas
- Python 3.13, gestion des dépendances via UV

### Commits clés
a84fc76 (origin/feature/api-fastapi, feature/api-fastapi) feat(api): add custom exception handlers and document error responses in Swagger
cd0f721 feat(api): add FastAPI app with lifespan and health/predict endpoints
16d7e5f feat(api): add Predictor class with frozen pipeline and decision threshold
b5a411c feat(api): generate PredictionInput schema with 326 typed fields
68d8be2 feat(scripts): add input schema generator from pipeline metadata
856203f chore(api): add api module skeleton




## Étape 2 — Export du modèle champion (`feature/model-export`)

**Objectif :** récupérer l'artefact du modèle XGBoost depuis MLflow (Projet 6) et préparer le reference dataset pour Evidently.

### Ajouté

- Extraction du pipeline scikit-learn (`ColumnTransformer` + XGBoost) depuis MLflow vers `models/xgboost_champion.pkl`
- Génération de `data/reference_data.parquet` (326 features + TARGET + colonnes de prédiction pour le monitoring)
- Métadonnées du modèle (run UUID, métriques, seuil) sauvegardées en JSON
- Sanity check : prédictions identiques entre l'export et le notebook source du Projet 6

### Stack

- MLflow (lecture de l'expérience `Credit_Risk_04_Final_Evaluation_Test_Set`, run `a3ff1e12347c4bfc9b484ac36916eb14`)
- joblib pour la sérialisation
- pyarrow pour le format parquet

---

## Étape 1 — Initialisation du projet (`feature/project-init`)

**Objectif :** poser les fondations techniques du projet.

### Ajouté

- Structure de dossiers : `api/`, `monitoring/`, `models/`, `tests/`, `scripts/`, `docker/`, `data/`, `.github/workflows/`
- Configuration `pyproject.toml` (Python 3.13, UV)
- `.gitignore`, `.gitattributes` (force LF pour compatibilité Docker/Linux), `.env.example`
- Stratégie Git Flow : branches `main`, `develop`, et `feature/*` par étape
- Conventional Commits adoptés sur tout le projet