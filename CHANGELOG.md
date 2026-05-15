# Changelog

Toutes les évolutions notables du projet sont documentées ici.

Le format suit la convention [Keep a Changelog](https://keepachangelog.com/),
et le projet adhère au versionnement sémantique [SemVer](https://semver.org/).

---

## [Unreleased]



### À venir

- Étape 7 : pipeline CI/CD GitHub Actions et déploiement Hugging Face Spaces
- Étape 8 : monitoring du drift avec Evidently et dashboard Streamlit

---
## Étape 6 — Conteneurisation Docker (`feature/docker`)

**Objectif :** conteneuriser l'API et la base de données pour préparer le déploiement Hugging Face Spaces (étape 7) et garantir la reproductibilité de l'environnement de dev.

### Ajouté

- `docker/Dockerfile` : build multi-stage en deux étapes
  - **Stage `builder`** basé sur `ghcr.io/astral-sh/uv:python3.13-bookworm-slim` : installe les dépendances dans un venv via `uv sync --frozen`, avec layer caching optimisé (`pyproject.toml` + `uv.lock` copiés avant le code applicatif)
  - **Stage `runtime`** basé sur `python:3.13-slim-bookworm` (sans UV ni outils de build) : copie le venv et le code depuis le builder, ajoute `curl` pour le healthcheck
  - User non-root `appuser` UID 1000 (convention Hugging Face Spaces)
  - `HEALTHCHECK` Docker sur `GET /health` (intervalle 30s, start-period 15s, retries 3)
  - Modèle `.pkl` copié dans l'image (autonomie HF Spaces, pas de dépendance externe)
- `docker/docker-compose.yml` : orchestration locale API + Postgres
  - Service `postgres` (image `postgres:16`) avec volume nommé `projet8-postgres-data` pour la persistance entre redémarrages
  - Service `api` avec `depends_on: service_healthy` pour attendre que Postgres soit réellement prêt avant de démarrer (évite les races condition sur Alembic au lifespan)
  - `healthcheck` Postgres via `pg_isready`
  - Réseau Docker interne : l'API parle à `postgres:5432` (nom du service + port interne), le port 5433 est exposé sur l'host pour le client SQL externe
  - Crédentials Postgres alignés avec `.env.example` (`credit_user` / `credit_pass` / `credit_scoring`)
- `.dockerignore` à la racine : exclusion des artefacts inutiles dans le contexte de build (`.venv`, `.git`, caches Python/pytest/coverage, `tests/`, `data/raw`, fichiers IDE et OS)

### Modifié

- `.env.example` :
  - Suppression des variables obsolètes (`PREDICTION_THRESHOLD=0.5` et `FEATURE_NAMES_PATH=models/feature_names.json`, traînantes depuis l'étape 4)
  - Correction du port Postgres : `5432` → `5433` (le port effectivement exposé par compose sur l'host)
  - Commentaires mis à jour pour référencer `docker/docker-compose.yml` et inclure la commande de démarrage

### Validé

- `docker compose up -d` : les deux containers démarrent et passent en `healthy`
- Alembic applique automatiquement la migration baseline (`206600e28d40 create predictions table`) au démarrage du lifespan FastAPI
- `GET /health` répond `{"status":"ok","model_loaded":true,"api_version":"0.1.0"}`
- `POST /predict` (avec `tests/fixtures/sample_request.json`) renvoie une décision cohérente avec un `request_id` UUID4
- Vérification de la persistance via `psql` depuis le container Postgres : la prédiction est correctement insérée avec tous les champs (`request_id`, `prediction_proba`, `prediction`, `threshold`, `timestamp`)
- **Non-régression complète** : 42 tests pytest passants, 0 warning, couverture globale 99% (identique à l'étape 5)

### Décisions techniques

- **Image de base UV** (`ghcr.io/astral-sh/uv:python3.13-bookworm-slim`) cohérente avec le Projet 7, embarque `uv` directement (pas besoin de l'installer)
- **Multi-stage** pour découpler build et runtime : l'image finale ne contient ni `uv` ni outils de build, juste le venv et le code applicatif
- **Modèle `.pkl` copié dans l'image** plutôt que monté en volume : HF Spaces nécessite l'autonomie complète, pas de stockage externe accessible
- **`docker/` plutôt que racine** : cohérent avec l'arborescence définie en étape 1, mais impose `docker build -f docker/Dockerfile .` depuis la racine pour avoir le bon contexte
- **`psycopg[binary]`** déjà dans les dépendances → pas besoin d'installer `libpq5` au runtime, les binaires sont embarqués
- **Crédentials Postgres en clair dans `docker-compose.yml`** : c'est du dev local, pas de secret. Pour HF Spaces (étape 7), passage par les Secrets HF Spaces.
- **Volume nommé explicite** (`name: projet8-postgres-data`) pour éviter le préfixage automatique par le nom du projet compose

### Stack

- Docker Engine + Docker Compose v2
- Image base builder : `ghcr.io/astral-sh/uv:python3.13-bookworm-slim`
- Image base runtime : `python:3.13-slim-bookworm`
- Postgres 16 (image officielle)

### Caractéristiques de l'image finale

- **Disk usage** : 2.74 GB (multi-arch amd64+arm64 par défaut)
- **Content size** : 774 MB (taille effective téléchargée/uploadée)
- L'image contient actuellement des dépendances qui seront sorties dans des `[dependency-groups]` UV à terme : `streamlit`, `evidently`, `plotly` (étape 8 monitoring), `pytest`, `pytest-cov`, `httpx` (groupe dev). Optimisation reportée à l'étape 9.

### Reste à faire (cleanup futurs)

- Refactorer `pyproject.toml` pour sortir les dépendances de tests (`pytest`, `pytest-cov`, `httpx`) et de monitoring (`streamlit`, `evidently`, `plotly`) dans des groupes dédiés (étape 9 ou 10)
- Tests de la suite pytest exécutés depuis le conteneur (étape 7, intégrée au workflow CI GitHub Actions)

### Commits clés

```
33b0fc9 chore(env): remove obsolete vars and fix Postgres port in .env.example
fd6efb4 feat(docker): add multi-stage Dockerfile and compose for API + Postgres
```
## Cleanup — Refactoring des dépendances en groupes UV (`chore/dependency-groups`)

**Objectif :** réorganiser `pyproject.toml` en dependency groups pour préparer l'étape 7 (CI/CD) et l'étape 8 (image Streamlit séparée pour le monitoring), et alléger l'image Docker de l'API.

### Modifié

- `pyproject.toml` :
  - `[project] dependencies` réduit aux 12 dépendances de runtime API (FastAPI, SQLAlchemy, Alembic, xgboost, scikit-learn, pandas, numpy, joblib, psycopg, python-dotenv, pydantic, uvicorn)
  - Nouveau groupe `[dependency-groups] dev` : `pytest`, `pytest-cov`, `httpx`
  - Nouveau groupe `[dependency-groups] monitoring` : `streamlit`, `evidently`, `plotly`
- `uv.lock` régénéré : réorganisation des paquets dans `[package.dev-dependencies]`, **aucune version résolue ne change**
- `docker/Dockerfile` : les deux commandes `uv sync --frozen` deviennent `uv sync --frozen --no-dev --no-group monitoring` (image runtime sans pytest ni streamlit)

### Validé

- `uv sync` (sans flag) installe `[project] dependencies` + le groupe `dev` (convention UV) → flux local de dev inchangé
- 42 tests passants, 0 warning, couverture 99% (non-régression complète)
- `import streamlit` échoue avec `ModuleNotFoundError` dans l'environnement local et dans l'image Docker (groupe `monitoring` non activé) → comportement attendu
- `docker compose up` : containers healthy, `/health` répond, `/predict` génère un `request_id` valide, persistance DB OK
- La prédiction de référence sur `tests/fixtures/sample_request.json` reste identique au bit près (`probability = 0.8837481141090393`) → image fonctionnellement équivalente

### Gain sur l'image Docker

| Métrique | Avant | Après | Gain |
|---|---|---|---|
| Content size | 774 MB | 618 MB | −156 MB |
| Disk usage | 2.74 GB | 1.89 GB | −850 MB |

### Bénéfices pour la suite

- Image API plus légère pour le déploiement Hugging Face Spaces (étape 7)
- À l'étape 8 (monitoring Streamlit + Evidently), on pourra créer une **deuxième image Docker dédiée** au dashboard avec `uv sync --group monitoring --no-default-groups`, sans embarquer xgboost/scikit-learn

### Décisions techniques

- **Groupe `dev` UV plutôt que `[project.optional-dependencies]`** : les `dependency-groups` sont la convention UV moderne (PEP 735), automatiquement activés par `uv sync`, et compatibles avec les flags `--no-dev` / `--group` / `--no-group`. Plus idiomatique pour ce projet déjà 100% UV.
- **`monitoring` non activé par défaut** : il faut le demander explicitement via `--group monitoring` (contrairement à `dev`). Évite d'installer Streamlit/Evidently dans tous les contextes où on n'en a pas besoin.

### Commits clés

```
PR #3 : chore(deps): split dependencies into project/dev/monitoring groups
```


## Étape 5 — Tests unitaires et d'intégration (`feature/tests`)

**Objectif :** instrumenter le projet avec une suite pytest exhaustive (unitaires + intégration) couvrant la logique métier, la persistance et le contrat HTTP, avant la conteneurisation Docker (étape 6).

### Ajouté

- Infrastructure pytest avec stratégie pyramide : unitaires rapides (mocks) + intégration bout-en-bout (vrai modèle XGBoost)
- `tests/conftest.py` : fixtures partagées
  - `test_engine` (session-scoped) : SQLite fichier temporaire via `tmp_path_factory`, tables créées via `Base.metadata.create_all` (Alembic non appliqué en test, cf. décision plus bas)
  - `db_session` (function-scoped) : nettoyage `DELETE FROM predictions` entre tests pour isolation
  - `client` (function-scoped) : `TestClient` FastAPI avec `app.dependency_overrides[get_db]` redirigé vers la SQLite de test
  - `sample_payload_dict` et `sample_prediction_input` : payload réel à 326 features chargé depuis `tests/fixtures/sample_request.json`
- `tests/fixtures/sample_request.json` : payload de test versionné (extrait de `reference_data.parquet`, ligne 0). Stocké dans `tests/fixtures/` plutôt que `data/` qui est gitignored, pour fonctionner en CI sur fresh clone.
- **Tests unitaires** (25 tests) :
  - `tests/unit/test_feature_naming.py` (9 tests) : conversion pandas ↔ SQL, idempotence, double-underscore collapse, reverse mapping
  - `tests/unit/test_predictor.py` (11 tests) : pipeline mocké pour tester la logique pure (extraction proba classe 1, application du seuil, comportement à l'égalité, `by_alias=True`, `ModelNotLoadedError`, `FileNotFoundError`)
  - `tests/unit/test_db_service.py` (5 tests) : insertion réelle en SQLite de test (mocker la session reviendrait à tester SQLAlchemy lui-même), vérification que `commit()` a bien été appelé via `expire_all()`, multi-insertion
- **Tests d'intégration** (10 tests, vrai modèle XGBoost chargé via le `lifespan`) :
  - `tests/integration/test_root.py` (2 tests) : structure de la page d'accueil
  - `tests/integration/test_health.py` (4 tests) : statut `ok` quand modèle chargé, `degraded` quand `app.state.predictor=None`, exposition de la version API
  - `tests/integration/test_predict.py` (10 tests) : happy path (200, structure de réponse, types, UUID4 valide), persistance en DB de test, validation 422 (champ manquant, mauvais type), 503 (modèle non chargé), **fail-open 200 sur erreur DB** (protège la décision d'archi #5), **500 sur exception générique** (TestClient configuré avec `raise_server_exceptions=False` pour que les handlers FastAPI soient invoqués)

### Modifié

- `api/main.py` :
  - Variable d'environnement `SKIP_ALEMBIC_ON_STARTUP` qui désactive l'auto-migration Alembic au démarrage (positionnée à `1` par le conftest avant l'import du module). Permet aux tests d'utiliser une SQLite de test sans déclencher Alembic sur cette base.
  - Cleanup étape 4 : `HTTP_422_UNPROCESSABLE_ENTITY` → `HTTP_422_UNPROCESSABLE_CONTENT` (dans `api/exceptions.py`)
- `pyproject.toml` : ajout de `[tool.pytest.ini_options].filterwarnings` pour silencer le `UserWarning` XGBoost de chargement de modèle sérialisé (le `warnings.filterwarnings()` posé dans `api/predictor.py` ne survit pas à l'initialisation des filtres par pytest)

### Validé

- **42 tests passants en ~3 secondes**, aucun warning
- **Couverture globale : 99%** (objectif initial 80%)
- Le test `TestPredictFailOpen.test_db_error_returns_200_anyway` verrouille la politique fail-open : si quelqu'un retire le `try/except SQLAlchemyError` de `/predict` un jour, le test échoue immédiatement.
- Le test d'intégration `TestPredictHappyPath` valide la chaîne complète : payload Pydantic 326 features → validation → vrai pipeline XGBoost → seuil F3 → persistance DB → format de réponse.

### Couverture par module

| Module | Couverture | Note |
|---|---|---|
| `api/config.py` | 100% | |
| `api/db_service.py` | 100% | |
| `api/exceptions.py` | 100% | |
| `api/feature_naming.py` | 100% | |
| `api/models.py` | 100% | |
| `api/predictor.py` | 100% | |
| `api/schemas.py` | 100% | |
| `api/main.py` | 92% | Lignes 43-46 non couvertes : bloc Alembic du `lifespan`, désactivé par design en test (cf. ci-dessous) |
| `api/database.py` | 73% | Fonction `get_db()` non testée directement (cf. ci-dessous) |

### Justifications des couvertures imparfaites

- **`api/database.py` (73%)** : la fonction `get_db()` est un wrapper trivial (`SessionLocal(); try: yield; finally: close()`) sans logique métier. La tester reviendrait à tester `try/finally` en Python. De plus, dans les tests d'intégration `get_db` est volontairement overridée via `app.dependency_overrides` pour rediriger vers la DB de test SQLite, donc la vraie fonction n'est jamais appelée par design. Le comportement effectif de `get_db()` (session par requête, fermeture automatique, absence de fuite) est validé indirectement par les tests d'intégration de `/predict` qui consomment des sessions DB réelles.
- **`api/main.py` lignes 43-46 (bloc Alembic du `lifespan`)** : ce bloc est explicitement désactivé en test via la variable `SKIP_ALEMBIC_ON_STARTUP=1` positionnée par le `conftest.py`. Le tester nécessiterait soit de lancer Alembic réel sur la DB de test (rejette notre stratégie d'isolation, et redondant avec `Base.metadata.create_all`), soit de mocker `alembic_command.upgrade` (test le mock, pas le code). Le comportement d'auto-migration au démarrage est validé manuellement à l'étape 4 et le sera à nouveau lors des tests CI sur Postgres réel à l'étape 7.

### Décisions techniques

- **DB de test = SQLite fichier temporaire** (pas in-memory) : reproduit fidèlement le comportement du fallback prod, évite les problèmes de partage de schéma multi-connexions de SQLite in-memory avec SQLAlchemy.
- **Isolation par `DELETE FROM`** plutôt que transaction-rollback : compatible avec le `commit()` que `log_prediction()` effectue (le rollback ne fonctionne pas si commit a déjà eu lieu).
- **Patch côté `api.main` et pas `api.db_service`** : `api/main.py` fait `from api.db_service import log_prediction`, donc le nom est lié au module `api.main` au moment de l'import. Patcher `api.db_service.log_prediction` n'aurait aucun effet sur le code utilisé par `/predict`.
- **`raise_server_exceptions=False` pour le test 500** : le défaut de `TestClient` (`True`) fait remonter les exceptions au test au lieu de laisser FastAPI invoquer son handler, ce qui empêchait la couverture de `generic_exception_handler`. Le client de la fixture standard reste avec le défaut (utile en debug).

### Stack

- pytest 9.0.3, pytest-cov 7.1.0, httpx 0.28.1 (via `TestClient` FastAPI)
- coverage 7.13.5, rapports `term-missing` + HTML dans `htmlcov/`

### Commits clés

7cd6086 (HEAD -> feature/tests, origin/feature/tests) test(api): add 500 internal error test and silence XGBoost warning
f9c919b test(api): add integration tests for POST /predict
44ed121 test(api): add integration tests for / and /health endpoints
7131bc6 test(db_service): add unit tests for log_prediction

## Étape 4 — Persistance des prédictions en base (`feature/database-postgres`)

**Objectif :** persister chaque prédiction en base pour alimenter le monitoring de drift à l'étape 8 (Evidently).

### Ajouté

- Module `api/database.py` : moteur SQLAlchemy 2.0, fabrique de sessions, `Base` ORM, dépendance FastAPI `get_db()`
- Stratégie multi-backend pilotée par `DATABASE_URL` : **PostgreSQL 16 (Docker, port 5433) en dev**, **SQLite en fallback** automatique pour CI et Hugging Face Spaces
- Module `api/feature_naming.py` : conversion centralisée des noms de features pandas/Pydantic ↔ SQL (gestion des `<lambda>`, `<lambda_0>`, `<lambda_1>`, `<lambda_2>` issus du feature engineering P6)
- Modèle ORM `api/models.py` auto-généré via `scripts/generate_prediction_model.py` à partir du contrat Pydantic (`PredictionInput`) — **typage correct** : 297 FLOAT, 22 INTEGER, 10 VARCHAR(64), 1 BOOLEAN, 3 colonnes système (timestamp, request_id, model_version)
- 3 index pour le monitoring : `request_id` (unique), `timestamp`, `model_version`
- Service `api/db_service.py` : fonction `log_prediction()` qui persiste un objet `Prediction` à partir d'un `PredictionInput` + sortie du `Predictor`
- Intégration dans `POST /predict` :
  - Génération d'un `request_id` UUID v4 côté serveur
  - Persistance synchrone avant le `return`
  - Politique **fail-open** : si l'INSERT échoue, on log l'erreur (niveau ERROR avec stack trace et `request_id`) mais la prédiction est tout de même retournée au client
  - Handler passé en `def` synchrone (option β) pour bénéficier du thread pool FastAPI
- Champ `request_id` ajouté à `PredictionOutput` (optionnel avec défaut vide, posé par le handler après l'appel au Predictor)
- Auto-migration Alembic au démarrage de l'API : `command.upgrade(config, "head")` dans le `lifespan`
- Script `scripts/sanity_check_db_service.py` : test de bout en bout `parquet → Pydantic → Predictor → log_prediction → DB → relecture`
- Script `scripts/generate_sample_request.py` : génère un payload JSON de test pour `POST /predict`

### Migrations Alembic

- `alembic init alembic/` avec `alembic/env.py` configuré pour lire `DATABASE_URL` depuis l'environnement (portable dev/CI/HF)
- Migration baseline `206600e28d40_create_predictions_table` : `CREATE TABLE predictions` avec 333 colonnes + 3 index
- Utilisation de `func.current_timestamp()` plutôt que `func.now()` pour garantir la portabilité Postgres ↔ SQLite

### Validé

- Sanity check de bout en bout sur les deux backends (PostgreSQL 16 et SQLite) : prédiction et features correctement persistées et relues
- Test fonctionnel HTTP `POST /predict` → 200 OK avec `request_id`, ligne effectivement insérée en base SQLite avec le même `request_id`
- Auto-migration au démarrage : la table `predictions` est créée si absente, idempotent si déjà à jour

### Bugs détectés et corrigés grâce au sanity check

- **Désalignement de types ORM** : la première version typait toutes les features en `Float`, alors que 10 sont des `str` (ex. `NAME_CONTRACT_TYPE = "Cash loans"`), 20 des `int`, 1 des `bool`. Corrigé en lisant les types depuis le contrat Pydantic.
- **`func.now()` non portable** : fonction valide sous PostgreSQL mais inexistante sous SQLite (`unknown function: now()`). Remplacée par `func.current_timestamp()`, supportée nativement par les deux moteurs.

### Stack

- SQLAlchemy 2.0 (style `Mapped[]`), Alembic 1.13+, psycopg v3 (driver Postgres moderne)
- PostgreSQL 16 dans Docker (image officielle, port 5433 pour éviter le conflit avec une instance native Windows)
- SQLite (intégré à Python) pour le fallback

### Reste à faire (cleanup)

- Mettre à jour `.env.example` : `PREDICTION_THRESHOLD=0.5` est obsolète (vraie valeur : `0.33381930539322036`), `FEATURE_NAMES_PATH=models/feature_names.json` réfère à un fichier qui n'existe plus
- Remplacer `HTTP_422_UNPROCESSABLE_ENTITY` (déprécié dans Starlette récent) par `HTTP_422_UNPROCESSABLE_CONTENT` dans `api/main.py`

### Commits clés
5a7f23c (HEAD -> feature/database-postgres, origin/feature/database-postgres) feat(api): auto-run alembic migrations on startup
2027996 feat(api): integrate prediction logging in /predict endpoint (fail-open)
d598298 fix(db): correct ORM types, add log_prediction service, validate end-to-end
c6f6eaf feat(db): initialize alembic and generate baseline migration
303a41f feat(db): add Prediction ORM model and naming utilities
cb6c945 feat(db): add DATABASE_URL config and engine setup
fc45798 chore(db): remove unused psycopg2-binary dependency


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