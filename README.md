# Projet 8 — Confirmez vos compétences en MLOps

[![CI](https://github.com/Isab-bot/Projet_8/actions/workflows/ci.yml/badge.svg)](https://github.com/Isab-bot/Projet_8/actions/workflows/ci.yml)
[![Deployed on HF Spaces](https://img.shields.io/badge/Deployed-Hugging%20Face%20Spaces-blue?logo=huggingface)](https://huggingface.co/spaces/Fox6768/projet8-credit-scoring-api)

> Mise en production complète d'un modèle de scoring crédit XGBoost : API REST, conteneurisation, pipeline CI/CD, déploiement automatisé et monitoring de drift. Projet réalisé dans le cadre de la formation **AI Engineer** d'OpenClassrooms (alternance).

## 🔗 Lien avec l'étape précédente

Cette version **v1.1.0** prolonge la v1.0.0 (API déployée sur Hugging Face Spaces) avec le **monitoring local de drift** via Streamlit + Evidently, conformément au design-doc de l'étape 8.

L'API publique reste **inchangée côté déploiement** : même image GHCR, même Space HF, mêmes endpoints. Deux ajouts seulement :

- **Instrumentation** : deux colonnes `latency_ms` et `inference_ms` ajoutées à la table `predictions` (migration Alembic), peuplées par un middleware FastAPI et un wrap autour de l'inférence dans `POST /predict`
- **Dashboard de monitoring local** : 3 pages Streamlit alimentées par la base PostgreSQL Docker locale, qui comparent la distribution des données de référence (parquet du Projet 6) à celle des prédictions courantes

Le dashboard **n'est pas déployé sur Hugging Face Spaces** — c'est un choix volontaire détaillé dans la section [Décisions d'architecture](#-décisions-darchitecture).

---

---

## 🎯 Objectif et contexte

Ce projet consiste à mettre en production le modèle de scoring crédit développé lors du **Projet 6** (*Initiez-vous au MLOps*) pour l'entreprise fictive *Prêt à Dépenser*. Le modèle XGBoost — entraîné sur le dataset Home Credit (~300 000 dossiers, 326 features) avec un seuil F3 optimisé — est exposé via une API REST publique, déployée automatiquement sur Hugging Face Spaces à chaque release.

Le projet couvre l'ensemble du cycle MLOps post-entraînement :
- exposition du modèle via une **API REST** documentée
- **persistance des prédictions** en base pour le monitoring de drift à venir
- **conteneurisation Docker** multi-stage avec utilisateur non-root
- **pipeline CI/CD** GitHub Actions avec validation, build, registry et déploiement
- **déploiement automatique** sur Hugging Face Spaces via GHCR
- **monitoring de drift** avec Streamlit et Evidently (table `predictions` instrumentée, dashboard 3 pages, drift Evidently sur features et sorties modèle)

---

## 🌐 Démo live

L'API est accessible publiquement :

- **Page du Space** : https://huggingface.co/spaces/Fox6768/projet8-credit-scoring-api
- **API directe** : https://fox6768-projet8-credit-scoring-api.hf.space
- **Documentation Swagger interactive** : https://fox6768-projet8-credit-scoring-api.hf.space/docs

### Tester la prédiction en une commande

Depuis n'importe quel terminal (avec `curl` installé) :

```bash
curl -X POST "https://fox6768-projet8-credit-scoring-api.hf.space/predict" \
  -H "Content-Type: application/json" \
  --data-binary "@tests/fixtures/sample_request.json"
```

Réponse attendue (exemple) :

```json
{
  "request_id": "76e778b9-8a82-4169-9970-5fa1aa962670",
  "probability": 0.8837481141090393,
  "decision": 1,
  "threshold": 0.33381930539322036
}
```

> **Note** : la base SQLite du Space HF est éphémère (réinitialisée à chaque redémarrage). C'est un choix assumé pour ce projet portfolio (voir [Décisions d'architecture](#-décisions-darchitecture)). En contexte de production réelle, il suffirait de pointer `DATABASE_URL` vers un Postgres managé sans toucher au code.

---

## 🧠 Le modèle

| Caractéristique | Valeur |
|---|---|
| Algorithme | XGBoost (gradient boosting) |
| Pipeline | scikit-learn `ColumnTransformer` + XGBoost |
| Nombre de features | 326 (après feature engineering Projet 6) |
| Seuil de décision | `0.33381930539322036` (optimisé sur F3) |
| Métrique principale | F3 = 0.5583 (rappel privilégié, coût asymétrique des erreurs métier) |
| Dataset d'entraînement | Home Credit Default Risk (304 527 dossiers) |
| Run MLflow source | `a3ff1e12347c4bfc9b484ac36916eb14` (Projet 6) |
| Format de sérialisation | `joblib` (`models/xgboost_champion.pkl`, 3.4 MB) |

Le modèle est **figé** depuis le Projet 6 : aucun ré-entraînement, aucun versioning dynamique. Il est embarqué directement dans l'image Docker pour garantir l'autonomie du déploiement.

---

## 🏗️ Architecture

```mermaid
flowchart TB
    Client["🌐 Client HTTP<br/>(curl, Swagger, app tierce)"]

    subgraph HF["🤗 Hugging Face Spaces (Docker Space)"]
        direction TB
        API["⚡ FastAPI + Uvicorn<br/>port 8000<br/>+ middleware latence (v1.1.0)<br/>+ wrap inférence (v1.1.0)"]
        Predictor["🧠 Predictor<br/>(xgboost_champion.pkl)"]
        ORM["🗄️ SQLAlchemy + Alembic"]
        DB["💾 SQLite éphémère<br/>(fail-open)"]
        API --> Predictor
        API --> ORM
        ORM --> DB
    end

    subgraph CICD["🔁 GitHub Actions"]
        direction TB
        CI["✅ Workflow CI<br/>ruff + pytest (74 tests)"]
        CD["🚀 Workflow CD<br/>validate → build → push → deploy"]
    end

    GHCR["📦 GHCR Registry<br/>ghcr.io/isab-bot/projet8-api<br/>tags : :latest + :sha-&lt;commit&gt;"]
    Repo["🐙 Repo GitHub<br/>Isab-bot/Projet_8<br/>(source de vérité)"]

    Client -- "HTTPS" --> API
    GHCR -- "docker pull (image SHA)" --> HF
    CD -- "docker push" --> GHCR
    CD -- "git push (Dockerfile + README)" --> HF
    Repo -- "push main → trigger" --> CD
    Repo -- "push/PR main+develop → trigger" --> CI

    classDef external fill:#1e3a5f,stroke:#3b82f6,color:#fff
    classDef internal fill:#1e3a2f,stroke:#10b981,color:#fff
    class Client,Repo external
    class HF,CICD,GHCR internal
```

---

## 🛠️ Stack technique

| Domaine | Outil |
|---|---|
| Langage | Python 3.13 |
| Framework API | FastAPI 0.136 + Uvicorn |
| Modèle ML | XGBoost (artefact Projet 6) + scikit-learn pipeline |
| Validation / sérialisation | Pydantic v2 |
| ORM | SQLAlchemy 2.0 (style `Mapped[]`) |
| Migrations DB | Alembic |
| DB dev | PostgreSQL 16 (Docker) |
| DB prod (HF Spaces) | SQLite éphémère (fallback automatique) |
| Conteneurisation | Docker multi-stage (UV builder + Python slim runtime) |
| Registry d'images | GHCR (GitHub Container Registry) |
| CI/CD | GitHub Actions |
| Déploiement | Hugging Face Spaces (Docker Space) |
| Tests | pytest 9 + pytest-cov 7 + httpx |
| Linting | ruff |
| Gestionnaire de paquets | UV (Astral, PEP 735 dependency-groups) |
| Monitoring | Streamlit + Evidently 0.7.21 + PyArrow 24.0 |

---

## 🚀 Démarrage rapide en local

### Prérequis

- **Docker Desktop** (Windows) ou **Docker Engine** + **Docker Compose v2** (Linux/Mac)
- Git
- (Optionnel) `curl` pour tester les endpoints depuis le terminal

### Cloner et lancer la stack complète

```powershell
# Cloner le repo
git clone https://github.com/Isab-bot/Projet_8.git
cd Projet_8

# (Optionnel) Copier le template d'environnement
Copy-Item .env.example .env

# Lancer l'API + Postgres via Docker Compose
docker compose -f docker/docker-compose.yml up -d

# Vérifier que tout est healthy
docker compose -f docker/docker-compose.yml ps
```

Au démarrage, l'API applique automatiquement les migrations Alembic (création de la table `predictions`) puis charge le modèle XGBoost. Une fois `healthy` :

- **API** : http://localhost:8000
- **Swagger** : http://localhost:8000/docs
- **Postgres** : `localhost:5433` (utilisateur `credit_user`, mot de passe `credit_pass`, base `credit_scoring`)

### Tester une prédiction en local

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  --data-binary "@tests/fixtures/sample_request.json"
```

Tu peux ensuite vérifier que la prédiction a bien été persistée :

```powershell
docker exec -it $(docker compose -f docker/docker-compose.yml ps -q postgres) `
  psql -U credit_user -d credit_scoring -c "SELECT request_id, prediction_proba, prediction FROM predictions ORDER BY id DESC LIMIT 5;"
```

### Arrêter la stack

```powershell
docker compose -f docker/docker-compose.yml down

# Pour supprimer aussi le volume Postgres (perte des données)
docker compose -f docker/docker-compose.yml down -v
```

---

## 🧪 Qualité et tests

Le projet est instrumenté avec **74 tests pytest** (unitaires + intégration) couvrant la logique métier, la persistance, le contrat HTTP, l'instrumentation de latence/inférence et les helpers de monitoring (mapping SHAP, parsing Evidently). La couverture globale du code applicatif (hors UI Streamlit) est de **99 %**. Exécution en ~12 secondes en local et ~30 secondes en CI.Le projet est instrumenté avec **74 tests pytest** (unitaires + intégration) couvrant la logique métier, la persistance, le contrat HTTP, l'instrumentation de latence/inférence et les helpers de monitoring (mapping SHAP, parsing Evidently). La couverture globale du code applicatif (hors UI Streamlit) est de **99 %**. Exécution en ~12 secondes en local et ~30 secondes en CI.

### Lancer les tests en local

```powershell
# Installer les dépendances de dev (groupe activé par défaut)
uv sync

# Linter
uv run ruff check .

# Tests + couverture
uv run pytest --cov

# Rapport HTML détaillé
uv run pytest --cov --cov-report=html
# → ouvrir htmlcov/index.html
```

### Stratégie de tests

- **Unitaires** (52 tests) : logique pure isolée par des mocks (Predictor, naming, service DB, mapping SHAP, parsing Evidently)
- **Intégration** (22 tests) : `TestClient` FastAPI avec vrai modèle XGBoost chargé, SQLite de test pour vérifier la persistance bout-en-bout ; les tests d'instrumentation valident que `latency_ms` et `inference_ms` sont peuplées après chaque appel
- **Stratégie monitoring** : seules les fonctions pures sont testées (stratégie β du design-doc §5.2). Les pages Streamlit elles-mêmes sont validées visuellement, pas via pytest, car ce sont des couches UI sans logique métier complexe.

### Points verrouillés par tests

| Politique | Test | Pourquoi |
|---|---|---|
| Fail-open DB | `TestPredictFailOpen.test_db_error_returns_200_anyway` | Si quelqu'un retire le `try/except SQLAlchemyError`, le test échoue. L'observabilité ne doit jamais casser le service observé. |
| Validation Pydantic | `TestPredict422` | Tout payload mal formé doit retourner 422 avec un message lisible. |
| Modèle non chargé | `TestPredict503` | Si `app.state.predictor=None`, l'API renvoie 503 (pas 500). |
| Cohérence des prédictions | `TestPredictHappyPath` | La probabilité retournée est identique à celle du reference dataset (vérifie qu'aucune dérive n'a été introduite dans le pipeline). |

---

## 🔁 CI/CD

Deux workflows GitHub Actions distincts, avec des responsabilités séparées :

### Workflow CI (`.github/workflows/ci.yml`)

- **Déclencheurs** : push et pull request vers `main` et `develop`
- **Job unique** sur `ubuntu-latest` :
  1. Checkout du code
  2. Setup UV (Python 3.13, cache activé)
  3. `uv sync --frozen` (installe les dépendances de prod + le groupe `dev`)
  4. `uv run ruff check .`
  5. `uv run pytest --cov`
- **Concurrency** : un seul run actif par branche, annulation des précédents
- **Durée typique** : ~25 s

### Workflow CD (`.github/workflows/cd.yml`)

- **Déclencheurs** :
  - `workflow_dispatch` (manuel) — pour les re-déploiements, les corrections d'urgence et le debug
  - `push: branches: [main]` — pour le déploiement automatique en production
- **Trois jobs séquentiels** (`needs:` chaîné) :
  1. **`validate`** — re-exécute ruff + pytest (filet de sécurité avant tout déploiement)
  2. **`build-and-push-ghcr`** — build de l'image multi-stage depuis `docker/Dockerfile`, push vers GHCR avec deux tags : `:latest` (pointeur mouvant) et `:sha-<commit-court>` (traçabilité)
  3. **`deploy-to-hf`** — clone le repo Git Hugging Face Space, copie `huggingface/README.md` et réécrit le `huggingface/Dockerfile` avec le tag SHA (via `sed`), puis push vers HF
- **Concurrency** : un seul déploiement à la fois, pas d'annulation (pour éviter de couper un déploiement en cours)
- **Durée typique** : ~3 minutes (validate ~30 s + build & push ~2 min + deploy ~15 s)

### Pourquoi le tag SHA dans le Dockerfile HF ?

HF Spaces ne déclenche un rebuild que lorsque son repo Git reçoit un commit. Avec un tag `:latest` figé dans le Dockerfile HF, le contenu ne change jamais entre deux déploiements GHCR, donc HF ne rebuild pas et reste sur son ancienne image en cache. En réécrivant le tag en `:sha-<commit-court>` (différent à chaque commit), on garantit :
- un **rebuild HF automatique** à chaque déploiement
- une **traçabilité totale** : le Dockerfile HF affiche en clair quel commit GitHub produit l'image qui tourne
- un **rollback facile** : éditer le Dockerfile HF pour pointer sur n'importe quel `sha-*` antérieur

---

## 📦 Déploiement

### Flux nominal 

Toute fusion sur `main` déclenche le pipeline CD complet. Le déploiement HF Spaces se fait sans intervention manuelle.

```mermaid
flowchart LR
    PR["📝 PR<br/>develop → main"]
    Merge["🔀 Merge<br/>sur main"]
    CD["🚀 Workflow CD<br/>(GitHub Actions)"]
    Image["📦 Image GHCR<br/>:sha-&lt;commit&gt;"]
    Push["📤 Push<br/>repo HF Space"]
    Rebuild["🔄 Rebuild auto<br/>HF Spaces"]
    Running["✅ Space Running"]

    PR --> Merge
    Merge -->|trigger auto| CD
    CD -->|docker push| Image
    CD -->|git push| Push
    Push -->|déclenche| Rebuild
    Image -->|docker pull| Rebuild
    Rebuild --> Running

    classDef trigger fill:#1e3a5f,stroke:#3b82f6,color:#fff
    classDef build fill:#3a1e5f,stroke:#a855f7,color:#fff
    classDef result fill:#1e3a2f,stroke:#10b981,color:#fff
    class PR,Merge trigger
    class CD,Image,Push,Rebuild build
    class Running result
```

### Déclenchement manuel

Pour redéployer la même image (par exemple après un rollback HF), ou pour déboguer :

1. Aller sur https://github.com/Isab-bot/Projet_8/actions/workflows/cd.yml
2. Cliquer sur **Run workflow**
3. Sélectionner la branche (par défaut `develop`, ou n'importe quelle autre)
4. Confirmer

### Rollback

Trois niveaux possibles, du plus rapide au plus structurel :

1. **Rollback côté HF uniquement** : éditer `Dockerfile` dans le repo HF Space pour pointer sur un tag `sha-*` antérieur. HF rebuild automatiquement avec l'ancienne image. Effet : immédiat (~1 min). Pas de modification côté GitHub.
2. **Revert Git + redéploiement** : `git revert <commit>` sur `main` → la PR déclenche le CD → nouvelle image avec l'état antérieur du code.
3. **Hotfix branch** : créer une branche `hotfix/...` depuis le dernier commit stable, corriger, PR vers `main`.

---

## 🔍 Monitoring de drift (étape 8)

Le dashboard de monitoring est une application **Streamlit locale** qui compare les données de référence (issues du training Projet 6) aux prédictions accumulées en base par l'API. Il répond aux 3 questions clés du monitoring ML : *le modèle voit-il les mêmes données qu'à l'entraînement ?*, *son temps de réponse reste-t-il stable ?*, *ses sorties dérivent-elles ?*.

### Vue d'ensemble

```mermaid
flowchart LR
    Ref["📄 reference_data.parquet<br/>(10 000 obs, Projet 6)"]
    Sim["⚙️ simulate_production.py<br/>(3 000 requêtes,<br/>drift artificiel injecté)"]
    API["⚡ API FastAPI<br/>(local Docker)"]
    DB[("💾 PostgreSQL<br/>localhost:5433")]
    Streamlit["📊 Streamlit dashboard<br/>3 pages"]
    Evidently["🔬 Evidently 0.7.21<br/>(K-S + Z-test)"]

    Sim -->|POST /predict<br/>3000 fois| API
    API -->|INSERT| DB
    Ref -->|read_parquet| Streamlit
    DB -->|SELECT| Streamlit
    Streamlit --> Evidently
    Evidently --> Streamlit

    classDef data fill:#1e3a5f,stroke:#3b82f6,color:#fff
    classDef code fill:#1e3a2f,stroke:#10b981,color:#fff
    classDef tool fill:#3a1e5f,stroke:#a855f7,color:#fff
    class Ref,DB data
    class Sim,API,Streamlit code
    class Evidently tool
```

### Les 3 pages du dashboard

| Page | Contenu |
|---|---|
| **Métriques API** | Taux d'acceptation, distribution des scores avec ligne de seuil, percentiles de latence et de temps d'inférence (p50/p95/p99) |
| **Drift Evidently** | Résumé global (X/N features drift), drift détaillé sur les top 5 features SHAP, drift de `prediction_proba`, rapport HTML Evidently complet dans des expanders |
| **Données de production** | KPI globaux (total, instrumentées, dernière prédiction), aperçu interactif de la table `predictions` avec slider 10→100 lignes et multi-select des colonnes |

### Lancer le dashboard en local

Prérequis : la stack Docker (`API + Postgres`) doit tourner. Voir [Démarrage rapide en local](#-démarrage-rapide-en-local).

```powershell
# 1. (Si la table predictions est vide) Simuler de la production avec drift artificiel
uv run python scripts/simulate_production.py

# 2. Lancer Streamlit
$env:PYTHONPATH = (Get-Location).Path
uv run streamlit run monitoring/streamlit_app.py
```

Le dashboard s'ouvre sur http://localhost:8501 avec une page d'accueil + 3 pages accessibles via la sidebar.

### Aperçu des autres pages du dashboard

**Page Métriques API** : taux d'acceptation, distribution des scores prédits avec ligne de seuil, et percentiles de latence et d'inférence.

![Page Métriques API](docs/screenshots/streamlit_metriques_api.png)

**Page Drift Evidently** : résumé global (X/N features drift), détail sur le top 5 SHAP, drift de `prediction_proba`, et rapports HTML complets dans des expanders.

![Page Drift Evidently](docs/screenshots/streamlit_drift_evidently.png)

### Comment interpréter le drift

Evidently compare chaque feature entre référence et production via un test statistique :

- **Features numériques** : test de **Kolmogorov-Smirnov** sur la distribution
- **Features catégorielles** : test **Z** sur les proportions

Pour chaque feature, on récupère une **p-value**. La convention retenue est `p < 0.05 → drift détecté` (défaut Evidently). Interpréter prudemment :

- Le drift sur un grand nombre de features peut traduire un **désalignement preprocessing training/serving** plutôt qu'une vraie dérive métier
- Un cas typique observé sur ce projet : la feature `DAYS_EMPLOYED` est flaggée en drift alors que ce n'est pas une vraie dérive — c'est le preprocessing du Projet 6 (transformation `-x/365` + traitement des anomalies `365243 → 0`) qui n'est appliqué que côté référence et pas côté simulation production. **C'est exactement le genre d'anomalie qu'un monitoring doit révéler.**

### Données simulées et drift artificiel

Le script `scripts/simulate_production.py` rejoue **3000 requêtes** vers l'API à partir d'un échantillon de `df_test_final.parquet` (48 744 obs disponibles, seed fixée à 42). Pour rendre le drift visible, le script injecte un drift contrôlé sur **3 des 5 features SHAP top** :

| Feature | Transformation |
|---|---|
| `EXT_SOURCE_2` | Multiplication × 0.85 (baisse de score sociodémographique) |
| `EXT_SOURCE_3` | Bruit gaussien `N(0, 0.05)` clippé sur [0, 1] |
| `DAYS_EMPLOYED` | Shift additif `-1` (l'unité est l'année) |

Le drift est ainsi reproductible et permet de valider que le pipeline de monitoring détecte effectivement ce qu'il doit détecter.

### Stockage des prédictions

### Stockage des prédictions

Les captures ci-dessous montrent l'état de la table `predictions` après une simulation complète : schéma (extrait `\d predictions` sous `psql`), données stockées, et même table vue depuis le dashboard Streamlit.

**Schéma de la table — colonnes techniques et premières features métier**

![Schéma de la table predictions (haut)](docs/screenshots/sql_schema_psql_01.png)

On voit ici les colonnes techniques (`id`, `request_id`, `timestamp`, `model_version`, `threshold`, `prediction_proba`, `prediction`) avec leurs types et contraintes `NOT NULL`. Les défauts `CURRENT_TIMESTAMP` et `nextval('predictions_id_seq')` garantissent un horodatage et un identifiant uniques pour chaque insertion.

**Schéma de la table — fin du schéma + index**

![Schéma de la table predictions (bas)](docs/screenshots/sql_schema_psql_02.png)

On voit ici les dernières colonnes d'agrégation (notamment celles renommées `_lambda` côté SQL), les colonnes d'instrumentation `latency_ms` et `inference_ms` ajoutées à l'étape 8, et les 4 index : clé primaire `predictions_pkey`, contrainte d'unicité sur `request_id`, et index secondaires sur `model_version` et `timestamp` (utilisés par les requêtes du dashboard).

**Échantillon des données stockées**

![Données dans la table predictions](docs/screenshots/sql_data_psql.png)

Cinq prédictions récentes affichées via `SELECT ... ORDER BY timestamp DESC LIMIT 5`. Chaque ligne contient l'identifiant unique, la timestamp UTC, la décision (`prediction`), la probabilité prédite, le seuil de décision appliqué et les deux mesures d'instrumentation.

**Même table vue depuis le dashboard Streamlit**

![Page Données de production](docs/screenshots/streamlit_donnees_production.png)

### Évolutions possibles (hors scope étape 8)

Volontairement non implémentées pour cadrer le périmètre, mais identifiées dans le design-doc §2.3 comme axes d'amélioration :

- **Ratio temps d'inférence / latence totale** : diagnostic *where-time-goes* (combien de temps passé hors modèle)
- **Volume de prédictions par fenêtre temporelle** : nécessite un échelonnement des `timestamp` lors de la simulation
- **Taux d'erreur API** : nécessite de revenir sur la politique fail-open (la lever conditionnellement pour comptabiliser les erreurs sans casser le service)
- **Performance métier réelle** : nécessite des labels de production (non disponibles en environnement portfolio)
- **Fairness / biais par sous-population** : très pertinent en credit scoring (variables protégées comme `CODE_GENDER`) mais hors scope formation étape 8

---


## 🧭 Décisions d'architecture

Les choix structurants du projet sont documentés ici pour faciliter la lecture du code et préparer la soutenance.

### Postgres en dev, SQLite en fallback

L'API supporte les deux backends via la variable d'environnement `DATABASE_URL`. En dev local, Docker Compose lance un Postgres 16 dédié. En CI et sur HF Spaces, la variable n'est pas définie : le code bascule automatiquement sur SQLite (`data/predictions.db`). Cette dualité permet de prouver que la stack scale (Postgres = production-ready) tout en restant déployable gratuitement (SQLite = pas d'infrastructure managée). En production réelle, il suffirait d'injecter une `DATABASE_URL` pointant vers un Postgres managé (Neon, Supabase, RDS, Cloud SQL...) sans aucune modification du code applicatif.

### SQLite éphémère sur HF Spaces (assumé)

La SQLite est réinitialisée à chaque redémarrage du Space HF. C'est un choix volontaire pour un projet portfolio gratuit : pas de coût d'hébergement, fallback automatique déjà testé. Pour l'étape 8 (monitoring de drift), la décision retenue a été de **garder le dashboard en local seul** (pas de 2ème Space HF déployé), alimenté par la base PostgreSQL Docker locale. Cette approche évite à la fois le coût du Persistent Storage HF et la complexité d'une DB managée externe, tout en restant représentative d'un usage MLOps réel (le monitoring tourne typiquement séparé de l'API).

### Politique fail-open sur la persistance

Si l'INSERT en base échoue dans `POST /predict`, l'erreur est loggée (niveau ERROR, avec stack trace et `request_id`) mais la prédiction est tout de même retournée au client. L'instrumentation ne doit jamais casser le service observé. Cette politique est **verrouillée par un test dédié** (`TestPredictFailOpen.test_db_error_returns_200_anyway`) qui échoue si quelqu'un retire le `try/except`.

### Modèle `.pkl` embarqué dans l'image Docker

Le pipeline scikit-learn + XGBoost (3.4 MB) est copié directement dans l'image, plutôt que monté en volume ou téléchargé au démarrage. HF Spaces ne fournit pas de stockage externe accessible sans configuration spécifique, et le modèle est figé depuis le Projet 6 — il n'y a aucune raison de le re-charger dynamiquement. Cette approche garantit l'autonomie complète du container et la reproductibilité du déploiement.

### Dockerfile multi-stage (UV builder + Python slim runtime)

L'image finale est buildée en deux étapes : un stage `builder` basé sur l'image officielle UV (qui installe les dépendances dans un venv), puis un stage `runtime` basé sur `python:3.13-slim-bookworm` qui copie uniquement le venv et le code applicatif. L'image runtime ne contient ni UV ni outils de build, ce qui réduit la surface d'attaque et la taille (618 MB content size, 1.89 GB disk usage).

### User non-root UID 1000


Le container tourne en tant qu'utilisateur `appuser` (UID 1000) pour respecter la convention HF Spaces et appliquer le principe du moindre privilège. Le dossier `/app/data/` (pour la SQLite éphémère) est créé en root pendant le build puis chown à `appuser`, parce que SQLite refuse de créer une base dans un dossier inexistant.

### UV `dependency-groups` (PEP 735)

Les dépendances sont organisées en trois ensembles distincts :
- `[project] dependencies` : 12 paquets de runtime API (FastAPI, SQLAlchemy, xgboost, etc.)
- `[dependency-groups] dev` : pytest, pytest-cov, httpx, ruff — activé par défaut par `uv sync`
- `[dependency-groups] monitoring` : streamlit, evidently 0.7.21, plotly, pyarrow 24.0 — non activé par défaut, à demander explicitement via `--group monitoring`. Activé en CI pour pouvoir exécuter les tests de monitoring sans mock d'Evidently.

Cette organisation permet de produire une image Docker minimale (`uv sync --frozen --no-dev --no-group monitoring`) et de préparer une seconde image dédiée au monitoring à l'étape 8.

### Tag SHA dans le Dockerfile HF (vs `:latest`)

Le `huggingface/Dockerfile` dans le repo GitHub utilise `:latest` (lisible, testable en local), mais le workflow CD réécrit ce tag en `:sha-<commit-court>` via `sed` avant de pousser vers HF. Ce détail technique garantit que HF reçoit un vrai commit Git à chaque déploiement (donc un rebuild automatique) et apporte une traçabilité totale entre les images en production et les commits GitHub.

### Git Flow + Conventional Commits + PR systématiques

Le projet suit une convention Git Flow stricte :
- `main` = branche de release (déclenche le CD)
- `develop` = branche d'intégration (déclenche la CI uniquement)
- `feature/*`, `fix/*`, `chore/*` = branches de travail, mergées vers `develop` via PR

Tous les commits suivent les [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`). Aucune modification ne touche `develop` ou `main` sans passer par une Pull Request avec CI verte.

---

## 📁 Structure du repo

```mermaid
flowchart LR
    Root["📁 Projet_8/"]

    subgraph Application["🎯 Code applicatif"]
        direction TB
        Api["📂 api/<br/>FastAPI, Predictor,<br/>schémas, ORM, exceptions"]
        Models["📂 models/<br/>xgboost_champion.pkl<br/>+ métadonnées"]
        Alembic["📂 alembic/<br/>Migrations DB"]
    end

    subgraph Tests["🧪 Tests"]
        direction TB
        TestsDir["📂 tests/<br/>unit/ + integration/<br/>+ fixtures/"]
    end

    subgraph Conteneur["🐳 Conteneurisation"]
        direction TB
        Docker["📂 docker/<br/>Dockerfile multi-stage<br/>+ docker-compose.yml"]
        HF["📂 huggingface/<br/>README + Dockerfile HF<br/>(source HF Spaces)"]
    end

    subgraph Pipeline["🔁 Pipeline CI/CD"]
        direction TB
        Workflows["📂 .github/workflows/<br/>ci.yml + cd.yml"]
    end

    subgraph Utilitaires["🛠️ Utilitaires & Config"]
        direction TB
        Scripts["📂 scripts/<br/>Scripts one-shot<br/>(extract, generate, etc.)"]
        Config["📄 pyproject.toml · uv.lock · alembic.ini<br/>.env.example · .gitignore"]
        Doc["📄 README.md"]
    end

    subgraph Runtime["💾 Données runtime"]
        direction TB
        Data["📂 data/<br/>(gitignored, généré au démarrage)"]
    end

    Root --> Application
    Root --> Tests
    Root --> Conteneur
    Root --> Pipeline
    Root --> Utilitaires
    Root --> Runtime

    classDef code fill:#1e3a2f,stroke:#10b981,color:#fff
    classDef test fill:#3a2f1e,stroke:#f59e0b,color:#fff
    classDef container fill:#1e3a5f,stroke:#3b82f6,color:#fff
    classDef cicd fill:#3a1e5f,stroke:#a855f7,color:#fff
    classDef config fill:#3a1e2f,stroke:#ec4899,color:#fff
    classDef runtime fill:#2f2f2f,stroke:#6b7280,color:#fff

    class Application code
    class Tests test
    class Conteneur container
    class Pipeline cicd
    class Utilitaires config
    class Runtime runtime
```

---

## 📊 Livrables OpenClassrooms

-- [x] API FastAPI fonctionnelle (endpoints `/`, `/health`, `/predict`, `/docs`)
- [x] Tests unitaires et d'intégration (74 tests, couverture 99 % du code applicatif)
- [x] Dockerfile multi-stage
- [x] Stockage des prédictions (PostgreSQL en dev, SQLite en prod HF)
- [x] Pipeline CI GitHub Actions (ruff + pytest sur push/PR `main`/`develop`)
- [x] Pipeline CD GitHub Actions (build & push GHCR, déploiement HF auto sur `main`)
- [x] Déploiement Hugging Face Spaces (image publique, API accessible)
- [x] Documentation README (ce fichier)
- [x] Instrumentation API : middleware de latence, mesure du temps d'inférence, persistance en DB (`latency_ms`, `inference_ms`)
- [x] Dashboard de monitoring 3 pages (Métriques API, Drift Evidently, Données de production)
- [x] Drift Evidently : global sur 324 features, top 5 SHAP détaillé, sortie modèle `prediction_proba`
- [x] Script de simulation de production avec drift artificiel reproductible (seed=42, N=3000)
- [ ] Optimisation modèle (compétence OpenClassrooms identifiée a posteriori) — **étape 9 à venir, v1.2.0**
- [ ] Documentation utilisateur étendue (MkDocs) — **étape 10 à venir**

---

## 👤 Auteur

Isabelle

Projet réalisé dans le cadre du parcours *AI Engineer*. Le modèle source provient du Projet 6 (*Initiez-vous au MLOps*) du même parcours. Versions livrées :

- **v1.0.0** — API REST déployée, pipeline CI/CD, conteneurisation (étape 7)
- **v1.1.0** — Instrumentation + dashboard de monitoring de drift (étape 8, ce livrable)

Repo GitHub : https://github.com/Isab-bot/Projet_8

