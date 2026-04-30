# Projet 8 — Confirmez vos compétences en MLOps

Déploiement et monitoring d'un modèle de scoring crédit pour l'entreprise *Prêt à Dépenser*.

## 🎯 Contexte

Ce projet consiste à mettre en production le modèle de scoring crédit développé lors du Projet 6 (*Initiez-vous au MLOps*), avec :
- Une **API FastAPI** servant le modèle XGBoost
- Un **dashboard Streamlit** pour le monitoring du data drift
- Un **pipeline CI/CD** automatisé via GitHub Actions
- Un **déploiement Docker** sur Hugging Face Spaces

## 📦 Stack technique

| Composant | Outil |
|-----------|-------|
| API | FastAPI + Uvicorn |
| Modèle | XGBoost (artefact Projet 6) |
| Base de données | PostgreSQL |
| Monitoring | Streamlit + Evidently |
| Conteneurisation | Docker |
| CI/CD | GitHub Actions |
| Déploiement | Hugging Face Spaces |
| Gestionnaire de paquets | UV |

## 🚀 Démarrage rapide

### Prérequis
- Python 3.13
- UV ([installation](https://docs.astral.sh/uv/))
- Docker & Docker Compose

### Installation

```powershell
# Cloner le repo
git clone <url-du-repo>
cd Projet_8

# Installer les dépendances
uv sync

# Activer l'environnement
.venv\Scripts\Activate.ps1

# Copier le template d'env
Copy-Item .env.example .env
# Puis éditer .env avec vos valeurs
```

### Lancer l'API en local

```powershell
uv run uvicorn api.main:app --reload
```

### Lancer le dashboard de monitoring

```powershell
uv run streamlit run monitoring/dashboard.py
```

### Lancer la stack complète (Docker)

```powershell
docker-compose up --build
```

## 📁 Structure du projet

```
Projet_8/
├── api/              # API FastAPI
├── monitoring/       # Dashboard Streamlit + drift
├── models/           # Artefacts modèle (Projet 6)
├── tests/            # Tests unitaires
├── scripts/          # Scripts utilitaires
├── docker/           # Dockerfile
└── .github/          # CI/CD GitHub Actions
```

## 🧪 Tests

```powershell
uv run pytest
```

## 📊 Livrables

- [ ] API FastAPI fonctionnelle
- [ ] Tests unitaires (couverture > 80%)
- [ ] Dockerfile
- [ ] Dashboard de monitoring (drift, latence, scores)
- [ ] Stockage des prédictions PostgreSQL
- [ ] Pipeline CI/CD GitHub Actions
- [ ] Déploiement Hugging Face Spaces
- [ ] Documentation README

---

*Projet réalisé dans le cadre de la formation AI Engineer — OpenClassrooms*
