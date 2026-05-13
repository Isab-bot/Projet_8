---
title: Projet 8 Credit Scoring API
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: API FastAPI de scoring crédit (XGBoost) — Projet 8 OpenClassrooms
---

# Projet 8 — Credit Scoring API

API FastAPI déployée pour le projet 8 OpenClassrooms (parcours AI Engineer).
Modèle XGBoost de scoring crédit, 326 features, seuil F3 optimisé.

**Endpoints disponibles** :

- GET / — racine, informations API
- GET /health — healthcheck (statut du service et du modèle)
- POST /predict — prédiction de risque de défaut de paiement
- GET /docs — documentation Swagger interactive

**Code source** : https://github.com/Isab-bot/Projet_8

**Disclaimer** : projet pédagogique, modèle entraîné sur des données fictives (dataset Home Credit). Ne pas utiliser pour de vraies décisions de crédit.
