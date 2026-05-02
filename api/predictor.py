"""Encapsulation du modèle XGBoost et de la logique de prédiction.

Sépare la couche métier (chargement modèle, application du seuil)
de la couche HTTP (main.py). Le modèle est chargé une seule fois
au démarrage de l'API via le lifespan FastAPI.
"""

# TODO: classe Predictor avec méthodes load() et predict()
# TODO: application du seuil DECISION_THRESHOLD pour la décision binaire