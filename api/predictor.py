"""Encapsulation du modèle XGBoost et de la logique de prédiction.

Sépare la couche métier (chargement modèle, application du seuil)
de la couche HTTP (main.py). Le modèle est chargé une seule fois
au démarrage de l'API via le lifespan FastAPI.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd

from api.config import DECISION_THRESHOLD, MODEL_PATH
from api.schemas import PredictionInput, PredictionOutput

# Le modèle a été pickled avec une version antérieure de XGBoost.
# Le warning émis au chargement est inoffensif (compatibilité ascendante OK)
# et est supprimé pour ne pas polluer les logs en production.
warnings.filterwarnings(
    "ignore",
    message=".*If you are loading a serialized model.*",
    category=UserWarning,
)


class ModelNotLoadedError(RuntimeError):
    """Levée quand on tente de prédire alors que le modèle n'est pas chargé."""


class Predictor:
    """Wrapper autour du pipeline scikit-learn (preprocessor + XGBoost classifier).

    Le pipeline est chargé une seule fois à l'instanciation, puis réutilisé
    pour toutes les prédictions (point de vigilance MLOps : pas de chargement
    par requête).

    Attributes:
        model_path: Chemin du fichier .pkl contenant le pipeline.
        threshold: Seuil de décision optimal (F3) issu du Projet 6.
        pipeline: Pipeline scikit-learn chargé en mémoire.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threshold: float = DECISION_THRESHOLD,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Charge le pipeline depuis le disque.

        Raises:
            FileNotFoundError: Si le fichier .pkl est introuvable.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable: {self.model_path}. "
                "Vérifiez que l'étape 2 (extraction MLflow) a bien été exécutée."
            )
        return joblib.load(self.model_path)

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Prédit la probabilité de défaut et la décision pour un dossier crédit.

        Args:
            input_data: Features validées par Pydantic (326 features).

        Returns:
            PredictionOutput contenant probabilité, décision binaire et seuil utilisé.

        Raises:
            ModelNotLoadedError: Si le pipeline n'a pas été chargé correctement.
        """
        if self.pipeline is None:
            raise ModelNotLoadedError("Le pipeline n'est pas chargé en mémoire.")

        # Reconstruction du DataFrame avec les noms ORIGINAUX (alias)
        # attendus par le préprocesseur du pipeline (ex: 'BUREAU_CREDIT_ACTIVE_<lambda>')
        features_dict = input_data.model_dump(by_alias=True)
        df = pd.DataFrame([features_dict])

        # Probabilité de la classe positive (défaut de crédit = classe 1)
        probability = float(self.pipeline.predict_proba(df)[0, 1])

        # Application du seuil métier
        decision = int(probability >= self.threshold)

        return PredictionOutput(
            probability=probability,
            decision=decision,
            threshold=self.threshold,
        )