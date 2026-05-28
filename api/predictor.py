"""Encapsulation du modèle de scoring crédit et de la logique de prédiction.

Sépare la couche métier (chargement modèle, application du seuil) de la
couche HTTP (main.py). Le modèle est chargé une seule fois au démarrage
de l'API via le lifespan FastAPI.

Deux backends d'inférence sont supportés (sélection via config.INFERENCE_BACKEND) :

- "joblib" (défaut) : pipeline scikit-learn complet chargé depuis
  models/xgboost_champion.pkl. Comportement historique, inchangé.

- "onnx" : preprocessor sklearn (models/preprocessor.pkl) + XGBClassifier
  converti en ONNX Runtime (models/xgboost_classifier.onnx). Optimisation
  de l'étape 9. Le preprocessing reste en sklearn (décision Q1) ; seule
  l'inférence du classifier passe par ONNX Runtime.

Les deux backends exposent la même interface publique : predict(input_data)
-> PredictionOutput. Le reste de l'application (main.py, tests) est agnostique
au backend utilisé.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api.config import (
    DECISION_THRESHOLD,
    INFERENCE_BACKEND,
    MODEL_PATH,
    ONNX_MODEL_PATH,
    PREPROCESSOR_PATH,
)
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
    """Wrapper autour du modèle de scoring (backend joblib ou ONNX).

    Le modèle est chargé une seule fois à l'instanciation, puis réutilisé
    pour toutes les prédictions (point de vigilance MLOps : pas de chargement
    par requête).

    Attributes:
        backend: "joblib" ou "onnx". Détermine le moteur d'inférence.
        threshold: Seuil de décision optimal (F3) issu du Projet 6.
        model_path: Chemin du pipeline .pkl (backend joblib).
        pipeline: Pipeline scikit-learn chargé (backend joblib ; None sinon).
        preprocessor: ColumnTransformer sklearn (backend onnx ; None sinon).
        onnx_session: Session ONNX Runtime (backend onnx ; None sinon).
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threshold: float = DECISION_THRESHOLD,
        backend: str | None = None,
    ) -> None:
        self.backend = backend if backend is not None else INFERENCE_BACKEND
        self.model_path = model_path
        self.threshold = threshold

        # Attributs des deux backends, initialisés à None puis remplis
        # selon le backend choisi.
        self.pipeline = None
        self.preprocessor = None
        self.onnx_session = None

        if self.backend == "onnx":
            self.preprocessor, self.onnx_session = self._load_onnx()
        else:
            # Backend joblib (défaut). Comportement historique inchangé.
            self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Charge le pipeline scikit-learn complet depuis le disque (backend joblib).

        Raises:
            FileNotFoundError: Si le fichier .pkl est introuvable.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable: {self.model_path}. "
                "Vérifiez que l'étape 2 (extraction MLflow) a bien été exécutée."
            )
        return joblib.load(self.model_path)

    def _load_onnx(self):
        """Charge le preprocessor sklearn et la session ONNX Runtime (backend onnx).

        Returns:
            Tuple (preprocessor, onnx_session).

        Raises:
            FileNotFoundError: Si un des deux artefacts ONNX est introuvable.
        """
        # Import local : onnxruntime n'est nécessaire que pour ce backend.
        import onnxruntime as ort

        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessor introuvable: {PREPROCESSOR_PATH}. "
                "Lancez scripts/convert_to_onnx.py pour le générer."
            )
        if not ONNX_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle ONNX introuvable: {ONNX_MODEL_PATH}. "
                "Lancez scripts/convert_to_onnx.py pour le générer."
            )

        preprocessor = joblib.load(PREPROCESSOR_PATH)
        onnx_session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        return preprocessor, onnx_session

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Prédit la probabilité de défaut et la décision pour un dossier crédit.

        Délègue au backend configuré (joblib ou onnx). Les deux produisent
        un PredictionOutput identique en structure.

        Args:
            input_data: Features validées par Pydantic (326 features).

        Returns:
            PredictionOutput contenant probabilité, décision binaire et seuil.

        Raises:
            ModelNotLoadedError: Si le modèle du backend n'est pas chargé.
        """
        if self.backend == "onnx":
            return self._predict_onnx(input_data)
        return self._predict_joblib(input_data)

    def _build_dataframe(self, input_data: PredictionInput) -> pd.DataFrame:
        """Reconstruit le DataFrame avec les noms ORIGINAUX (alias).

        Les noms d'alias (ex: 'BUREAU_CREDIT_ACTIVE_<lambda>') sont attendus
        par le preprocessor du pipeline. Partagé entre les deux backends.
        """
        features_dict = input_data.model_dump(by_alias=True)
        return pd.DataFrame([features_dict])

    def _predict_joblib(self, input_data: PredictionInput) -> PredictionOutput:
        """Inférence via le pipeline scikit-learn complet (backend joblib)."""
        if self.pipeline is None:
            raise ModelNotLoadedError("Le pipeline n'est pas chargé en mémoire.")

        df = self._build_dataframe(input_data)

        # Probabilité de la classe positive (défaut de crédit = classe 1)
        probability = float(self.pipeline.predict_proba(df)[0, 1])

        decision = int(probability >= self.threshold)

        return PredictionOutput(
            probability=probability,
            decision=decision,
            threshold=self.threshold,
        )

    def _predict_onnx(self, input_data: PredictionInput) -> PredictionOutput:
        """Inférence via preprocessor sklearn + ONNX Runtime (backend onnx).

        Le preprocessing (ColumnTransformer) reste en sklearn. Seule
        l'inférence du XGBClassifier passe par ONNX Runtime.
        """
        if self.onnx_session is None or self.preprocessor is None:
            raise ModelNotLoadedError("Le modèle ONNX n'est pas chargé en mémoire.")

        df = self._build_dataframe(input_data)

        # Étape 1 : preprocessing sklearn (identique au pipeline d'origine)
        x_transformed = self.preprocessor.transform(df)
        if hasattr(x_transformed, "toarray"):
            x_transformed = x_transformed.toarray()
        x_transformed = x_transformed.astype(np.float32)

        # Étape 2 : inférence ONNX Runtime
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: x_transformed})

        # Le XGBoost ONNX renvoie [label, probabilities]. Les probas peuvent
        # être au format liste de dicts {0: p0, 1: p1} ou array [N, 2].
        probas_output = outputs[1]
        if isinstance(probas_output, list) and isinstance(probas_output[0], dict):
            probability = float(probas_output[0][1])
        else:
            probability = float(np.asarray(probas_output)[0, 1])

        decision = int(probability >= self.threshold)

        return PredictionOutput(
            probability=probability,
            decision=decision,
            threshold=self.threshold,
        )
