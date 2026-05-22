"""Wrappers Evidently 0.7.21 pour les rapports de drift de la page monitoring.

Ce module est la couche d'abstraction entre la page Streamlit
``2_Drift_Evidently.py`` et la bibliothèque Evidently. Il expose :

- 3 fonctions de construction de rapports, une par métrique du design-doc :
    * ``build_global_drift_report`` (métrique 4 : drift sur toutes les
      features)
    * ``build_top5_drift_report`` (métrique 5 : drift détaillé sur les 5
      features SHAP)
    * ``build_prediction_proba_drift_report`` (métrique 6 : drift de la
      sortie modèle)

- 2 helpers de parsing pour transformer un ``Snapshot`` Evidently en
  structures Python simples utilisables côté Streamlit :
    * ``extract_drift_summary`` : extrait le résumé global
      (nombre de colonnes drift, part)
    * ``extract_column_drift`` : extrait la liste des p-values par colonne
      sous forme de ``pandas.DataFrame``

- 1 helper de configuration :
    * ``make_data_definition`` : construit un ``DataDefinition`` Evidently
      en détectant automatiquement les colonnes numériques et catégorielles
      à partir des dtypes pandas

**Convention d'API Evidently 0.7.21 utilisée :**

.. code-block:: python

    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    from evidently.metrics import ValueDrift

    data_def = DataDefinition(numerical_columns=[...], categorical_columns=[...])
    ref_ds = Dataset.from_pandas(ref_df, data_definition=data_def)
    prod_ds = Dataset.from_pandas(prod_df, data_definition=data_def)
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=prod_ds)

**Convention statistique :** Evidently utilise par défaut le test de
Kolmogorov-Smirnov pour les features numériques et le test de Z pour les
features catégorielles, avec un seuil de p-value de 0.05. Une p-value
strictement inférieure au seuil signifie *drift détecté* (rejet de
l'hypothèse "même distribution"). Ce seuil par défaut est conservé pour
rester conforme au design-doc (« tests statistiques : défauts Evidently »).
"""

from __future__ import annotations

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.report import Snapshot
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

from monitoring.core.shap_mapping import SHAP_TOP_5_FEATURES

# ---------------------------------------------------------------------------
# Construction du DataDefinition Evidently
# ---------------------------------------------------------------------------


def make_data_definition(df: pd.DataFrame) -> DataDefinition:
    """Construit un ``DataDefinition`` Evidently par détection auto des dtypes.

    Stratégie (cf. décision technique commit 5b, option i) :

    - Colonnes avec ``dtype`` numérique (``int*``, ``float*``, ``bool``)
      → ``numerical_columns``
    - Toutes les autres colonnes (``object``, ``category``, ``string``)
      → ``categorical_columns``

    Le parquet de référence et la table SQL sortent du même preprocessing
    Projet 6, donc les dtypes sont cohérents entre les deux. Aucune feature
    numérique du modèle n'est encodée en string.

    Parameters
    ----------
    df :
        DataFrame de référence ou production. Sert uniquement à récupérer
        les noms et dtypes des colonnes.

    Returns
    -------
    DataDefinition
        Definition Evidently prête à être passée à
        ``Dataset.from_pandas(..., data_definition=...)``.
    """
    # Note : on EXCLUT volontairement `bool` du numérique. Evidently appelle
    # `df.quantile()` sur les colonnes numériques, et numpy refuse de
    # soustraire des booléens (TypeError dans `_collect_numerical_stats`).
    # Les colonnes booléennes (ex: `drift_detected`, indicateurs binaires)
    # sont mieux traitées comme catégorielles via le Z-test d'Evidently.
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    return DataDefinition(
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
    )


def _make_datasets(
    ref_df: pd.DataFrame, prod_df: pd.DataFrame
) -> tuple[Dataset, Dataset]:
    """Construit deux ``Dataset`` Evidently partageant le même ``DataDefinition``.

    Le ``DataDefinition`` est construit à partir du DataFrame de référence
    (qui fait foi sur les dtypes attendus). Si la production a des colonnes
    avec des dtypes différents, c'est un signal de drift de schéma qui
    pourrait être ajouté en évolution future — pour l'instant on assume que
    les dtypes sont alignés.
    """
    data_def = make_data_definition(ref_df)
    ref_ds = Dataset.from_pandas(ref_df, data_definition=data_def)
    prod_ds = Dataset.from_pandas(prod_df, data_definition=data_def)
    return ref_ds, prod_ds


# ---------------------------------------------------------------------------
# Construction des 3 rapports de drift
# ---------------------------------------------------------------------------


def build_global_drift_report(
    ref_df: pd.DataFrame, prod_df: pd.DataFrame
) -> Snapshot:
    """Construit le rapport de drift global (métrique 4 du design-doc).

    Calcule le drift sur l'ensemble des colonnes communes aux deux
    DataFrames, en utilisant le ``DataDriftPreset`` sans argument. Le preset
    génère automatiquement :

    - un ``DriftedColumnsCount`` agrégé (nb de colonnes drift / part totale)
    - un ``ValueDrift`` par colonne (K-S pour numérique, Z-test pour
      catégoriel, seuil 0.05)

    Les colonnes "techniques" (``TARGET``, ``SK_ID_CURR``, ``Unnamed: 0``,
    ``prediction``, ``prediction_proba``) doivent être retirées en amont
    par l'appelant si on ne veut pas qu'elles figurent dans le rapport.

    Parameters
    ----------
    ref_df :
        DataFrame de référence (typiquement issu du parquet, déjà aligné
        sur les noms SQL via ``align_parquet_to_sql``).
    prod_df :
        DataFrame de production (issu de la table SQL ``predictions``).

    Returns
    -------
    Snapshot
        Snapshot Evidently exposant ``.dict()`` pour la synthèse et
        ``.get_html_str(as_iframe=False)`` pour le rapport HTML complet.
    """
    ref_ds, prod_ds = _make_datasets(ref_df, prod_df)
    report = Report(metrics=[DataDriftPreset()])
    return report.run(reference_data=ref_ds, current_data=prod_ds)


def build_top5_drift_report(
    ref_df: pd.DataFrame, prod_df: pd.DataFrame
) -> Snapshot:
    """Construit le rapport de drift détaillé sur le top 5 SHAP (métrique 5).

    Utilise ``DataDriftPreset(columns=SHAP_TOP_5_FEATURES)``, ce qui produit
    un rapport avec 1 ``DriftedColumnsCount`` agrégé sur les 5 + 1
    ``ValueDrift`` par feature (5 au total). Le détail par feature permet
    de répondre à la question : "Lesquelles des 5 features les plus
    importantes du modèle ont dérivé ?"

    Parameters
    ----------
    ref_df :
        DataFrame de référence. Doit contenir les 5 colonnes du top SHAP.
    prod_df :
        DataFrame de production. Doit contenir les 5 colonnes du top SHAP.

    Returns
    -------
    Snapshot
        Snapshot Evidently. Le détail par feature s'extrait via
        ``extract_column_drift(snapshot)``.
    """
    ref_ds, prod_ds = _make_datasets(ref_df, prod_df)
    report = Report(metrics=[DataDriftPreset(columns=SHAP_TOP_5_FEATURES)])
    return report.run(reference_data=ref_ds, current_data=prod_ds)


def build_prediction_proba_drift_report(
    ref_df: pd.DataFrame, prod_df: pd.DataFrame
) -> Snapshot:
    """Construit le rapport de drift de la sortie modèle (métrique 6).

    Utilise ``ValueDrift(column='prediction_proba')`` au lieu d'un preset :
    c'est sémantiquement correct ("je veux suivre le drift d'UNE colonne
    précise") et produit un rapport avec une seule métrique, donc un
    output minimal et un HTML léger.

    La colonne ``prediction_proba`` doit être présente dans les deux
    DataFrames. Elle représente la probabilité prédite par le modèle
    XGBoost champion d'être en défaut.

    Parameters
    ----------
    ref_df :
        DataFrame de référence contenant ``prediction_proba``.
    prod_df :
        DataFrame de production contenant ``prediction_proba``.

    Returns
    -------
    Snapshot
        Snapshot Evidently avec 1 seule métrique ``ValueDrift``.
    """
    ref_ds, prod_ds = _make_datasets(ref_df, prod_df)
    report = Report(metrics=[ValueDrift(column="prediction_proba")])
    return report.run(reference_data=ref_ds, current_data=prod_ds)


# ---------------------------------------------------------------------------
# Helpers de parsing des snapshots
# ---------------------------------------------------------------------------


def extract_drift_summary(snapshot: Snapshot) -> dict[str, float | int | None]:
    """Extrait le résumé global d'un snapshot de drift.

    Cherche la métrique ``DriftedColumnsCount`` dans le snapshot et
    retourne son ``value``, qui contient :

    - ``count`` : nombre de colonnes ayant drift (selon le seuil p-value
      0.05)
    - ``share`` : part des colonnes ayant drift (entre 0 et 1)

    Fonctionne pour le rapport global et le rapport top 5 (qui ont tous
    deux un ``DriftedColumnsCount``). Pour le rapport ``prediction_proba``
    qui n'utilise qu'un ``ValueDrift`` sans agrégation, cette fonction
    retourne un dictionnaire avec ``count=None`` et ``share=None``.

    Parameters
    ----------
    snapshot :
        Snapshot retourné par une des 3 fonctions ``build_*_drift_report``.

    Returns
    -------
    dict
        Dictionnaire avec les clés ``count`` (int ou None) et ``share``
        (float entre 0 et 1, ou None).
    """
    data = snapshot.dict()
    for metric in data.get("metrics", []):
        name = metric.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            value = metric.get("value", {})
            return {
                "count": int(value.get("count", 0)),
                "share": float(value.get("share", 0.0)),
            }
    # Cas du rapport prediction_proba : pas d'agrégat global.
    return {"count": None, "share": None}


def extract_column_drift(snapshot: Snapshot) -> pd.DataFrame:
    """Extrait le détail du drift par colonne sous forme de DataFrame.

    Parcourt les métriques ``ValueDrift(column=...)`` du snapshot et
    construit un DataFrame avec une ligne par colonne. Le DataFrame est
    trié par p-value croissante (les plus fortes preuves de drift en
    premier).

    Le seuil de p-value 0.05 est appliqué pour la colonne ``drift_detected``
    (conformément au défaut Evidently retenu dans le design-doc).

    Le nom du test (K-S ou Z-test) est extrait du ``metric_name`` qui suit
    le format ``ValueDrift(column=X,method=...,threshold=0.05)``.

    Parameters
    ----------
    snapshot :
        Snapshot retourné par une des 3 fonctions ``build_*_drift_report``.

    Returns
    -------
    pandas.DataFrame
        Colonnes : ``column`` (nom de la feature), ``method`` (nom du test
        statistique), ``p_value`` (float), ``drift_detected`` (bool).
        Trié par ``p_value`` croissante. Peut être vide si le snapshot ne
        contient aucune métrique ``ValueDrift``.
    """
    threshold = 0.05
    rows: list[dict[str, str | float | bool]] = []
    data = snapshot.dict()
    for metric in data.get("metrics", []):
        name = metric.get("metric_name", "")
        if not name.startswith("ValueDrift"):
            continue
        # Extraction depuis le metric_name de la forme
        # "ValueDrift(column=X,method=K-S p_value,threshold=0.05)"
        inner = name[len("ValueDrift(") : -1] if name.endswith(")") else name
        column_name = ""
        method = ""
        for part in inner.split(","):
            if part.startswith("column="):
                column_name = part[len("column=") :]
            elif part.startswith("method="):
                method = part[len("method=") :]
        p_value = float(metric.get("value", 1.0))
        rows.append(
            {
                "column": column_name,
                "method": method,
                "p_value": p_value,
                "drift_detected": p_value < threshold,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="p_value", ascending=True).reset_index(drop=True)
    return df
