"""Page Streamlit "Drift Evidently" — métriques 4, 5 et 6 du design-doc.

Cette page répond à la question : *"Les données qui arrivent en production
sont-elles distribuées comme celles sur lesquelles le modèle a été entraîné ?"*

Elle compare le dataset de référence (10 000 observations issues du parquet
``data/reference_data.parquet``) à un échantillon des 3 000 prédictions
les plus récentes stockées en base PostgreSQL.

Trois rapports Evidently sont produits :

1. **Résumé global** — KPI de drift sur l'ensemble des features
2. **Top 5 SHAP** — détail par feature parmi les 5 plus importantes du modèle
3. **prediction_proba** — drift de la sortie du modèle (probabilité de défaut)

Pour chaque rapport, la page affiche une synthèse Streamlit native (KPI +
tableau) et un expander contenant le rapport HTML complet d'Evidently.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from monitoring.core.db import fetch_predictions, get_predictions_count
from monitoring.core.drift import (
    build_global_drift_report,
    build_prediction_proba_drift_report,
    build_top5_drift_report,
    extract_column_drift,
    extract_drift_summary,
)
from monitoring.core.shap_mapping import (
    SHAP_TO_RAW,
    SHAP_TOP_5_FEATURES,
    align_parquet_to_sql,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

REFERENCE_PARQUET_PATH = Path("data/reference_data.parquet")
PRODUCTION_SAMPLE_SIZE = 3000

# Colonnes à exclure du drift global : techniques (id, target, prediction)
# ou suivies à part (prediction_proba en métrique 6).
EXCLUDED_COLUMNS = {
    "TARGET",
    "prediction",
    "prediction_proba",
    "SK_ID_CURR",
    "Unnamed: 0",
    "request_id",
    "timestamp",
    "model_version",
    "id",
    "latency_ms",
    "inference_ms",
}

# ---------------------------------------------------------------------------
# Chargement des données (avec cache)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def load_reference() -> pd.DataFrame:
    """Charge le parquet de référence et aligne les noms sur la SQL."""
    df = pd.read_parquet(REFERENCE_PARQUET_PATH)
    return align_parquet_to_sql(df)


@st.cache_data(ttl=60)
def load_production(limit: int = PRODUCTION_SAMPLE_SIZE) -> pd.DataFrame:
    """Charge les N dernières prédictions depuis la base SQL."""
    return fetch_predictions(limit=limit)


@st.cache_data(ttl=60)
def count_production() -> int:
    """Compte total de prédictions en base."""
    return get_predictions_count()


def restrict_to_common_features(
    ref_df: pd.DataFrame, prod_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Restreint les deux DataFrames à l'intersection de leurs features.

    Exclut les colonnes techniques (cf. ``EXCLUDED_COLUMNS``) et ne garde
    que les colonnes présentes des deux côtés. C'est nécessaire car
    Evidently crashe avec ``Unexpected feature type unknown`` si une
    colonne demandée manque de la ``DataDefinition``.

    Returns
    -------
    tuple
        ``(ref_aligned, prod_aligned, common_columns)``.
    """
    ref_cols = set(ref_df.columns) - EXCLUDED_COLUMNS
    prod_cols = set(prod_df.columns) - EXCLUDED_COLUMNS
    common = sorted(ref_cols & prod_cols)
    return ref_df[common], prod_df[common], common

def detect_asymmetric_categoricals(
    ref_df: pd.DataFrame, prod_df: pd.DataFrame
) -> tuple[list[str], dict[str, dict[str, list[str]]]]:
    """Détecte les colonnes catégorielles avec modalités déséquilibrées.

    Une colonne est considérée asymétrique si au moins une modalité est
    présente d'un seul côté (ref ou prod). Evidently crashe sur ces
    colonnes avec ``Column ([modality]) is partially present in data``
    car le Z-test ne peut pas évaluer une modalité à fréquence 0.

    Cause typique dans ce projet : encodages incohérents entre le
    parquet de référence (ex: ``OCCUPATION_TYPE_GRP`` contenant '0'
    pour les modalités rares) et la table SQL (qui contient les
    libellés bruts comme 'Drivers', 'Managers').

    Parameters
    ----------
    ref_df : DataFrame de référence (10k obs).
    prod_df : DataFrame de production (3k obs).

    Returns
    -------
    tuple
        ``(columns_to_exclude, asymmetries)`` où :
        - ``columns_to_exclude`` : liste des noms de colonnes à retirer
        - ``asymmetries`` : détail ``{col: {'only_ref': [...], 'only_prod': [...]}}``
          pour affichage UI / debug
    """
    columns_to_exclude: list[str] = []
    asymmetries: dict[str, dict[str, list[str]]] = {}

    cat_cols = ref_df.select_dtypes(exclude=["number"]).columns.tolist()
    for c in cat_cols:
        if c not in prod_df.columns:
            continue
        vals_ref = set(ref_df[c].dropna().astype(str).unique())
        vals_prod = set(prod_df[c].dropna().astype(str).unique())
        only_ref = vals_ref - vals_prod
        only_prod = vals_prod - vals_ref
        if only_ref or only_prod:
            columns_to_exclude.append(c)
            asymmetries[c] = {
                "only_ref": sorted(only_ref),
                "only_prod": sorted(only_prod),
            }
    return columns_to_exclude, asymmetries

# ---------------------------------------------------------------------------
# Calculs Evidently (avec cache)
# ---------------------------------------------------------------------------
# Note : on cache les helpers de calcul, pas les Snapshot eux-mêmes (Streamlit
# cache le retour de fonction, donc le Snapshot l'est aussi). Les Snapshot
# sont sérialisables car Evidently les a conçus comme tels (méthode .dumps()).


@st.cache_data(ttl=60, show_spinner=False)
def compute_global_drift(
    ref_df: pd.DataFrame, prod_df: pd.DataFrame
) -> tuple[dict, str]:
    """Calcule le drift global et retourne le résumé + le HTML."""
    snap = build_global_drift_report(ref_df, prod_df)
    return extract_drift_summary(snap), snap.get_html_str(as_iframe=False)


@st.cache_data(ttl=60, show_spinner=False)
def compute_top5_drift(
    ref_df: pd.DataFrame, prod_df: pd.DataFrame
) -> tuple[dict, pd.DataFrame, str]:
    """Calcule le drift sur le top 5 SHAP, retourne résumé + détail + HTML."""
    snap = build_top5_drift_report(ref_df, prod_df)
    return (
        extract_drift_summary(snap),
        extract_column_drift(snap),
        snap.get_html_str(as_iframe=False),
    )


@st.cache_data(ttl=60, show_spinner=False)
def compute_proba_drift(
    ref_df: pd.DataFrame, prod_df: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Calcule le drift de prediction_proba, retourne détail + HTML."""
    snap = build_prediction_proba_drift_report(ref_df, prod_df)
    return extract_column_drift(snap), snap.get_html_str(as_iframe=False)


# ---------------------------------------------------------------------------
# Layout de la page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Drift Evidently", layout="wide")
st.title("Drift Evidently")
st.caption(
    f"Comparaison entre les données de référence (parquet, 10 000 obs) "
    f"et les {PRODUCTION_SAMPLE_SIZE} prédictions les plus récentes "
    f"stockées en base. Tests statistiques : Kolmogorov-Smirnov (numérique) "
    f"et Z-test (catégoriel), seuil de p-value 0.05 (défauts Evidently)."
)

# --- Sidebar : bouton refresh ---
with st.sidebar:
    st.markdown("### Actions")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Vide le cache et recharge tous les rapports.")

# --- Chargement des données ---
try:
    n_prod_total = count_production()
except Exception as exc:  # noqa: BLE001
    st.error(f"Connexion à la base impossible : {exc}")
    st.stop()

if n_prod_total == 0:
    st.warning(
        "La table `predictions` est vide. Lance d'abord "
        "`scripts/simulate_production.py` pour alimenter la base."
    )
    st.stop()

ref_full = load_reference()
prod_full = load_production(limit=PRODUCTION_SAMPLE_SIZE)
n_prod_used = len(prod_full)

st.info(
    f"**Référence** : {len(ref_full):,} observations · "
    f"**Production utilisée** : {n_prod_used:,} sur {n_prod_total:,} en base"
)

# Restriction à l'intersection pour le drift global.
ref_common, prod_common, common_cols = restrict_to_common_features(
    ref_full, prod_full
)

# Détection et exclusion des colonnes catégorielles à modalités déséquilibrées.
# Evidently crashe avec "Column ([X]) is partially present in data" si une
# modalité est présente d'un seul côté. Voir docstring de detect_asymmetric_categoricals.
asymmetric_cols, asymmetries = detect_asymmetric_categoricals(ref_common, prod_common)
if asymmetric_cols:
    cols_to_keep = [c for c in common_cols if c not in asymmetric_cols]
    ref_common = ref_common[cols_to_keep]
    prod_common = prod_common[cols_to_keep]
    common_cols = cols_to_keep
# ---------------------------------------------------------------------------
# §1 — RÉSUMÉ GLOBAL
# ---------------------------------------------------------------------------

st.divider()
st.header("§1 — Résumé global")

col1, col2, col3 = st.columns(3)

with st.spinner("Calcul du drift global..."):
    global_summary, global_html = compute_global_drift(ref_common, prod_common)

with st.spinner("Calcul du drift top 5 SHAP..."):
    top5_summary, top5_detail, top5_html = compute_top5_drift(ref_common, prod_common)

with st.spinner("Calcul du drift prediction_proba..."):
    # prediction_proba a été exclue par restrict_to_common_features, on la rajoute.
    ref_proba = ref_full[["prediction_proba"]]
    prod_proba = prod_full[["prediction_proba"]]
    proba_detail, proba_html = compute_proba_drift(ref_proba, prod_proba)

with col1:
    n_drift = global_summary["count"]
    n_total = len(common_cols)
    share = global_summary["share"]
    st.metric(
        label="Drift global",
        value=f"{n_drift} / {n_total}",
        delta=f"{share:.1%} des features",
        delta_color="off",
    )

with col2:
    n_drift_top5 = top5_summary["count"]
    share_top5 = top5_summary["share"]
    st.metric(
        label="Drift top 5 SHAP",
        value=f"{n_drift_top5} / 5",
        delta=f"{share_top5:.1%} du top 5",
        delta_color="off",
    )

with col3:
    if not proba_detail.empty:
        p_value = proba_detail.iloc[0]["p_value"]
        is_drift = bool(proba_detail.iloc[0]["drift_detected"])
        st.metric(
            label="Drift prediction_proba",
            value=f"p = {p_value:.4f}",
            delta="Drift détecté" if is_drift else "Pas de drift",
            delta_color="inverse" if is_drift else "off",
        )
    else:
        st.metric(label="Drift prediction_proba", value="—")

# ---------------------------------------------------------------------------
# §2 — DRIFT TOP 5 SHAP (DÉTAIL)
# ---------------------------------------------------------------------------

st.divider()
st.header("§2 — Drift top 5 SHAP (détail par feature)")

with st.expander("📋 Mapping nom SHAP → nom brut", expanded=False):
    mapping_df = pd.DataFrame(
        [
            {"Nom SHAP (transformer)": k, "Nom brut (parquet/SQL)": v}
            for k, v in SHAP_TO_RAW.items()
        ]
    )
    st.dataframe(mapping_df, hide_index=True, use_container_width=True)

st.caption(
    "**Note sur DAYS_EMPLOYED** : un drift peut apparaître sur cette feature "
    "indépendamment d'un drift métier réel. Cause connue : le preprocessing "
    "du Projet 6 (`-df['DAYS_EMPLOYED'] / 365` + traitement des anomalies "
    "365243 → 0) a été appliqué au parquet de référence mais pas au fichier "
    "`df_test_final.parquet` utilisé pour simuler la production. C'est un cas "
    "réaliste de **défaut d'alignement preprocessing training/serving**, "
    "exactement ce qu'un monitoring est censé détecter."
)

st.dataframe(
    top5_detail.style.format({"p_value": "{:.4f}"}).map(
        lambda v: "background-color: #ffcccc" if v is True else "",
        subset=["drift_detected"],
    ),
    hide_index=True,
    use_container_width=True,
)

# Top 5 SHAP : on monitore aussi explicitement les 5 features.
missing_top5 = [c for c in SHAP_TOP_5_FEATURES if c not in prod_full.columns]
if missing_top5:
    st.warning(
        f"Features SHAP absentes de la table prod : {missing_top5}. "
        "Le rapport top 5 n'est calculé que sur les features disponibles."
    )

with st.expander("📊 Rapport HTML Evidently complet — Top 5 SHAP", expanded=False):
    st.components.v1.html(top5_html, height=900, scrolling=True)

# ---------------------------------------------------------------------------
# §3 — DRIFT prediction_proba
# ---------------------------------------------------------------------------

st.divider()
st.header("§3 — Drift de prediction_proba (sortie modèle)")

st.caption(
    "Compare la distribution des probabilités prédites par le modèle entre "
    "la référence et la production. Un drift ici indique que le modèle se "
    "comporte différemment qu'à l'entraînement, ce qui peut signaler soit un "
    "changement de la population d'entrée, soit une dérive du modèle lui-même."
)

st.dataframe(
    proba_detail.style.format({"p_value": "{:.4f}"}).map(
        lambda v: "background-color: #ffcccc" if v is True else "",
        subset=["drift_detected"],
    ),
    hide_index=True,
    use_container_width=True,
)

with st.expander(
    "📊 Rapport HTML Evidently complet — prediction_proba", expanded=False
):
    st.components.v1.html(proba_html, height=700, scrolling=True)

# ---------------------------------------------------------------------------
# §4 — DRIFT GLOBAL (RAPPORT HTML COMPLET)
# ---------------------------------------------------------------------------

st.divider()
st.header("§4 — Drift global (rapport HTML complet)")

st.caption(
    f"Rapport Evidently complet calculé sur les **{len(common_cols)} features** "
    f"communes entre référence et production (colonnes techniques exclues : "
    f"`TARGET`, `SK_ID_CURR`, `prediction`, `prediction_proba`, identifiants "
    f"et métadonnées). Le rapport HTML est volumineux (~4 MB) ; il ne s'affiche "
    f"qu'à l'ouverture de l'expander ci-dessous."
)

if asymmetric_cols:
    with st.expander(
        f"⚠️ {len(asymmetric_cols)} colonne(s) catégorielle(s) exclue(s) du "
        f"drift global (modalités déséquilibrées)",
        expanded=False,
    ):
        st.markdown(
            "Ces colonnes ont des modalités présentes d'un seul côté "
            "(ref ou prod) ce qui empêche Evidently de calculer le drift "
            "via Z-test. C'est généralement le signe d'un **désalignement "
            "d'encodage entre training et serving** — exactement le genre "
            "d'anomalie qu'un monitoring est censé révéler."
        )
        for col, detail in asymmetries.items():
            st.markdown(f"**`{col}`**")
            if detail["only_ref"]:
                st.write(f"- Modalités présentes uniquement en référence : `{detail['only_ref']}`")
            if detail["only_prod"]:
                st.write(f"- Modalités présentes uniquement en production : `{detail['only_prod']}`")

with st.expander("📊 Rapport HTML Evidently complet — Drift global", expanded=False):
    st.components.v1.html(global_html, height=1200, scrolling=True)
