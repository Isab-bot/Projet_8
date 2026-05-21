"""Dashboard de monitoring — Page d'accueil.

Affiche un état général de la base de prédictions et sert de point d'entrée
pour la navigation multi-pages. Les pages dédiées (Métriques API, Drift
Evidently, Données de production) sont chargées depuis monitoring/pages/.

Lancement :
    streamlit run monitoring/streamlit_app.py

Configuration :
    DATABASE_URL : URL PostgreSQL (défaut localhost:5433/credit_scoring)
"""

from __future__ import annotations

import streamlit as st

from monitoring.core.db import (
    get_instrumented_predictions_count,
    get_last_prediction_timestamp,
    get_predictions_count,
)

st.set_page_config(
    page_title="Monitoring credit scoring API",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cache des lectures DB (TTL 60 s, design-doc §3.4)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def cached_total() -> int:
    return get_predictions_count()


@st.cache_data(ttl=60)
def cached_instrumented() -> int:
    return get_instrumented_predictions_count()


@st.cache_data(ttl=60)
def cached_last_timestamp():
    return get_last_prediction_timestamp()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Monitoring")
st.sidebar.caption("Projet 8 — Credit scoring API")

if st.sidebar.button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Navigation : utilisez le menu ci-dessus pour accéder aux pages.")


# ---------------------------------------------------------------------------
# Page principale
# ---------------------------------------------------------------------------

st.title("Monitoring credit scoring API")
st.caption(
    "Vue d'ensemble de la base de prédictions. "
    "Détails par page dans le menu latéral."
)

try:
    total = cached_total()
    instrumented = cached_instrumented()
    last_ts = cached_last_timestamp()
except Exception as exc:  # noqa: BLE001
    st.error(
        "Impossible de se connecter à la base de prédictions. "
        f"Vérifiez que PostgreSQL tourne et que DATABASE_URL est correcte.\n\n"
        f"Erreur : {exc}"
    )
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Prédictions totales", value=f"{total:,}".replace(",", " "))

with col2:
    st.metric(
        label="Avec instrumentation",
        value=f"{instrumented:,}".replace(",", " "),
        help="Prédictions avec latency_ms renseignée (postérieures à la branche 1).",
    )

with col3:
    if last_ts is None:
        st.metric(label="Dernière prédiction", value="—")
    else:
        st.metric(
            label="Dernière prédiction (UTC)",
            value=last_ts.strftime("%Y-%m-%d %H:%M:%S"),
        )

st.markdown("---")
st.subheader("Pages disponibles")
st.markdown(
    """
    - **Métriques API** : taux d'acceptation, distribution des scores,
      percentiles de latence et d'inférence.
    - **Drift Evidently** : drift global et drift détaillé sur le top 5
      des features SHAP (référence vs production).
    - **Données de production** : aperçu de la table `predictions`.
    """
)
