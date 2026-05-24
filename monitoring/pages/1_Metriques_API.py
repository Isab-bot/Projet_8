"""Page Métriques API : taux d'acceptation, distribution des scores,
percentiles de latence et d'inférence."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from monitoring.core.db import fetch_predictions
from monitoring.core.metrics import (
    acceptance_rate,
    inference_percentiles,
    latency_percentiles,
    score_distribution,
)

# Constantes
N_RECENT: int = 3000
MODEL_THRESHOLD: float = 0.33381930539322036  # Seuil F3 du Projet 6

# Colonnes minimales à charger
COLS = ["prediction", "prediction_proba", "latency_ms", "inference_ms", "timestamp"]


st.set_page_config(page_title="Métriques API", layout="wide")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_recent(limit: int):
    """Charge les N dernières prédictions, filtrées sur les colonnes utiles."""
    return fetch_predictions(limit=limit, columns=COLS)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Métriques API")
if st.sidebar.button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Fenêtre d'analyse : {N_RECENT:,} prédictions les plus récentes."
    .replace(",", " ")
)
st.sidebar.caption(f"Seuil modèle : {MODEL_THRESHOLD:.4f}")


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

st.title("Métriques API")
st.caption(
    "Indicateurs de performance opérationnelle de l'API et distribution "
    "des scores prédits sur la fenêtre la plus récente."
)

try:
    df = load_recent(limit=N_RECENT)
except Exception as exc:  # noqa: BLE001
    st.error(f"Erreur de connexion à la base : {exc}")
    st.stop()

if len(df) == 0:
    st.warning("Aucune prédiction en base. Lancez le script de simulation.")
    st.stop()

# Plage temporelle couverte par l'échantillon
ts_min = df["timestamp"].min()
ts_max = df["timestamp"].max()
ts_min_str = ts_min.strftime("%Y-%m-%d %H:%M:%S") if ts_min is not None else "—"
ts_max_str = ts_max.strftime("%Y-%m-%d %H:%M:%S") if ts_max is not None else "—"

st.caption(
    f"Échantillon analysé : **{len(df):,}** prédictions, "
    f"de **{ts_min_str}** à **{ts_max_str}** (UTC)."
    .replace(",", " ")
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Section 1 — Taux d'acceptation et distribution des scores
# ---------------------------------------------------------------------------

st.subheader("Distribution des scores prédits")

accept = acceptance_rate(df)
n_accepted = int((df["prediction"] == 0).sum())
n_refused = int((df["prediction"] == 1).sum())

col_a, col_b, col_c = st.columns(3)
col_a.metric(
    "Taux d'acceptation",
    f"{accept * 100:.1f} %",
    help="Proportion de prédictions classées 0 (pas de défaut prédit).",
)
col_b.metric("Acceptés", f"{n_accepted:,}".replace(",", " "))
col_c.metric("Refusés", f"{n_refused:,}".replace(",", " "))

# Histogramme compact avec ligne de seuil
scores = score_distribution(df)
fig_hist = px.histogram(
    scores,
    nbins=50,
    title=None,
    labels={"value": "prediction_proba"},
)
fig_hist.add_vline(
    x=MODEL_THRESHOLD,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Seuil = {MODEL_THRESHOLD:.4f}",
    annotation_position="top right",
)
fig_hist.update_layout(
    xaxis_title="Probabilité prédite",
    yaxis_title="Nombre de prédictions",
    showlegend=False,
    bargap=0.05,
    height=280,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# Section 2 — Latence et inférence
# ---------------------------------------------------------------------------

st.subheader("Latence et temps d'inférence")

lat = latency_percentiles(df)
inf = inference_percentiles(df)

st.markdown("**Latence API totale (ms)**")
lc1, lc2, lc3, lc4 = st.columns(4)
lc1.metric("Moyenne", f"{lat['mean']:.2f}")
lc2.metric("p50", f"{lat['p50']:.2f}")
lc3.metric("p95", f"{lat['p95']:.2f}")
lc4.metric("p99", f"{lat['p99']:.2f}")

st.markdown("**Temps d'inférence modèle (ms)**")
ic1, ic2, ic3, ic4 = st.columns(4)
ic1.metric("Moyenne", f"{inf['mean']:.2f}")
ic2.metric("p50", f"{inf['p50']:.2f}")
ic3.metric("p95", f"{inf['p95']:.2f}")
ic4.metric("p99", f"{inf['p99']:.2f}")

# Note interprétative
ratio = (inf["mean"] / lat["mean"] * 100) if lat["mean"] > 0 else 0
st.caption(
    f"Sur cet échantillon, l'inférence représente **{ratio:.1f} %** "
    f"de la latence totale. Le reste correspond au middleware, à la "
    f"sérialisation Pydantic et à l'écriture en base. La distribution "
    f"est très resserrée (p99 = {lat['p99']:.2f} ms vs moyenne = "
    f"{lat['mean']:.2f} ms), les chiffres ci-dessus suffisent à la "
    f"caractériser."
)
