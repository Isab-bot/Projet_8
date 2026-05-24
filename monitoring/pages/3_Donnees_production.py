"""Page Streamlit "Données de production" — aperçu de la table predictions.

Cette page est l'écran d'inventaire de la base PostgreSQL : elle permet à
l'utilisateur de visualiser concrètement les lignes stockées par l'API,
de choisir le volume affiché (10 à 100 dernières prédictions) et la liste
des colonnes visibles parmi les 335 disponibles.

C'est la page qui sera screenshotée pour le livrable de l'étape 8
(captures dans ``docs/screenshots/`` — cf. design-doc §6.1).

Couvre la **métrique non numérotée** "aperçu de la table predictions" du
design-doc §3.4. Aucun calcul Evidently, aucune transformation lourde :
juste un SELECT et un affichage tabulaire.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from monitoring.core.db import (
    fetch_predictions,
    get_instrumented_predictions_count,
    get_last_prediction_timestamp,
    get_predictions_count,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Colonnes affichées par défaut dans le tableau. Choix : les colonnes
# techniques (timestamp, request_id, model_version, threshold, prediction,
# prediction_proba) + latence/inférence + 2 features importantes
# (EXT_SOURCE_2, CODE_GENDER) pour donner une idée du payload.
DEFAULT_COLUMNS: list[str] = [
    "timestamp",
    "request_id",
    "model_version",
    "prediction",
    "prediction_proba",
    "threshold",
    "latency_ms",
    "inference_ms",
    "EXT_SOURCE_2",
    "CODE_GENDER",
]

MIN_ROWS = 10
MAX_ROWS = 100
DEFAULT_ROWS = 20


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def cached_count() -> int:
    """Compte total des prédictions en base (cached)."""
    return get_predictions_count()


@st.cache_data(ttl=60)
def cached_instrumented_count() -> int:
    """Compte des prédictions avec instrumentation (cached)."""
    return get_instrumented_predictions_count()


@st.cache_data(ttl=60)
def cached_last_timestamp() -> pd.Timestamp | None:
    """Timestamp de la dernière prédiction (cached)."""
    return get_last_prediction_timestamp()


@st.cache_data(ttl=60)
def cached_sample(limit: int, columns: tuple[str, ...]) -> pd.DataFrame:
    """Récupère un échantillon de prédictions.

    ``columns`` est un tuple (pas une liste) pour rester hashable côté
    cache_data Streamlit.
    """
    cols = list(columns) if columns else None
    return fetch_predictions(limit=limit, columns=cols)


@st.cache_data(ttl=60)
def cached_all_columns() -> list[str]:
    """Récupère la liste complète des colonnes de la table predictions.

    Utilisée pour le multi-select. On lit une seule ligne pour récupérer
    le schéma — pas besoin de scanner toute la table.
    """
    df = fetch_predictions(limit=1)
    return df.columns.tolist()


# ---------------------------------------------------------------------------
# Helpers d'affichage
# ---------------------------------------------------------------------------


def format_time_since(ts: pd.Timestamp | None) -> str:
    """Formate un timestamp en `il y a Xh Ymin` (UTC)."""
    if ts is None:
        return "—"
    now = pd.Timestamp(datetime.now(tz=timezone.utc))
    delta = now - ts
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f"il y a {total_seconds} s"
    minutes, _ = divmod(total_seconds, 60)
    if minutes < 60:
        return f"il y a {minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"il y a {hours}h {minutes:02d}min"
    days, hours = divmod(hours, 24)
    return f"il y a {days}j {hours}h"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Données de production", layout="wide")
st.title("Données de production")
st.caption(
    "Aperçu de la table `predictions` de la base PostgreSQL. Chaque ligne "
    "correspond à une prédiction effectuée par l'API : entrée client, "
    "sortie modèle, métadonnées techniques (timestamp, version, latence, "
    "inférence). C'est la source unique alimentant tous les rapports de "
    "monitoring."
)

# Sidebar : bouton Refresh
with st.sidebar:
    st.markdown("### Actions")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Vide le cache et recharge la table.")

# --- Connexion DB ---
try:
    total = cached_count()
    instrumented = cached_instrumented_count()
    last_ts = cached_last_timestamp()
except Exception as exc:  # noqa: BLE001
    st.error(f"Connexion à la base impossible : {exc}")
    st.stop()

if total == 0:
    st.warning(
        "La table `predictions` est vide. Lance "
        "`scripts/simulate_production.py` pour alimenter la base."
    )
    st.stop()

# ---------------------------------------------------------------------------
# §1 — KPI globaux
# ---------------------------------------------------------------------------

st.divider()
st.header("§1 — KPI globaux")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Prédictions totales",
        value=f"{total:,}".replace(",", " "),
    )

with col2:
    if total > 0:
        share = instrumented / total
        st.metric(
            label="Prédictions instrumentées",
            value=f"{instrumented:,}".replace(",", " "),
            delta=f"{share:.1%} du total",
            delta_color="off",
        )
    else:
        st.metric(label="Prédictions instrumentées", value="0")

with col3:
    if last_ts is not None:
        st.metric(
            label="Dernière prédiction",
            value=format_time_since(last_ts),
            delta=last_ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
            delta_color="off",
        )
    else:
        st.metric(label="Dernière prédiction", value="—")

st.caption(
    "Une prédiction est dite *instrumentée* si les colonnes `latency_ms` et "
    "`inference_ms` sont renseignées. Les lignes pré-existantes à "
    "l'instrumentation (étape 7) ont ces deux colonnes à NULL."
)

# ---------------------------------------------------------------------------
# §2 — Échantillon de la table
# ---------------------------------------------------------------------------

st.divider()
st.header("§2 — Échantillon de la table `predictions`")

# Récupération des colonnes disponibles pour le multi-select.
try:
    all_columns = cached_all_columns()
except Exception as exc:  # noqa: BLE001
    st.error(f"Impossible de lire le schéma de la table : {exc}")
    st.stop()

# Contrôles utilisateur
control_col1, control_col2 = st.columns([1, 3])

with control_col1:
    n_rows = st.slider(
        "Nombre de lignes",
        min_value=MIN_ROWS,
        max_value=MAX_ROWS,
        value=DEFAULT_ROWS,
        step=10,
        help="Les N prédictions les plus récentes (tri par `timestamp` décroissant).",
    )

with control_col2:
    # Valeurs par défaut : intersection entre DEFAULT_COLUMNS et all_columns
    # (toutes ne sont pas garanties d'exister, p.ex. EXT_SOURCE_2 si schéma
    # modifié).
    default_selection = [c for c in DEFAULT_COLUMNS if c in all_columns]
    selected_columns = st.multiselect(
        "Colonnes à afficher",
        options=all_columns,
        default=default_selection,
        help=(
            f"{len(all_columns)} colonnes disponibles dans la table. "
            "Par défaut : 10 colonnes essentielles (techniques + 2 features)."
        ),
    )

# Garde-fou : si l'utilisateur retire toutes les colonnes
if not selected_columns:
    st.warning("Sélectionne au moins une colonne pour afficher le tableau.")
    st.stop()

# Récupération du sample
with st.spinner("Chargement de l'échantillon..."):
    df = cached_sample(limit=n_rows, columns=tuple(selected_columns))

st.caption(
    f"Affichage des **{len(df):,}** prédictions les plus récentes "
    f"sur **{total:,}** au total. Tri par `timestamp` décroissant."
    .replace(",", " ")
)

# Formatage du timestamp si présent : conversion en string lisible
df_display = df.copy()
if "timestamp" in df_display.columns:
    df_display["timestamp"] = pd.to_datetime(df_display["timestamp"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

st.dataframe(df_display, use_container_width=True, hide_index=True)
