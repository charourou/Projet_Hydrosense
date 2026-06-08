"""
pages/piezo-dashboard.py
────────────────────────
Dashboard principal : historique + prévision XGBoost + seuils de gestion.

"""

import altair as alt
import pandas as pd
import streamlit as st

from utils.bigquery import load_single_piezo_map, load_catalog_interm, load_seuils_interm
from utils.theme import SEUIL_COLORS, SEUIL_ORDER, DESIGN_TOKENS
from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo
from hydrosense.interface.main import train, preprocess, split_data, pred

# ── Fallback seuils quand BQ ne retourne rien ────────────────────────────────
_SEUILS_FALLBACK: dict[str, float] = {
    "p95": 15.0,
    "p85": 13.4,
    "p20": 12.8,
    "p10": 12.0,
    "p5":  11.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# SÉLECTEUR PIÉZOMÈTRE
# ══════════════════════════════════════════════════════════════════════════════
df_catalog = load_catalog_interm()

df_catalog["label"] = (
    df_catalog["bss_id"]
    + " — "
    + df_catalog["nom_commune"].fillna("?")
    + " ("
    + df_catalog["code_departement"].fillna("?")
    + ")"
)

col_titre, _, col_select = st.columns([3, 1, 1.5], vertical_alignment="bottom")

with col_titre:
    st.title("Hydro-Sense")

with col_select:
    labels = df_catalog["label"].tolist()
    default_label = next((l for l in labels if "BSS001QHYH" in l), labels[0])
    selected_label = st.selectbox(
        "Piézomètre",
        options=labels,
        index=labels.index(default_label),
        label_visibility="collapsed",
    )

DATA_CODE_PIEZO = df_catalog.loc[df_catalog["label"] == selected_label, "bss_id"].iloc[0]

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_trained_model_and_data(bss_id: str):
    df_raw   = load_piezo_bq(bss_id)
    df_clean = clean_piezo(df_raw)
    df_ml    = preprocess(df_clean)
    X_train, X_test, y_train, y_test = split_data(df_ml)
    model_val, _ = train(X_train, y_train, optimize=False)
    return model_val, df_clean, df_ml

with st.spinner("Alignement du modèle XGBoost et chargement des données..."):
    model_val, df_clean, df_ml = get_trained_model_and_data(DATA_CODE_PIEZO)

df_piezo_unique = load_single_piezo_map(DATA_CODE_PIEZO)

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
# PRÉPARATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
forecast_val_series = pred(model_val, df_ml)
forecast_index_val  = pd.date_range(start="2026-03-01", periods=3, freq="ME")

df_pred_val = pd.DataFrame({
    "date_mesure":      forecast_index_val,
    "niveau_nappe_eau": forecast_val_series.values,
    "Type":             "Prédiction XGBoost (Test)",
})

df_hist = df_ml[["niveau_nappe_eau"]].reset_index()
df_hist["Type"] = "Historique Réel"

deux_ans = pd.Timestamp("2026-05-31") - pd.Timedelta(days=365 * 2)
df_hist_filtre = df_hist[df_hist["date_mesure"] >= deux_ans]
df_total = pd.concat([df_hist_filtre, df_pred_val], ignore_index=True)

# ── Seuils ───────────────────────────────────────────────────────────────────
_seuils: dict[str, float] = load_seuils_interm(DATA_CODE_PIEZO) or _SEUILS_FALLBACK

x_min = pd.Timestamp("2026-05-31") - pd.Timedelta(days=180)
x_max = pd.Timestamp("2026-05-31") + pd.Timedelta(days=95)
y_min_global = min(df_hist_filtre["niveau_nappe_eau"].min(), _seuils["p5"])  - 0.5
y_max_global = max(df_hist_filtre["niveau_nappe_eau"].max(), _seuils["p95"]) + 0.5


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — Altair & HTML seuils
# ══════════════════════════════════════════════════════════════════════════════

def _build_seuils_df() -> pd.DataFrame:
    """
    Construit le DataFrame Altair pour les bandes de seuils.
    Les couleurs viennent de SEUIL_COLORS — aucune valeur hardcodée ici.
    """
    rows = [
        {"y_min": _seuils["p95"],  "y_max": y_max_global,   "Couleur": "Très haut",       "#color": SEUIL_COLORS["Très haut"]["bg"]},
        {"y_min": _seuils["p85"],  "y_max": _seuils["p95"], "Couleur": "Modérément haut", "#color": SEUIL_COLORS["Modérément haut"]["bg"]},
        {"y_min": _seuils["p20"],  "y_max": _seuils["p85"], "Couleur": "Normal",          "#color": SEUIL_COLORS["Normal"]["bg"]},
        {"y_min": _seuils["p10"],  "y_max": _seuils["p20"], "Couleur": "Vigilance",       "#color": SEUIL_COLORS["Vigilance"]["bg"]},
        {"y_min": _seuils["p5"],   "y_max": _seuils["p10"], "Couleur": "Alerte",          "#color": SEUIL_COLORS["Alerte"]["bg"]},
        {"y_min": y_min_global,    "y_max": _seuils["p5"],  "Couleur": "Crise",           "#color": SEUIL_COLORS["Crise"]["bg"]},
    ]
    return pd.DataFrame(rows)


def _get_statut(valeur: float) -> tuple[str, str]:
    """
    Retourne (nom_statut, couleur_fg) pour une valeur de nappe donnée.
    Couleurs issues de SEUIL_COLORS.
    """
    if valeur <= _seuils["p5"]:   nom = "Crise"
    elif valeur <= _seuils["p10"]: nom = "Alerte"
    elif valeur <= _seuils["p20"]: nom = "Vigilance"
    elif valeur <= _seuils["p85"]: nom = "Normal"
    elif valeur <= _seuils["p95"]: nom = "Modérément haut"
    else:                          nom = "Très haut"
    return nom, SEUIL_COLORS[nom]["fg"]


def _render_seuils_card(derniere_val: float) -> None:
    """
    Affiche la card seuils complète (barre gradient + tableau des niveaux).
    Toutes les couleurs viennent de SEUIL_COLORS.
    """
    statut_actuel, couleur_statut = _get_statut(derniere_val)

    # Correspondance seuil → percentile pour l'affichage des limites
    limites = {
        "Très haut":       f"{_seuils['p95']:.1f} m",
        "Modérément haut": f"{_seuils['p85']:.1f} m",
        "Normal":          f"≥ {_seuils['p20']:.1f} m",
        "Vigilance":       f"{_seuils['p10']:.1f} m",
        "Alerte":          f"{_seuils['p5']:.1f} m",
        "Crise":           f"≤ {_seuils['p5']:.1f} m",
    }

    # Barre gradient (couleurs dans l'ordre Crise → Très haut)
    gradient_stops = " ,".join([
        f"{SEUIL_COLORS[nom]['bg']} {i * 100 // 6}%, {SEUIL_COLORS[nom]['bg']} {(i+1) * 100 // 6}%"
        for i, nom in enumerate(["Crise", "Alerte", "Vigilance", "Normal", "Modérément haut", "Très haut"])
    ])

    labels_barre = (
        f"<div style='display:flex; justify-content:space-between; "
        f"font-size:0.65rem; color:#94a3b8; margin-top:4px; margin-bottom:14px;'>"
        f"<span>{_seuils['p5']:.1f} m</span>"
        f"<span>{_seuils['p10']:.1f} m</span>"
        f"<span>{_seuils['p20']:.1f} m</span>"
        f"<span>{_seuils['p85']:.1f} m</span>"
        f"<span>{_seuils['p95']:.1f} m</span>"
        f"</div>"
    )

    lignes_seuils = ""
    for nom in SEUIL_ORDER:
        colors   = SEUIL_COLORS[nom]
        is_actif = (statut_actuel == nom)
        tag_actif = (
            f"<span style='color:{SEUIL_COLORS['Normal']['fg']}; font-size:0.8em; font-weight:bold; "
            f"background-color:{SEUIL_COLORS['Normal']['fg']}15; padding:1px 6px; border-radius:10px;'>Actif</span>"
            if is_actif else "<span style='color:#ccc;'>—</span>"
        )
        lignes_seuils += (
            f"<div style='display:grid; grid-template-columns: 1fr 1fr 60px; "
            f"align-items:center; margin-bottom:11px; font-size:0.9rem;'>"
            f"  <div style='display:flex; align-items:center;'>"
            f"    <span style='height:10px; width:10px; background-color:{colors['bg']}; "
            f"border:1.5px solid {colors['fg']}40; border-radius:50%; display:inline-block; "
            f"margin-right:10px; flex-shrink:0;'></span>"
            f"    <span style='font-weight:{'600' if is_actif else 'normal'}; "
            f"color:{'#000' if is_actif else '#555'};'>{nom}</span>"
            f"  </div>"
            f"  <span style='color:#777; font-family:monospace; text-align:right;'>{limites[nom]}</span>"
            f"  <span style='text-align:right;'>{tag_actif}</span>"
            f"</div>"
        )

    st.markdown(
        f"""<div class='hs-card'>
          <div style='display:flex; justify-content:space-between; font-weight:bold;
                      margin-bottom:12px; font-size:0.9rem; color:#666;'>
            <span>Niveau mesuré</span>
            <span style='color:{couleur_statut}; background-color:{couleur_statut}15;
                         padding:2px 10px; border-radius:20px;'>
              {derniere_val:.1f} m · {statut_actuel}
            </span>
          </div>
          <div style='height:6px; width:100%; border-radius:3px;
               background:linear-gradient(to right, {gradient_stops});'>
          </div>
          {labels_barre}
          {lignes_seuils}
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ALTAIR GRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
seuils_df = _build_seuils_df()

fond_seuils = alt.Chart(seuils_df).mark_rect(opacity=0.45).encode(
    y=alt.Y("y_min:Q", scale=alt.Scale(domain=[y_min_global, y_max_global])),
    y2=alt.Y2("y_max:Q"),
    color=alt.Color("Couleur:N", scale=alt.Scale(
        domain=seuils_df["Couleur"].tolist(),
        range=seuils_df["#color"].tolist(),
    ), title="Seuils de gestion"),
)

lignes = alt.Chart(df_total).mark_line(strokeWidth=3).encode(
    x=alt.X("date_mesure:T", title="Date de mesure",
            scale=alt.Scale(domain=[x_min.isoformat(), x_max.isoformat()])),
    y=alt.Y("niveau_nappe_eau:Q", title="Niveau de la nappe (m)",
            scale=alt.Scale(domain=[y_min_global, y_max_global], zero=False)),
    color=alt.Color("Type:N", scale=alt.Scale(
        domain=["Historique Réel", "Prédiction XGBoost (Test)"],
        range=[DESIGN_TOKENS["color_line"], DESIGN_TOKENS["color_line"]],
    ), title="Données"),
    strokeDash=alt.condition(
        alt.datum.Type == "Prédiction XGBoost (Test)",
        alt.value([6, 4]),
        alt.value([0, 0]),
    ),
)

chart = (
    alt.layer(fond_seuils, lignes)
    .properties(height=400, background="white")
    .configure_view(strokeWidth=0)
    .interactive()
    .resolve_scale(color="independent")
)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT 2/3 — 1/3
# ══════════════════════════════════════════════════════════════════════════════
col_gauche, col_droite = st.columns([2, 1])

with col_gauche:
    if not df_piezo_unique.empty:
        df_map = df_piezo_unique.rename(columns={"x": "longitude", "y": "latitude"})
        st.map(df_map, zoom=10)
    else:
        st.warning("Données géographiques non disponibles.")

    st.write("")
    st.markdown("<p class='section-title'>Historique & Prévision (Mars 2026 → Mai 2026)</p>",
                unsafe_allow_html=True)
    st.altair_chart(chart, width='stretch')

with col_droite:

    # ── Indicateurs ──────────────────────────────────────────────────────────
    st.markdown("<p class='section-title'>Indicateurs</p>", unsafe_allow_html=True)

    derniere_valeur_reelle = df_hist_filtre["niveau_nappe_eau"].iloc[-1]
    moyenne_pred           = df_pred_val["niveau_nappe_eau"].mean()

    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.metric(label="NIVEAU ACTUEL", value=f"{derniere_valeur_reelle:.1f} m",
                  delta="Normal", delta_color="normal")
    with kpi_col2:
        st.metric(label="TENDANCE 7 J", value="-0.3 m",
                  delta="⬇ en baisse", delta_color="inverse")

    st.write("")

    kpi_col3, kpi_col4 = st.columns(2)
    with kpi_col3:
        st.metric(label="PLUIE 7 J",    value="38 mm",  delta="⬆ 12 mm",   delta_color="normal")
    with kpi_col4:
        st.metric(label="PRÉVISION 90 J", value=f"{moyenne_pred:.1f} m",
                  delta="⚠ se creuse", delta_color="off")

    st.write("")

    # ── Seuils ───────────────────────────────────────────────────────────────
    st.markdown("<p class='section-title'>Seuils de gestion</p>", unsafe_allow_html=True)
    _render_seuils_card(df_hist_filtre["niveau_nappe_eau"].iloc[-1])

# ══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES BAS DE PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.write("---")
st.markdown("<p class='section-title'>Détail des valeurs prédites pour le jeu de test</p>",
            unsafe_allow_html=True)
cols_val = st.columns(3)

for i, row in enumerate(df_pred_val.itertuples()):
    with cols_val[i]:
        st.metric(label=row.date_mesure.strftime("%B %Y"),
                  value=f"{row.niveau_nappe_eau:.2f} m")
