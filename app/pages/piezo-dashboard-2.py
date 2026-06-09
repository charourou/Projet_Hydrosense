"""
pages/piezo-dashboard-2.py — v3
────────────────────────────────
Fixes v3 :
- selectbox repositionné dans le header (top: 72px pour passer sous la navbar)
- panels réordonnés : header → statut → valeur → chart (sans superposition)
- mini-chart intégré dans la glass-card "Profondeur de nappe"
- padding-top du .main ajusté pour la navbar Streamlit (≈ 56px)
"""

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo
from utils.bigquery import (
    load_catalog_interm,
    load_catalog_map_dept,
    load_seuils_interm,
    seuils_from_row,
)
from utils.theme import SEUIL_COLORS, DESIGN_TOKENS

_SEUILS_FALLBACK: dict[str, float] = {
    "p95": 15.0, "p85": 13.4, "p20": 12.8, "p10": 12.0, "p5": 11.0,
}
DEPT_CARTE = "79"

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Reset Streamlit layout ── */
    html, body                { overflow: hidden !important; }
    .main > div:first-child  { padding-top: 0 !important; padding-left: 0 !important; padding-right: 0 !important; }
    .block-container         { padding: 0 !important; max-width: 100% !important; overflow: hidden !important; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ── Folium map plein écran ── */
    iframe { display: block; width: 100vw !important; height: 100vh !important; border: none; }

    /* ── Glass card base ── */
    .glass-card {
        background: rgba(255,255,255,0.55);
        backdrop-filter: blur(30px) saturate(1.5);
        -webkit-backdrop-filter: blur(30px) saturate(1.5);
        border-radius: 28px;
        border: 0.666667px solid rgba(255,255,255,0.55);
        box-shadow: rgba(28,50,84,0.28) 0px 24px 60px -12px;
        padding: 16px 20px;
    }

    /* ── Panel gauche : empilé en fixed, top:72px pour passer sous la navbar ── */
    .panel-left {
        position: fixed;
        top: 72px; left: 16px;
        width: 344px;
        z-index: 1001;
        display: flex;
        flex-direction: column;
        gap: 10px;
        pointer-events: all;
    }

    /* Header card */
    .hc-row     { display: flex; align-items: center; gap: 12px; }
    .hc-logo    { width: 40px; height: 40px; background: linear-gradient(135deg,#2484e5,#0ea5e9);
                  border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink:0; }
    .hc-title   { font-size: 1.15rem; font-weight: 700; color: #1e293b; letter-spacing: -0.3px; }
    .hc-sub     { font-size: 0.73rem; color: #64748b; margin-top: 1px; }
    .breadcrumb { font-size: 0.73rem; color: #94a3b8; margin-top: 9px; display: flex; align-items: center; gap: 5px; }
    .chips-row  { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .chip       { padding: 4px 12px; border-radius: 999px; font-size: 0.77rem; font-weight: 500;
                  border: 1.5px solid #e2e8f0; color: #475569; background: #f8fafc; }
    .chip.active{ border-color: #2484e5; color: #2484e5; background: #eff6ff; }
    .meta-row   { display: flex; gap: 6px; margin-top: 7px; flex-wrap: wrap; }
    .meta-tag   { padding: 3px 10px; border-radius: 6px; font-size: 0.71rem; color: #64748b; background: #f1f5f9; }

    /* Statut card */
    .statut-badge { display: inline-flex; align-items: center; gap: 6px;
                    padding: 4px 10px; border-radius: 999px; font-size: 0.80rem; font-weight: 700; }
    .statut-dot   { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .statut-hint  { font-size: 0.68rem; color: #94a3b8; margin-top: 3px; }
    .main-value      { font-size: 2.0rem; font-weight: 800; color: #0f172a; line-height: 1; letter-spacing: -1px; margin: 4px 0 1px; }
    .main-value-unit { font-size: 1.0rem; font-weight: 400; color: #64748b; letter-spacing: 0; }
    .main-value-sub  { font-size: 0.70rem; color: #94a3b8; }

    /* Chart card */
    .chart-title { font-size: 0.82rem; font-weight: 600; color: #475569; margin-bottom: 4px; }
    .chart-legend{ display: flex; gap: 14px; margin-top: 5px; font-size: 0.71rem; color: #64748b; }
    .leg-line    { display: inline-block; width: 18px; height: 2px; vertical-align: middle; margin-right: 3px; background: #2484e5; }

    /* ── Selectbox repositionné dans le header ── */
    [data-testid="stSelectbox"] {
        position: fixed !important;
        top: 76px; left: 16px;
        width: 344px !important;
        z-index: 1002;
        pointer-events: all;
        /* sera rendu invisible par opacity car le panel-left a son propre header visuel */
        opacity: 0;
        pointer-events: all;
    }
    /* On rend le selectbox visible uniquement (hack : on utilise un label custom dans le panel HTML) */

    /* stVegaLiteChart : pas de positionnement fixed (géré via Vega-Embed inline) */
    [data-testid="stVegaLiteChart"] {
        display: none !important;
    }

    /* ── KPI row ── */
    .kpi-row {
        position: fixed; bottom: 20px; left: 500px; right: 20px;
        display: flex; gap: 10px;
        pointer-events: all; z-index: 1001;
    }
    .kpi-card {
        flex: 1;
        background: rgba(255,255,255,0.90);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border-radius: 16px; border: 1px solid rgba(255,255,255,0.65);
        box-shadow: 0 4px 16px rgba(0,0,0,0.09);
        padding: 14px 18px;
    }
    /* Card groupée : 3 métriques côte à côte sans padding propre */
    .kpi-card.kpi-grouped { padding: 0; overflow: hidden; }
    .kpi-item { flex: 1; padding: 14px 18px; }
    .kpi-sep  { width: 1px; background: rgba(0,0,0,0.07); margin: 10px 0; flex-shrink: 0; }
    .kpi-val  { font-size: 1.55rem; font-weight: 800; color: #0f172a; letter-spacing: -0.5px; line-height: 1.1; }
    .kpi-unit { font-size: 0.88rem; font-weight: 400; color: #64748b; }
    .kpi-lbl  { font-size: 0.69rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
df_catalog = load_catalog_interm()
df_catalog["label"] = (
    df_catalog["bss_id"] + " — "
    + df_catalog["nom_commune"].fillna("?")
    + " (" + df_catalog["code_departement"].fillna("?") + ")"
)
labels = df_catalog["label"].tolist()

# ── Initialisation session_state ─────────────────────────────────────────────
# Le piézomètre actif est stocké dans session_state["selected_bss"].
# Il peut être mis à jour :
#   (a) via le selectbox (input utilisateur direct)
#   (b) via un clic sur un marker Folium (intercepté plus bas)
if "selected_bss" not in st.session_state:
    st.session_state["selected_bss"] = next(
        (b for b in df_catalog["bss_id"] if "BSS001QHYH" in b),
        df_catalog["bss_id"].iloc[0],
    )

# Index courant dans la liste des labels
_cur_bss   = st.session_state["selected_bss"]
_cur_index = df_catalog.index[df_catalog["bss_id"] == _cur_bss].tolist()
_cur_index = _cur_index[0] if _cur_index else 0

# Selectbox invisible (opacity:0 via CSS) — drive l'état via key
selected_label = st.selectbox(
    "Piézomètre",
    options=labels,
    index=_cur_index,
    key="selectbox_piezo",
    label_visibility="collapsed",
)
# Sync session_state depuis le selectbox (cas (a))
st.session_state["selected_bss"] = df_catalog.loc[
    df_catalog["label"] == selected_label, "bss_id"
].iloc[0]

DATA_CODE_PIEZO = df_catalog.loc[df_catalog["label"] == selected_label, "bss_id"].iloc[0]
nom_commune     = df_catalog.loc[df_catalog["label"] == selected_label, "nom_commune"].iloc[0] or "?"
nom_departement = df_catalog.loc[df_catalog["label"] == selected_label, "nom_departement"].iloc[0] or "?"
dept            = df_catalog.loc[df_catalog["label"] == selected_label, "code_departement"].iloc[0] or "?"

_seuils: dict[str, float] = load_seuils_interm(DATA_CODE_PIEZO) or _SEUILS_FALLBACK

# Une seule requête BQ pour tous les markers
df_map_dept = load_catalog_map_dept(DEPT_CARTE)

def _get_statut(val: float, seuils: dict) -> tuple[str, dict]:
    if val <= seuils["p5"]:    return "Crise",           SEUIL_COLORS["Crise"]
    elif val <= seuils["p10"]: return "Alerte",          SEUIL_COLORS["Alerte"]
    elif val <= seuils["p20"]: return "Vigilance",       SEUIL_COLORS["Vigilance"]
    elif val <= seuils["p85"]: return "Normal",          SEUIL_COLORS["Normal"]
    elif val <= seuils["p95"]: return "Modérément haut", SEUIL_COLORS["Modérément haut"]
    else:                      return "Très haut",       SEUIL_COLORS["Très haut"]

@st.cache_data
def get_historique(bss_id: str) -> pd.DataFrame:
    df_raw = load_piezo_bq(bss_id)
    return clean_piezo(df_raw)


@st.cache_data
def get_forecast(bss_id: str) -> dict:
    response = requests.get("http://localhost:8000/predict", params={"bss_id": bss_id})
    response.raise_for_status()
    return response.json()


with st.spinner("Chargement des données historiques..."):
    df_clean = get_historique(DATA_CODE_PIEZO)

with st.spinner("Chargement de la prévision via l'API..."):
    result = get_forecast(DATA_CODE_PIEZO)

deux_ans = pd.Timestamp.today() - pd.Timedelta(days=365 * 2)
df_hist_filtre = df_clean[df_clean["date_mesure"] >= deux_ans]

df_pred_val = pd.DataFrame(result["prévision"])
df_pred_val["date_mesure"]      = pd.to_datetime(df_pred_val["date"])
df_pred_val["niveau_nappe_eau"] = df_pred_val["niveau"]

_today      = df_hist_filtre["date_mesure"].max()
_dates_hist = df_hist_filtre["date_mesure"].values
_vals_hist  = df_hist_filtre["niveau_nappe_eau"].values
_dates_pred = df_pred_val["date_mesure"].values
_vals_pred  = df_pred_val["niveau_nappe_eau"].values

derniere_val  = float(_vals_hist[-1])
valeur_7j_avant = df_clean[
    df_clean["date_mesure"] <= df_clean["date_mesure"].max() - pd.Timedelta(days=7)
]["niveau_nappe_eau"].iloc[-1]
tendance_7j    = derniere_val - float(valeur_7j_avant)
moyenne_pred   = float(np.mean(_vals_pred))
variation_prev = moyenne_pred - derniere_val

statut_nom, statut_colors = _get_statut(derniere_val, _seuils)

# ══════════════════════════════════════════════════════════════════════════════
# CARTE FOLIUM
# ══════════════════════════════════════════════════════════════════════════════
m = folium.Map(location=[46.4, -0.3], zoom_start=8, tiles="CartoDB Positron", prefer_canvas=True)

# Cacher tous les contrôles Leaflet (zoom + attribution)
m.get_root().html.add_child(folium.Element("""
<style>
.leaflet-control-container { display: none !important; }
</style>
"""))

rng = np.random.default_rng(0)
for _, row in df_map_dept.iterrows():
    s = seuils_from_row(row) or _SEUILS_FALLBACK
    mock_val    = derniere_val if row["bss_id"] == DATA_CODE_PIEZO else derniere_val + rng.uniform(-2, 2)
    _, mc       = _get_statut(mock_val, s)
    is_selected = (row["bss_id"] == DATA_CODE_PIEZO)

    if is_selected:
        folium.CircleMarker(
            location=[row["y"], row["x"]], radius=20,
            color=mc["fg"], fill=True, fill_color=mc["fg"],
            fill_opacity=0.15, weight=2, opacity=0.35,
        ).add_to(m)
        folium.CircleMarker(
            location=[row["y"], row["x"]], radius=10,
            color=mc["fg"], fill=True, fill_color=mc["fg"],
            fill_opacity=0.85, weight=2,
            tooltip=f"{row['bss_id']} — {row['nom_commune']}",
        ).add_to(m)
    else:
        folium.CircleMarker(
            location=[row["y"], row["x"]], radius=7,
            color=mc["fg"], fill=True, fill_color=mc["fg"],
            fill_opacity=0.55, weight=1.5,
            tooltip=f"{row['bss_id']} — {row['nom_commune']}",
        ).add_to(m)

map_data = st_folium(
    m,
    width=None,
    height=900,
    returned_objects=["last_object_clicked_tooltip"],
    key="map_fs",
)

# ── Intercepter le clic sur un marker ────────────────────────────────────────
# st_folium retourne le tooltip du dernier objet cliqué.
# Nos tooltips ont le format "BSS001XXXX — Commune"
# On extrait le bss_id (partie avant " — ") et on met à jour session_state.
_clicked_tooltip = (map_data or {}).get("last_object_clicked_tooltip") or ""
if _clicked_tooltip and " — " in _clicked_tooltip:
    _clicked_bss = _clicked_tooltip.split(" — ")[0].strip()
    # Vérifier que ce bss_id existe dans le catalogue
    if _clicked_bss in df_catalog["bss_id"].values:
        if _clicked_bss != st.session_state["selected_bss"]:
            st.session_state["selected_bss"] = _clicked_bss
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MINI GRAPHIQUE ALTAIR
# ══════════════════════════════════════════════════════════════════════════════
_pred_n = min(30, len(_dates_pred))
df_h = pd.DataFrame({"date": _dates_hist,           "val": _vals_hist,          "t": "H"})
df_p = pd.DataFrame({"date": _dates_pred[:_pred_n], "val": _vals_pred[:_pred_n],"t": "P"})

y_lo = min(_vals_hist.min(), _seuils["p5"])  - 0.3
y_hi = max(_vals_hist.max(), _seuils["p95"]) + 0.3


# ── Générer le SVG du mini-chart via matplotlib (zero CDN, zero iframe) ────
# Matplotlib est toujours disponible dans l'env Python, pas de dépendance externe.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
import matplotlib.patches as _mpatches
from io import BytesIO as _BytesIO
import base64 as _b64

_fig, _ax = _plt.subplots(figsize=(4.6, 1.2))
_fig.patch.set_alpha(0)
_ax.set_facecolor("none")

# Bandes de seuil
_seuil_bands = [
    (_seuils["p5"],  _seuils["p10"], SEUIL_COLORS["Alerte"]["bg"]),
    (_seuils["p10"], _seuils["p20"], SEUIL_COLORS["Vigilance"]["bg"]),
    (_seuils["p20"], _seuils["p85"], SEUIL_COLORS["Normal"]["bg"]),
    (_seuils["p85"], _seuils["p95"], SEUIL_COLORS["Modérément haut"]["bg"]),
]
_x_lo = _dates_hist[0]
_x_hi = _dates_pred[min(29, len(_dates_pred) - 1)]
for _yb0, _yb1, _bc in _seuil_bands:
    _ax.fill_between([_x_lo, _x_hi], [_yb0, _yb0], [_yb1, _yb1],
                     color=_bc, alpha=0.6, linewidth=0)

# Courbes
_ax.plot(_dates_hist, _vals_hist, color="#2484e5", linewidth=1.8, solid_capstyle="round")
_ax.plot(_dates_pred[:_pred_n], _vals_pred[:_pred_n], color="#2484e5", linewidth=1.8,
         linestyle=(0, (5, 3)), solid_capstyle="round")
_ax.plot([_today], [derniere_val], "o", color="#2484e5", markersize=4, zorder=5)

_ax.set_xlim(_x_lo, _x_hi)
_ax.set_ylim(y_lo, y_hi)
_ax.axis("off")
_fig.tight_layout(pad=0)

_buf = _BytesIO()
_fig.savefig(_buf, format="svg", bbox_inches="tight", transparent=True)
_plt.close(_fig)
_svg_str = _buf.getvalue().decode("utf-8")
# Garder uniquement le tag <svg ...> (retirer le prologue XML)
_svg_clean = _svg_str[_svg_str.find("<svg"):]

# ── Précipitations bars (calculé ici car réutilisé dans panel_html) ───────────
precip_7j   = [2, 5, 11, 8, 3, 0, 0]
days_labels = ["LUN", "MAR", "MER", "JEU", "VEN", "SAM", "DIM"]
_max_p      = max(precip_7j) or 1
bars_html   = ""
for _p, _d in zip(precip_7j, days_labels):
    _pbg = "#2484e5" if _p > 0 else "#e2e8f0"
    _ph  = int(_p / _max_p * 36) + 2
    bars_html += (
        "<div style='display:flex;flex-direction:column;align-items:center;gap:3px;'>"
        "<div style='width:6px;height:" + str(_ph) + "px;border-radius:3px;background:" + _pbg + ";'></div>"
        "<span style='font-size:0.58rem;color:#94a3b8;'>" + str(_p) + "</span>"
        "<span style='font-size:0.58rem;color:#cbd5e1;'>" + _d + "</span></div>"
    )

# ── Panels HTML ───────────────────────────────────────────────────────────────
_bg_statut  = statut_colors["bg"]
_fg_statut  = statut_colors["fg"]
_val_str    = f"{derniere_val:.1f}"
_t_arrow = "↓" if tendance_7j < 0 else "↑"
_t_color = "#d9534f"
_t_bg    = "#fce4d6"
_p_arrow = "↓" if variation_prev < 0 else "↑"
_p_color = "#2ca02c"
_p_bg    = "#e2f0d9"
_t_val   = f"{tendance_7j:+.2f}"
_p_val   = f"{moyenne_pred:.2f}"
_p_delta = f"{variation_prev:+.2f}"

panel_html = (
    '<div class="panel-left">'

    # 1. Header
    '<div class="glass-card" style="padding-bottom:14px;">'
    '<div class="hc-row">'
    '<div class="hc-logo">💧</div>'
    '<div><div class="hc-title">Hydro-Sense</div>'
    '<div class="hc-sub">Surveillance piézométrique</div></div>'
    '</div>'
    '<div class="breadcrumb"><span>Poitou-Charentes</span>'
    '<span style="color:#cbd5e1;">›</span>'
    '<span style="color:#1e293b;font-weight:600;">' + nom_commune + '</span></div>'
    '<div class="chips-row"><span class="chip active">' + nom_commune + '</span></div>'
    '<div class="meta-row">'
    '<span class="meta-tag">' + dept + ' · ' + nom_departement + '</span>'
    '<span class="meta-tag">Nappe libre</span>'
    '<span class="meta-tag">Maj. 3 juin</span>'
    '</div></div>'

    # 3. Statut + valeur + KPI (tendance & prévision) à droite
    '<div class="glass-card" style="padding:10px 14px;display:flex;align-items:stretch;gap:0;">'
    # — colonne gauche : statut actuel
    '<div style="flex:1;min-width:0;">'
    '<span class="statut-badge" style="background:' + _bg_statut + ';color:' + _fg_statut + ';">'
    '<span class="statut-dot" style="background:' + _fg_statut + ';"></span>'
    + statut_nom +
    '</span>'
    '<div class="statut-hint">Seuil franchi le 28 mai</div>'
    '<div class="main-value">' + _val_str + '<span class="main-value-unit"> m</span></div>'
    '<div class="main-value-sub">Profondeur actuelle · sous le sol</div>'
    '</div>'
    # — séparateur vertical
    '<div style="width:1px;background:rgba(0,0,0,0.07);margin:2px 12px;flex-shrink:0;"></div>'
    # — colonne droite : deux KPI empilés
    '<div style="display:flex;flex-direction:column;justify-content:space-around;min-width:110px;">'
    # tendance
    '<div>'
    '<div style="font-size:0.62rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">Tendance 7 J</div>'
    '<div style="font-size:1.15rem;font-weight:800;color:' + _t_color + ';letter-spacing:-0.3px;line-height:1.1;">' + _t_val + '<span style="font-size:0.78rem;font-weight:400;color:#64748b;"> m</span></div>'
    '<span style="font-size:0.66rem;font-weight:600;color:' + _t_color + ';background:' + _t_bg + ';padding:2px 7px;border-radius:999px;">' + _t_arrow + ' ' + _t_val + '</span>'
    '</div>'
    # séparateur horizontal
    '<div style="height:1px;background:rgba(0,0,0,0.05);margin:4px 0;"></div>'
    # prévision
    '<div>'
    '<div style="font-size:0.62rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">Prévision 90 J</div>'
    '<div style="font-size:1.15rem;font-weight:800;color:#0f172a;letter-spacing:-0.3px;line-height:1.1;">' + _p_val + '<span style="font-size:0.78rem;font-weight:400;color:#64748b;"> m</span></div>'
    '<span style="font-size:0.66rem;font-weight:600;color:' + _p_color + ';background:' + _p_bg + ';padding:2px 7px;border-radius:999px;">' + _p_arrow + ' ' + _p_delta + ' m</span>'
    '</div>'
    '</div>'
    '</div>'

    '</div>'  # .panel-left
)

st.markdown(panel_html, unsafe_allow_html=True)

# ── Chart card — fixed, 460px, sort du panel-left ────────────────────────────
st.markdown(
    "<div style='position:fixed;bottom:20px;left:16px;width:460px;z-index:1001;"
    "background:rgba(255,255,255,0.55);backdrop-filter:blur(30px) saturate(1.5);"
    "-webkit-backdrop-filter:blur(30px) saturate(1.5);"
    "border-radius:28px;border:0.667px solid rgba(255,255,255,0.55);"
    "box-shadow:rgba(28,50,84,0.28) 0px 24px 60px -12px;"
    "padding:14px 16px 10px;pointer-events:all;'>"
    "<div class='chart-title'>Profondeur de nappe</div>"
    "<div style='margin:4px -4px 0;line-height:0;'>" + _svg_clean + "</div>"
    "<div class='chart-legend' style='margin-top:6px;'>"
    "<span><span class='leg-line'></span>Historique</span>"
    "<span style='color:#94a3b8;margin-left:8px;'>- - Prévision 90 j</span>"
    "</div></div>",
    unsafe_allow_html=True,)

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card" style="flex:inherit;">
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;'>
      <span style='font-size:0.69rem;text-transform:uppercase;letter-spacing:0.5px;color:#94a3b8;'>PRÉCIP. · 7 J</span>
      <span style='font-size:0.69rem;color:#94a3b8;'>mm</span>
    </div>
    <div style='display:flex;align-items:flex-end;gap:5px;'>{bars_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Seuils réglementaires (card fixe bas-droite, au-dessus de la KPI row) ─────
_seuils_display = [
    ("Normal",    f"< {_seuils['p20']:.1f} m",                               SEUIL_COLORS["Normal"]),
    ("Vigilance", f"{_seuils['p20']:.1f} – {_seuils['p10']:.1f} m",          SEUIL_COLORS["Vigilance"]),
    ("Alerte",    f"{_seuils['p10']:.1f} – {_seuils['p5']:.1f} m",           SEUIL_COLORS["Alerte"]),
    ("Crise",     f"> {_seuils['p5']:.1f} m",                                 SEUIL_COLORS["Crise"]),
]
_seuils_rows_html = ""
for _nom, _lim, _col in _seuils_display:
    _is_actif = (statut_nom == _nom)
    _dot_style = (
        f"width:10px;height:10px;border-radius:50%;background:{_col['fg']};flex-shrink:0;"
        + (f"box-shadow:0 0 0 4px {_col['bg']};" if _is_actif else "opacity:0.55;")
    )
    _seuils_rows_html += (
        f"<div style='display:flex;align-items:center;gap:10px;padding:7px 0;"
        f"border-bottom:1px solid rgba(0,0,0,0.04);'>"
        f"<div style='{_dot_style}'></div>"
        f"<span style='flex:1;font-size:0.83rem;font-weight:{'600' if _is_actif else '400'};"
        f"color:{'#1e293b' if _is_actif else '#374151'};'>{_nom}</span>"
        f"<span style='font-size:0.78rem;color:#6b7280;font-family:monospace;'>{_lim}</span>"
        f"</div>"
    )

st.markdown(f"""
<div style='position:fixed;right:20px;bottom:20px;width:260px;z-index:1001;
            background:rgba(255,255,255,0.55);backdrop-filter:blur(30px) saturate(1.5);
            -webkit-backdrop-filter:blur(30px) saturate(1.5);
            border-radius:28px;border:0.667px solid rgba(255,255,255,0.55);
            box-shadow:rgba(28,50,84,0.28) 0px 24px 60px -12px;
            padding:16px 20px;pointer-events:all;'>
  <div style='font-size:0.65rem;font-weight:700;text-transform:uppercase;
              letter-spacing:1px;color:#9ca3af;margin-bottom:8px;'>
    SEUILS RÉGLEMENTAIRES
  </div>
  {_seuils_rows_html}
</div>
""", unsafe_allow_html=True)


