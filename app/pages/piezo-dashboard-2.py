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
import streamlit as st
from streamlit_folium import st_folium

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
    .main > div:first-child  { padding-top: 0 !important; padding-left: 0 !important; padding-right: 0 !important; }
    .block-container         { padding: 0 !important; max-width: 100% !important; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ── Folium map plein écran ── */
    iframe { display: block; width: 100vw !important; height: 100vh !important; border: none; }

    /* ── Glass card base ── */
    .glass-card {
        background: rgba(255,255,255,0.90);
        backdrop-filter: blur(18px) saturate(180%);
        -webkit-backdrop-filter: blur(18px) saturate(180%);
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.65);
        box-shadow: 0 8px 32px rgba(0,0,0,0.11), 0 2px 6px rgba(0,0,0,0.05);
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
    .statut-badge { display: inline-flex; align-items: center; gap: 7px;
                    padding: 6px 14px; border-radius: 999px; font-size: 0.88rem; font-weight: 700; }
    .statut-dot   { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .statut-hint  { font-size: 0.71rem; color: #94a3b8; margin-top: 5px; }
    .main-value      { font-size: 3.2rem; font-weight: 800; color: #0f172a; line-height: 1; letter-spacing: -2px; margin: 6px 0 2px; }
    .main-value-unit { font-size: 1.3rem; font-weight: 400; color: #64748b; letter-spacing: 0; }
    .main-value-sub  { font-size: 0.76rem; color: #94a3b8; }

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
        position: fixed; bottom: 20px; left: 376px; right: 20px;
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

# Données mock (remplacer par vraies données en prod)
_today      = pd.Timestamp("2026-06-03")
_dates_hist = pd.date_range(end=_today, periods=60, freq="D")
np.random.seed(42)
_vals_hist  = 12.4 + np.cumsum(np.random.randn(60) * 0.05)
_dates_pred = pd.date_range(start=_today + pd.Timedelta(days=1), periods=90, freq="D")
_vals_pred  = _vals_hist[-1] + np.cumsum(np.random.randn(90) * 0.04) + 0.3

derniere_val      = float(_vals_hist[-1])
tendance_7j       = float(_vals_hist[-1] - _vals_hist[-8])
jours_avant_seuil = 14

statut_nom, statut_colors = _get_statut(derniere_val, _seuils)

# ══════════════════════════════════════════════════════════════════════════════
# CARTE FOLIUM
# ══════════════════════════════════════════════════════════════════════════════
m = folium.Map(location=[46.4, -0.3], zoom_start=8, tiles="CartoDB Positron", prefer_canvas=True)

rng = np.random.default_rng(0)
for _, row in df_map_dept.iterrows():
    s = seuils_from_row(row) or _SEUILS_FALLBACK
    mock_val    = derniere_val if row["bss_id"] == DATA_CODE_PIEZO else derniere_val + rng.uniform(-2, 2)
    _, mc       = _get_statut(mock_val, s)
    is_selected = (row["bss_id"] == DATA_CODE_PIEZO)

    # Pré-calcul hors f-string (Python 3.10 n'accepte pas les backslashes dans les f-strings)
    _bss      = row["bss_id"]
    _commune  = row["nom_commune"]
    _items = [
        ("Normal",    "< " + f"{s['p20']:.1f}" + " m"),
        ("Vigilance", f"{s['p20']:.1f}" + " – " + f"{s['p10']:.1f}" + " m"),
        ("Alerte",    f"{s['p10']:.1f}" + " – " + f"{s['p5']:.1f}"  + " m"),
        ("Crise",     "> " + f"{s['p5']:.1f}"  + " m"),
    ]
    _rows = "".join(
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;font-size:12px;'>"
        "<div style='width:4px;height:16px;border-radius:2px;background:"
        + SEUIL_COLORS[n]["fg"] +
        ";flex-shrink:0;'></div>"
        "<span style='flex:1;color:#374151;font-weight:500;'>" + n + "</span>"
        "<span style='color:#6b7280;font-family:monospace;'>" + lim + "</span></div>"
        for n, lim in _items
    )
    popup_html = (
        "<div style='font-family:system-ui,sans-serif;padding:4px;min-width:220px;'>"
        "<div style='font-size:11px;font-weight:600;color:#1e293b;margin-bottom:6px;'>"
        + _bss + " · " + _commune + "</div>"
        "<div style='font-size:10px;text-transform:uppercase;letter-spacing:1px;"
        "color:#9ca3af;margin-bottom:8px;'>SEUILS RÉGLEMENTAIRES</div>"
        + _rows +
        "</div>"
    )

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
            popup=folium.Popup(folium.IFrame(popup_html, width=260, height=180), max_width=280),
            tooltip=f"{row['bss_id']} — {row['nom_commune']}",
        ).add_to(m)
    else:
        folium.CircleMarker(
            location=[row["y"], row["x"]], radius=7,
            color=mc["fg"], fill=True, fill_color=mc["fg"],
            fill_opacity=0.55, weight=1.5,
            popup=folium.Popup(folium.IFrame(popup_html, width=260, height=180), max_width=280),
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
df_h = pd.DataFrame({"date": _dates_hist,      "val": _vals_hist,   "t": "H"})
df_p = pd.DataFrame({"date": _dates_pred[:30], "val": _vals_pred[:30], "t": "P"})

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

_fig, _ax = _plt.subplots(figsize=(3.4, 1.2))
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
_x_hi = _dates_pred[29]
for _yb0, _yb1, _bc in _seuil_bands:
    _ax.fill_between([_x_lo, _x_hi], [_yb0, _yb0], [_yb1, _yb1],
                     color=_bc, alpha=0.6, linewidth=0)

# Courbes
_ax.plot(_dates_hist, _vals_hist, color="#2484e5", linewidth=1.8, solid_capstyle="round")
_ax.plot(_dates_pred[:30], _vals_pred[:30], color="#2484e5", linewidth=1.8,
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

# ── Panels HTML ───────────────────────────────────────────────────────────────
_bg_statut  = statut_colors["bg"]
_fg_statut  = statut_colors["fg"]
_val_str    = f"{derniere_val:.1f}"

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
    '<span class="meta-tag">' + dept + ' · ' + nom_commune + '</span>'
    '<span class="meta-tag">Nappe libre</span>'
    '<span class="meta-tag">Maj. 3 juin</span>'
    '</div></div>'

    # 2. Placeholder selectbox
    '<div style="height:44px;"></div>'

    # 3. Statut + valeur
    '<div class="glass-card">'
    '<span class="statut-badge" style="background:' + _bg_statut + ';color:' + _fg_statut + ';">'
    '<span class="statut-dot" style="background:' + _fg_statut + ';"></span>'
    + statut_nom +
    '</span>'
    '<div class="statut-hint">Seuil franchi le 28 mai</div>'
    '<div class="main-value">' + _val_str + '<span class="main-value-unit"> m</span></div>'
    '<div class="main-value-sub">Profondeur actuelle · sous le sol</div>'
    '</div>'

    # 4. Card graphique — SVG matplotlib inline
    '<div class="glass-card" style="padding:14px 16px 8px;">'
    '<div class="chart-title">Profondeur de nappe</div>'
    '<div style="margin:4px -4px 0; line-height:0;">' + _svg_clean + '</div>'
    '<div class="chart-legend" style="margin-top:6px;">'
    '<span><span class="leg-line"></span>Historique</span>'
    '<span style="color:#94a3b8;margin-left:8px;">- - Prévision 90 j</span>'
    '</div></div>'

    '</div>'  # .panel-left
)

st.markdown(panel_html, unsafe_allow_html=True)


# ── KPI cards ─────────────────────────────────────────────────────────────────
precip_7j   = [2, 5, 11, 8, 3, 0, 0]
days_labels = ["LUN", "MAR", "MER", "JEU", "VEN", "SAM", "DIM"]
max_p       = max(precip_7j) or 1
bars_html = ""
for p, d in zip(precip_7j, days_labels):
    _bg = "#2484e5" if p > 0 else "#e2e8f0"
    _h  = int(p / max_p * 36) + 2
    bars_html += (
        "<div style='display:flex;flex-direction:column;align-items:center;gap:3px;'>"
        "<div style='width:6px;height:" + str(_h) + "px;border-radius:3px;"
        "background:" + _bg + ";'></div>"
        "<span style='font-size:0.58rem;color:#94a3b8;'>" + str(p) + "</span>"
        "<span style='font-size:0.58rem;color:#cbd5e1;'>" + d + "</span></div>"
    )

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-val">↓ {abs(tendance_7j):.1f}<span class="kpi-unit"> m</span></div>
    <div class="kpi-lbl">Tendance / mois</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val" style="color:#2484e5;">J - {jours_avant_seuil}</div>
    <div class="kpi-lbl">Avant seuil</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val" style="color:#2484e5;">82<span class="kpi-unit"> %</span></div>
    <div class="kpi-lbl">Confiance modèle</div>
  </div>
  <div class="kpi-card" style="flex:1.4;">
    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;'>
      <span style='font-size:0.69rem;text-transform:uppercase;letter-spacing:0.5px;color:#94a3b8;'>PRÉCIP. · 7 J</span>
      <span style='font-size:0.69rem;color:#94a3b8;'>mm</span>
    </div>
    <div style='display:flex;align-items:flex-end;gap:5px;'>{bars_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)