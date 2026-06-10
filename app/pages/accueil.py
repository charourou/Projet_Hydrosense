import streamlit as st
from utils.api_client import load_catalog_interm
from utils.theme import DESIGN_TOKENS

t = DESIGN_TOKENS

st.markdown(f"""
<style>
    .hero {{
        background: linear-gradient(135deg, #1e3a5f 0%, #2484e5 100%);
        border-radius: 20px;
        padding: 40px 36px 36px;
        margin-bottom: 28px;
        color: white;
    }}
    .hero-title {{
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0 0 8px;
    }}
    .hero-sub {{
        font-size: 1.0rem;
        opacity: 0.85;
        max-width: 560px;
        line-height: 1.55;
        margin: 0 0 20px;
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 999px;
        padding: 5px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }}
    .kpi-strip {{
        display: flex;
        gap: 14px;
        margin-bottom: 28px;
    }}
    .kpi-box {{
        flex: 1;
        background: {t['bg_card']};
        border-radius: {t['radius_card']};
        border: 1px solid {t['border_card']};
        box-shadow: {t['shadow_card']};
        padding: 18px 20px;
    }}
    .kpi-box .val {{
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.5px;
        line-height: 1;
    }}
    .kpi-box .lbl {{
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 5px;
    }}
    .nav-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-bottom: 28px;
    }}
    .nav-card {{
        background: {t['bg_card']};
        border-radius: {t['radius_card']};
        border: 1px solid {t['border_card']};
        box-shadow: {t['shadow_card']};
        padding: 22px 20px;
        cursor: pointer;
        transition: box-shadow 0.15s;
    }}
    .nav-card:hover {{ box-shadow: 0 8px 28px rgba(36,132,229,0.15); }}
    .nav-icon  {{ font-size: 1.8rem; margin-bottom: 10px; }}
    .nav-title {{ font-size: 1.0rem; font-weight: 700; color: #1e293b; margin-bottom: 5px; }}
    .nav-desc  {{ font-size: 0.78rem; color: #64748b; line-height: 1.45; }}
    .footer {{
        font-size: 0.72rem;
        color: #94a3b8;
        text-align: center;
        padding-top: 8px;
    }}
</style>

<div class="hero">
    <div class="hero-title">💧 Hydro-Sense</div>
    <div class="hero-sub">
        Outil de surveillance et de prévision du risque de sécheresse
        sur les aquifères de Vendée et du Poitou-Charentes.
    </div>
    <span class="hero-badge">🎓 Le Wagon · Data Science &amp; AI · Nantes 2026</span>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
with st.spinner(""):
    df = load_catalog_interm()

nb_piezo = len(df)
nb_dept  = df["code_departement"].nunique()
nb_comm  = df["nom_commune"].nunique()

_fmt = lambda n: f"{n:,}".replace(",", " ")

st.markdown(f"""
<div class="kpi-strip">
    <div class="kpi-box">
        <div class="val">{_fmt(nb_piezo)}</div>
        <div class="lbl">Piézomètres</div>
    </div>
    <div class="kpi-box">
        <div class="val">{nb_dept}</div>
        <div class="lbl">Départements</div>
    </div>
    <div class="kpi-box">
        <div class="val">{_fmt(nb_comm)}</div>
        <div class="lbl">Communes</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Navigation cards ───────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-grid">
    <div class="nav-card">
        <div class="nav-icon">📊</div>
        <div class="nav-title">Dashboard piézomètre</div>
        <div class="nav-desc">Séries temporelles, niveaux actuels, prévisions à 90 jours et seuils réglementaires.</div>
    </div>
    <div class="nav-card">
        <div class="nav-icon">🗺️</div>
        <div class="nav-title">Carte des piézos</div>
        <div class="nav-desc">Visualisation géographique de toutes les stations de mesure sur la France.</div>
    </div>
    <div class="nav-card">
        <div class="nav-icon">📋</div>
        <div class="nav-title">Catalogue</div>
        <div class="nav-desc">Métadonnées des piézomètres : localisation, département, altitude, dates de mesure.</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="footer">Données issues du réseau ADES — Agences de l\'eau & BRGM</div>', unsafe_allow_html=True)
