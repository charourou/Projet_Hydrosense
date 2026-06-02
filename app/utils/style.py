import streamlit as st
from datetime import datetime


def apply_hydrosense_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    :root {
      --bg: #F5F5F7; --bg-elevated: #FFFFFF; --bg-sunken: #EFEFF2; --bg-hover: #F0F0F3;
      --fg1: #1D1D1F; --fg2: #424245; --fg3: #6E6E73; --fg4: #86868B;
      --hairline: #D2D2D7; --hairline-soft: #E8E8ED;
      --blue: #0071E3; --blue-tint: rgba(0,113,227,0.10);
      --alert-1: #5E8C73; --alert-2: #C99A3C; --alert-3: #C16C3A; --alert-4: #B23B3B;
      --alert-1-soft: rgba(94,140,115,0.12);
      --alert-2-soft: rgba(201,154,60,0.14);
      --alert-3-soft: rgba(193,108,58,0.14);
      --alert-4-soft: rgba(178,59,59,0.13);
      --r-sm: 8px; --r-md: 12px; --r-lg: 18px; --r-xl: 24px; --r-pill: 980px;
      --shadow-sm: 0 1px 2px rgba(0,0,0,0.04),0 1px 1px rgba(0,0,0,0.03);
      --shadow-md: 0 4px 16px rgba(0,0,0,0.06),0 1px 2px rgba(0,0,0,0.04);
      --shadow-lg: 0 12px 40px rgba(0,0,0,0.10),0 2px 6px rgba(0,0,0,0.05);
      --font-sans: "Inter",-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif;
    }

    /* ---- Page ---- */
    html, body, [data-testid="stAppViewContainer"] {
      background: var(--bg) !important;
      font-family: var(--font-sans) !important;
      color: var(--fg2) !important;
      -webkit-font-smoothing: antialiased;
    }
    [data-testid="stHeader"], footer { display: none !important; }
    [data-testid="stMainBlockContainer"] {
      padding: 0 2rem 3rem !important;
      max-width: 1280px;
      margin: 0 auto;
    }

    /* ---- Navbar (st.navigation top) ---- */
    [data-testid="stNavigation"] {
      background: var(--bg) !important;
      border-bottom: 1px solid var(--hairline-soft) !important;
    }
    [data-testid="stNavigation"] a {
      font-family: var(--font-sans) !important;
      font-size: 13px !important;
      font-weight: 500 !important;
      color: var(--fg3) !important;
    }
    [data-testid="stNavigation"] a:hover { color: var(--fg1) !important; }
    [data-testid="stNavigation"] a[aria-current="page"] {
      color: var(--blue) !important;
      border-bottom: 2px solid var(--blue) !important;
    }

    /* ---- Titres ---- */
    h1 { font-size: 28px !important; font-weight: 500 !important;
         letter-spacing: -0.02em !important; color: var(--fg1) !important; }
    h2 { font-size: 20px !important; font-weight: 500 !important;
         letter-spacing: -0.01em !important; color: var(--fg1) !important; }
    h3 { font-size: 16px !important; font-weight: 500 !important;
         letter-spacing: -0.005em !important; color: var(--fg1) !important; }

    /* ---- Card ---- */
    .hs-card {
      background: var(--bg-elevated);
      border-radius: var(--r-lg);
      box-shadow: var(--shadow-md);
      padding: 24px 26px;
      margin-bottom: 16px;
    }
    .hs-card-hero { border-radius: var(--r-xl); }
    .hs-card-title {
      font-size: 11px; font-weight: 500; letter-spacing: .06em;
      text-transform: uppercase; color: var(--fg4); margin-bottom: 12px;
    }

    /* ---- KPI ---- */
    .hs-kpi { padding: 18px 18px 16px; }
    .hs-kpi-val {
      font-size: 34px; font-weight: 400; letter-spacing: -0.015em;
      color: var(--fg1); line-height: 1; font-variant-numeric: tabular-nums;
    }
    .hs-kpi-unit { font-size: 14px; color: var(--fg3); margin-left: 3px; }

    /* ---- Badge pill ---- */
    .hs-badge {
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 12px; font-weight: 500; padding: 3px 9px;
      border-radius: var(--r-pill); font-variant-numeric: tabular-nums;
    }
    .hs-badge-normal     { background: var(--alert-1-soft); color: var(--alert-1); }
    .hs-badge-surv       { background: var(--alert-2-soft); color: var(--alert-2); }
    .hs-badge-alerte     { background: var(--alert-3-soft); color: var(--alert-3); }
    .hs-badge-crise      { background: var(--alert-4-soft); color: var(--alert-4); }

    /* ---- Dataframe ---- */
    [data-testid="stDataFrame"] {
      border-radius: var(--r-md) !important;
      box-shadow: var(--shadow-sm) !important;
      border: 1px solid var(--hairline-soft) !important;
    }

    /* ---- Boutons ---- */
    [data-testid="stButton"] button {
      font-family: var(--font-sans) !important;
      font-size: 13px !important; font-weight: 500 !important;
      border-radius: var(--r-pill) !important;
      border: none !important;
      background: var(--bg-sunken) !important;
      color: var(--fg2) !important;
    }
    [data-testid="stButton"] button:hover {
      background: var(--bg-hover) !important;
      color: var(--fg1) !important;
    }

    /* ---- Selectbox ---- */
    [data-testid="stSelectbox"] label {
      font-size: 11px !important; font-weight: 500 !important;
      letter-spacing: .06em !important; text-transform: uppercase !important;
      color: var(--fg4) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def hs_header(subtitle="Surveillance piézométrique · Poitou-Charentes"):
    """Header Apple-style : titre à gauche, heure à droite"""
    now_str = datetime.now().strftime("Maj %H:%M")
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;padding:20px 4px 28px">
      <div>
        <div style="font-size:26px;font-weight:500;letter-spacing:-0.02em;color:var(--fg1);line-height:1">
          Hydro-Sense
        </div>
        <div style="font-size:12px;color:var(--fg3);letter-spacing:.01em;margin-top:4px">
          {subtitle}
        </div>
      </div>
      <div style="font-size:12px;color:var(--fg4);font-variant-numeric:tabular-nums">{now_str}</div>
    </div>
    """, unsafe_allow_html=True)


def hs_card(content_fn, title=None, hero=False):
    """Wrapper card avec ombre et coins arrondis"""
    radius = "var(--r-xl)" if hero else "var(--r-lg)"
    st.markdown(f'<div style="background:#fff;border-radius:{radius};box-shadow:var(--shadow-md);padding:24px 26px;margin-bottom:16px">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="hs-card-title">{title}</div>', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)


def hs_kpi(label, value, unit="", niveau=None):
    """Carte KPI avec badge niveau d'alerte optionnel"""
    BADGE = {
        "Normal":       ("hs-badge-normal",  "#5E8C73"),
        "Surveillance": ("hs-badge-surv",    "#C99A3C"),
        "Alerte":       ("hs-badge-alerte",  "#C16C3A"),
        "Crise":        ("hs-badge-crise",   "#B23B3B"),
    }
    badge_html = ""
    if niveau and niveau in BADGE:
        cls, color = BADGE[niveau]
        badge_html = f'<span class="hs-badge {cls}"><span style="width:7px;height:7px;border-radius:50%;background:{color};display:inline-block"></span>{niveau}</span>'

    st.markdown(f"""
    <div style="background:#fff;border-radius:var(--r-lg);box-shadow:var(--shadow-md);padding:18px 18px 16px;margin-bottom:12px">
      <div class="hs-card-title">{label}</div>
      <div style="display:flex;align-items:baseline;gap:4px;margin:12px 0 10px">
        <span class="hs-kpi-val">{value}</span>
        <span class="hs-kpi-unit">{unit}</span>
      </div>
      {badge_html}
    </div>
    """, unsafe_allow_html=True)


def hs_badge(niveau):
    """Badge pill standalone"""
    BADGE = {
        "Normal":       ("hs-badge-normal",  "#5E8C73"),
        "Surveillance": ("hs-badge-surv",    "#C99A3C"),
        "Alerte":       ("hs-badge-alerte",  "#C16C3A"),
        "Crise":        ("hs-badge-crise",   "#B23B3B"),
    }
    if niveau in BADGE:
        cls, color = BADGE[niveau]
        st.markdown(f"""
        <span class="hs-badge {cls}">
          <span style="width:7px;height:7px;border-radius:50%;background:{color};display:inline-block"></span>
          {niveau}
        </span>
        """, unsafe_allow_html=True)
