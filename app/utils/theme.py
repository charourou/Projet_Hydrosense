"""
utils/theme.py
──────────────

"""

# ── Palette des seuils de gestion ────────────────────────────────────────────
# Chaque niveau : (bg_pastel, border/fg, label)
# Ordre : du plus critique au plus favorable
SEUIL_COLORS: dict[str, dict] = {
    "Crise":           {"bg": "#f8cbad", "fg": "#d9534f"},
    "Alerte":          {"bg": "#fce4d6", "fg": "#f0ad4e"},
    "Vigilance":       {"bg": "#fff2cc", "fg": "#e8c100"},
    "Normal":          {"bg": "#e2f0d9", "fg": "#2ca02c"},
    "Modérément haut": {"bg": "#d4edda", "fg": "#3a7d44"},
    "Très haut":       {"bg": "#c8e6fa", "fg": "#1a6fa8"},
}

# Ordre d'affichage (du plus favorable → moins favorable, pour la légende)
SEUIL_ORDER = ["Très haut", "Modérément haut", "Normal", "Vigilance", "Alerte", "Crise"]

# ── Tokens de design (card, typographie, couleur primaire) ───────────────────
DESIGN_TOKENS = {
    # Fond principal
    "bg_page":        "#F5F5F7",
    # Fond des cards
    "bg_card":        "#ffffff",
    # Bordure subtile des cards
    "border_card":    "#f1f5f9",
    # Shadow cards
    "shadow_card":    "0 4px 16px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
    # Radius
    "radius_card":    "12px",
    "radius_pill":    "999px",
    # Couleur ligne historique / prédiction
    "color_line":     "#2484e5",
    "color_line_pred":     "#95d2f8",
    # Couleur texte section titles
    "color_section":  "#475569",
    # Fond selectbox pill
    "bg_selectbox":   "#efeff2",
}

# ── Injection CSS globale ─────────────────────────────────────────────────────
def apply_global_css() -> None:
    """
    Injecte le CSS global de l'app via st.markdown().
    À appeler UNE SEULE FOIS, dans main.py après st.set_page_config().

    Utilise les variables de DESIGN_TOKENS pour éviter les couleurs hardcodées.
    """
    import streamlit as st

    t = DESIGN_TOKENS

    st.markdown(f"""
        <style>
            /* ── Card graphique Altair ── */
            [data-testid="stMainBlockContainer"] {{
                padding-top: 3rem;
            }}    

            /* ── Métriques KPI ── */
            [data-testid="stMetric"] {{
                background-color: {t['bg_card']} !important;
                padding: 15px !important;
                border-radius: {t['radius_card']} !important;
                border: 1px solid {t['border_card']} !important;
                box-shadow: {t['shadow_card']} !important;
            }}

            /* ── Titres de section ── */
            .section-title {{
                font-size: 1.05rem;
                font-weight: 700;
                color: {t['color_section']};
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 14px;
                margin-top: 5px;
            }}

            /* ── Card graphique Altair ── */
            [data-testid="stVegaLiteChart"] {{
                background-color: {t['bg_card']} !important;
                padding: 15px !important;
                border-radius: {t['radius_card']} !important;
                border: 1px solid {t['border_card']} !important;
                box-shadow: {t['shadow_card']} !important;
            }}
            /* Altair injecte backgroundColor du config.toml en style inline sur le SVG —
               on force transparent pour que le fond blanc du container s'applique */
            [data-testid="stVegaLiteChart"] svg,
            [data-testid="stVegaLiteChart"] .chart-wrapper {{
                background-color: transparent !important;
            }}

            /* ── Card carte DeckGL ── */
            [data-testid="stDeckGlJsonChart"] {{
                border-radius: {t['radius_card']} !important;
                border: 1px solid {t['border_card']} !important;
                box-shadow: {t['shadow_card']} !important;
                overflow: hidden !important;
            }}

            /* ── Sélecteur piézomètre — style pill ── */
            [data-testid="stSelectbox"] > div > div {{
                background-color: {t['bg_selectbox']} !important;
                border-radius: {t['radius_pill']} !important;
                border: none !important;
                box-shadow: none !important;
            }}

            /* ── Card seuils (classe utilitaire réutilisable) ── */
            .hs-card {{
                background-color: {t['bg_card']};
                padding: 15px;
                border-radius: {t['radius_card']};
                border: 1px solid {t['border_card']};
                box-shadow: {t['shadow_card']};
            }}
        </style>
    """, unsafe_allow_html=True)
