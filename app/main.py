"""
app/main.py
───────────
Point d'entrée de l'application Hydro-Sense.

"""

import streamlit as st

from utils.theme import apply_global_css

st.set_page_config(
    page_title="Hydro-Sense",
    page_icon="💧",
    layout="wide",
)

# Injection CSS unique pour toute l'app
apply_global_css()

pages = {
    "": [
        st.Page("pages/accueil.py",           title="🏠 Accueil"),
        st.Page("pages/piezo-dashboard.py",   title="📊 Dashboard piézomètre"),
        # st.Page("pages/piezo.py",             title="📈 Données piézométriques"),
        st.Page("pages/catalogue.py",         title="🗂️ Catalogue"),
        st.Page("pages/carte.py",             title="🗺️ Carte des piézos"),
        st.Page("pages/dashboard-maquette.py", title="🖌️ Dashboard-maquette"),
    ]
}

pg = st.navigation(pages, position="top")
pg.run()
