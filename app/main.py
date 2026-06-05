import streamlit as st



# from utils.style import apply_hydrosense_theme, hs_header

# apply_hydrosense_theme()
# hs_header()

st.set_page_config(
    page_title="Hydro-Sense",
    page_icon="💧",
    layout="wide",
)

pages = {
    "": [
        st.Page("pages/accueil.py",    title="🏠 Accueil"),
        st.Page("pages/piezo.py",      title="📈 Données piézométrique"),
        st.Page("pages/piezo-plot.py", title="📈 Courbe piézométrique"),
        st.Page("pages/catalogue.py",  title="🗂️ Catalogue"),
        st.Page("pages/carte.py",      title="🗺️ Carte des piézos"),
        st.Page("pages/dashboard-maquette.py",  title="📊 Dashboard-maquette"),
    ]
}

pg = st.navigation(pages, position="top")
pg.run()
