import streamlit as st

pages = {
    "": [
        st.Page("pages/accueil.py", title="🏠 Accueil"),
        st.Page("pages/piezo.py", title="📊 Données piézométriques"),
        st.Page("pages/catalogue.py", title="🗺️ Catalogue"),
        st.Page("pages/carte.py", title="🗺️ Carte des piézos"),
    ]
}

pg = st.navigation(pages, position="top")
pg.run()
