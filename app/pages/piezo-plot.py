import streamlit as st
import pandas as pd
import altair as alt  # On importe altair pour configurer l'axe finement
# from utils.bigquery import load_piezo

st.title("📈 Courbe piézométrique (Sur 1 an)")

# 1. Chargement des données

from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo

df_raw = load_piezo_bq("BSS001QHYH")
df = clean_piezo(df_raw)

# 2. Nettoyage des types
df['date_mesure'] = pd.to_datetime(df['date_mesure'])
df['niveau_nappe_eau'] = pd.to_numeric(df['niveau_nappe_eau'], errors='coerce')

# 3. Filtre sur 1 an
un_an_en_arriere = pd.Timestamp.now() - pd.Timedelta(days=365)
df_filtre = df[df['date_mesure'] >= un_an_en_arriere]

# 4. Création du graphique Altair avec l'échelle Y ajustée (zero=False)
chart = alt.Chart(df_filtre).mark_line(color='#1f77b4').encode(
    x=alt.X('date_mesure:T', title='Date de mesure'),
    y=alt.Y('niveau_nappe_eau:Q', title='Niveau de la nappe d\'eau', scale=alt.Scale(zero=False))
).properties(
    height=400
)

# 5. Affichage dans Streamlit
st.altair_chart(chart, use_container_width=True)