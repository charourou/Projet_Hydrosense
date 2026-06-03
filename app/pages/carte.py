import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium
from utils.bigquery import load_catalog_map

st.title("🗺️ Carte des piézomètres")

df = load_catalog_map()

m = folium.Map(location=[46.5, 2.5], zoom_start=6)

for _, row in df.iterrows():
    if pd.notna(row["x"]) and pd.notna(row["y"]):
        folium.CircleMarker(
            location=[row["y"], row["x"]],
            radius=5,
            popup=f"{row['bss_id']} — {row['nom_commune']} ({row['nom_departement']})",
            color="#1f77b4",
            fill=True,
        ).add_to(m)

st_folium(m, width=900, height=600, returned_objects=[])
