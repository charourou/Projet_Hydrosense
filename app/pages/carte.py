import folium
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from utils.api_client import load_catalog_map

st.title("🗺️ Carte des piézomètres")

df = load_catalog_map()

m = folium.Map(location=[46.5, 2.5], zoom_start=6, tiles="CartoDB Positron", prefer_canvas=True)

m.get_root().html.add_child(folium.Element("""
<style>
.leaflet-control-container { display: none !important; }
</style>
"""))

cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    if pd.notna(row["x"]) and pd.notna(row["y"]):
        folium.CircleMarker(
            location=[row["y"], row["x"]],
            radius=7,
            color="#2484e5",
            fill=True,
            fill_color="#2484e5",
            fill_opacity=0.55,
            weight=1.5,
            tooltip=f"{row['bss_id']} — {row['nom_commune']}",
            popup=f"{row['bss_id']} — {row['nom_commune']} ({row['nom_departement']})",
        ).add_to(cluster)

st_folium(m, width=None, height=700, returned_objects=[])
