import streamlit as st
from utils.api_client import load_catalog

st.title("🗺️ Catalogue des piézomètres")

df = load_catalog()
st.dataframe(df)
