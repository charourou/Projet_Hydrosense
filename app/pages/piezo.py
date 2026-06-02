import streamlit as st
from utils.bigquery import load_piezo

st.title("📊 Données piézométriques")

df = load_piezo()
st.dataframe(df)
