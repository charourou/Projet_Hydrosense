import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

# Connexion
load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET = os.getenv("BQ_DATASET_ID")
client = bigquery.Client(project=GCP_PROJECT_ID)

# Chargement des données
def load_piezo():
    query = f"""
        SELECT *
        FROM `{GCP_PROJECT_ID}.{DATASET}.piezo_test`
    """
    return client.query(query).to_dataframe()

def load_catalog():
    query = f"""
        SELECT *
        FROM `{GCP_PROJECT_ID}.{DATASET}.cat_piezo_raw`
    """
    return client.query(query).to_dataframe()

# Interface
st.title("💧 Hydro-Sense")

st.subheader("📊 Données piézométriques")
df_piezo = load_piezo()
st.dataframe(df_piezo)

st.subheader("🗺️ Catalogue des piézomètres")
df_catalog = load_catalog()
st.dataframe(df_catalog)
