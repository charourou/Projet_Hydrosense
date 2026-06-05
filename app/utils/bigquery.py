import os
import streamlit as st
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET = os.getenv("BQ_DATASET_ID")

client = bigquery.Client(project=GCP_PROJECT_ID)


@st.cache_data
# def load_piezo():
#     query = f"""
#         SELECT *
#         FROM `{GCP_PROJECT_ID}.{DATASET}.piezo_bourdet_test`
#     """
#     return client.query(query).to_dataframe()


@st.cache_data
def load_catalog():
    client = bigquery.Client(project=GCP_PROJECT_ID)  # client dans la fonction
    query = f"""
        SELECT *
        FROM `{GCP_PROJECT_ID}.{DATASET}.cat_piezo_raw`
    """
    return client.query(query).to_dataframe()

@st.cache_data
def load_catalog_map():
    client = bigquery.Client(project=GCP_PROJECT_ID)
    query = f"""
        SELECT bss_id, nom_commune, nom_departement, x, y
        FROM `{GCP_PROJECT_ID}.{DATASET}.cat_piezo_raw`
        WHERE x IS NOT NULL AND y IS NOT NULL
    """
    return client.query(query).to_dataframe()

from google.cloud import bigquery

@st.cache_data
def load_single_piezo_map(bss_id: str):
    """
    Charge les coordonnées et infos d'un seul piézomètre spécifique via son bss_id.
    """
    client = bigquery.Client(project=GCP_PROJECT_ID)
    
    # Requête SQL paramétrée avec @bss_id
    query = f"""
        SELECT bss_id, nom_commune, nom_departement, x, y
        FROM `{GCP_PROJECT_ID}.{DATASET}.cat_piezo_raw`
        WHERE bss_id = @bss_id 
          AND x IS NOT NULL 
          AND y IS NOT NULL
        LIMIT 1
    """
    
    # Configuration sécurisée du paramètre pour BigQuery
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("bss_id", "STRING", bss_id)
        ]
    )
    
    df = client.query(query, job_config=job_config).to_dataframe()
    return df
