import os
import streamlit as st
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET = os.getenv("BQ_DATASET_ID")

client = bigquery.Client(project=GCP_PROJECT_ID)


@st.cache_data
def load_catalog():
    client = bigquery.Client(project=GCP_PROJECT_ID)
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


@st.cache_data
def load_single_piezo_map(bss_id: str):
    """
    Charge les coordonnées et infos d'un seul piézomètre via son bss_id.
    """
    client = bigquery.Client(project=GCP_PROJECT_ID)
    query = f"""
        SELECT bss_id, nom_commune, nom_departement, x, y
        FROM `{GCP_PROJECT_ID}.{DATASET}.cat_piezo_raw`
        WHERE bss_id = @bss_id
          AND x IS NOT NULL
          AND y IS NOT NULL
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("bss_id", "STRING", bss_id)
        ]
    )
    return client.query(query, job_config=job_config).to_dataframe()


@st.cache_data
def load_catalog_interm():
    """
    Charge la liste des piézomètres depuis cat_piezo_interm
    pour alimenter le sélecteur de l'interface.
    Retourne bss_id, nom_commune, code_departement.
    """
    client = bigquery.Client(project=GCP_PROJECT_ID)
    query = f"""
        SELECT bss_id, nom_commune, code_departement
        FROM `{GCP_PROJECT_ID}.{DATASET}.cat_piezo_interm`
        WHERE bss_id IS NOT NULL
        ORDER BY code_departement, nom_commune
    """
    return client.query(query).to_dataframe()


@st.cache_data
def load_seuils_interm(bss_id: str) -> dict | None:
    """
    Charge les seuils percentiles depuis cat_piezo_interm
    en attendant les seuils réglementaires définitifs.

    Correspondance provisoire :
        Normal       → p85
        Surveillance → p50
        Alerte       → p20
        Crise        → p10

    Retourne None si le piézomètre est introuvable ou si
    les valeurs percentiles sont toutes nulles.
    """
    client = bigquery.Client(project=GCP_PROJECT_ID)
    query = f"""
        SELECT p95_global, p85_global, p20_global, p10_global, p5_global
        FROM `{GCP_PROJECT_ID}.{DATASET}.cat_piezo_interm`
        WHERE bss_id = @bss_id
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("bss_id", "STRING", bss_id)
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()

    if df.empty:
        return None

    row = df.iloc[0]

    # Si l'une des valeurs est nulle, on ne peut pas construire les seuils
    if row[["p95_global", "p85_global", "p20_global", "p10_global", "p5_global"]].isnull().any():
        return None

    return {
        "p95": float(row["p95_global"]),
        "p85": float(row["p85_global"]),
        "p20": float(row["p20_global"]),
        "p10": float(row["p10_global"]),
        "p5":  float(row["p5_global"]),
    }


@st.cache_data
def load_seuils(bss_id: str) -> dict | None:
    """
    Charge les 4 seuils réglementaires pour un piézomètre donné
    depuis la table chroniques_piezo.

    Colonnes attendues dans la table :
        seuil_vigilance, seuil_alerte, seuil_alerte_renforcee, seuil_crise

    Retourne un dict :
        {
            "vigilance":        float,
            "alerte":           float,
            "alerte_renforcee": float,
            "crise":            float,
        }
    ou None si le bss_id n'est pas trouvé.
    """
    client = bigquery.Client(project=GCP_PROJECT_ID)
    query = f"""
        SELECT
            seuil_vigilance,
            seuil_alerte,
            seuil_alerte_renforcee,
            seuil_crise
        FROM `{GCP_PROJECT_ID}.{DATASET}.chroniques_piezo`
        WHERE bss_id = @bss_id
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("bss_id", "STRING", bss_id)
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()

    if df.empty:
        return None

    row = df.iloc[0]
    return {
        "vigilance":        float(row["seuil_vigilance"]),
        "alerte":           float(row["seuil_alerte"]),
        "alerte_renforcee": float(row["seuil_alerte_renforcee"]),
        "crise":            float(row["seuil_crise"]),
    }