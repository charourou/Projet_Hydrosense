import os

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "hydro-sense-498112")
DATASET_ID = os.environ.get("BQ_DATASET_ID",  "piezometry")
TABLE_ID   = "chroniques_piezo"

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT BIGQUERY
# ─────────────────────────────────────────────────────────────────────────────

def load_piezo_bq(bss_id: str):

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    
    query = f"""
        SELECT date_mesure, niveau_nappe_eau
        FROM `{table_ref}`
        WHERE bss_id = '{bss_id}'
        ORDER BY date_mesure
    """
    
    client = bigquery.Client(project=PROJECT_ID)
    df = client.query(query).to_dataframe()

    if df.empty:
        raise ValueError(f"Aucune donnée trouvée pour {bss_id}")

    df["date_mesure"] = pd.to_datetime(df["date_mesure"])
    df = df.set_index("date_mesure").sort_index()

    print(f"{bss_id} : {len(df)} lignes chargées")

    return df