import os, warnings
import pandas as pd
from google.cloud import bigquery
from hydrosense.params import *

TABLE_ID   = "chroniques_piezo"

def load_piezo_bq(bss_id: str):

    table_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{TABLE_ID}"

    query = f"""
        SELECT date_mesure, niveau_nappe_eau
        FROM `{table_ref}`
        WHERE bss_id = '{bss_id}'
        ORDER BY date_mesure
    """

    client = bigquery.Client(project=GCP_PROJECT_ID)
    # Catch and ignore the specific BigQuery Storage API fallback warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="google.cloud.bigquery")
        df = client.query(query).to_dataframe()

    if df.empty:
        raise ValueError(f"Aucune donnée trouvée pour {bss_id}")

    df["date_mesure"] = pd.to_datetime(df["date_mesure"])
    df.sort_values(by="date_mesure", inplace=True)
    df.reset_index(drop=True, inplace=True)


    print(f"{bss_id} : {len(df)} lignes chargées")

    return df

def info_piezo(bss_id: str, raw = False):
    """
    Loads the information concerning the bss_id
    either from the cat_piezo_raw or the cat_piezo_interm table.

    """
    CAT_ID = 'cat_piezo_raw' if raw else 'cat_piezo_interm'

    table_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{CAT_ID}"

    query = f"""
        SELECT *
        FROM `{table_ref}`
        WHERE bss_id = '{bss_id}'
        """

    client = bigquery.Client(project=GCP_PROJECT_ID)
    info = client.query(query)

    return info.to_dataframe()


def save_dataframe_to_bq(df, bss_id, write_mode="WRITE_TRUNCATE"):
    """
    Enregistre dans une table BigQuery - Donneé FUSIONNEE piezo + meteo
    Format de table : pem_{bss_id}

    example :
    save_dataframe_to_bq(df_final, "BSS001QHYH")
    """
    client = bigquery.Client(project=GCP_PROJECT_ID)
    table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.pem_{bss_id}"

    # Configuration du job de chargement
    job_config = bigquery.LoadJobConfig(
                                        write_disposition=write_mode,
                                        )

    try:
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Attend la fin du job
        print(f"✅ Données chargées avec succès dans {table_id}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement dans BigQuery: {e}")
        raise
