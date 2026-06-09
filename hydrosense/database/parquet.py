import os, glob, re
from pathlib import Path
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import Conflict
from hydrosense.params import *


# --- CONFIGURATION ---
LOCAL_DATA_DIR = "/home/charourou/projects/Projet_Hydrosense/raw_data/"
TABLE_ID = "chroniques_piezo"
TEMP_PARQUET_FILE = "all_chroniques.parquet"

# Vérification de l'inclusion dans la liste des stations autorisées
STATIONS_AUTORISEES = []

client = bigquery.Client(project=PROJECT_ID)
table_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{TABLE_ID}"

schema = [
        bigquery.SchemaField("bss_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date_mesure", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("niveau_nappe_eau", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("profondeur_nappe", "FLOAT", mode="REQUIRED"),
        ]


# PIPELINE 1 : DONNÉES BRUTES (CSV -> chroniques_piezo)
def create_optimized_table():
    """Crée la table BigQuery cible avec Partitionnement et Clustering"""

    table = bigquery.Table(table_ref, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.YEAR,
        field="date_mesure"
    )

    # Clustering par code de forage
    table.clustering_fields = ["bss_id"]

    try:
        client.create_table(table)
        print(f"Table {table_ref} créée avec succès (Partitionnée + Clustérisée).")
    except Conflict:
        print(f"La table {table_ref} existe déjà. Les données y seront ajoutées.")

def pipeline_local_to_parquet():
    """Combine les CSV en un seul fichier parquet compressé"""

    csv_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.csv"))
    print(f"Trouvé {len(csv_files)} fichiers CSV à traiter.")

    all_chunks = []

    for i, file_path in enumerate(csv_files, 1):
        nom_fichier = os.path.basename(file_path)

        # Extraction du code BSS 'piezo_' via une Regex
        # gerer si le nom du fichier n'est pas un piezo_****.csv
        match = re.search(r"piezo_(.*?)\.csv", nom_fichier)
        if match:
            bss_id = match.group(1)
        else:
            continue

        # Vérification de l'inclusion dans la liste des stations autorisées
        if STATIONS_AUTORISEES and (bss_id not in STATIONS_AUTORISEES):
            continue

        try:
            # Lecture du CSV individuel
            df_filtered = pd.read_csv(file_path, sep = ';')

            df_filtered['bss_id'] = bss_id
            df_filtered['date_mesure'] = pd.to_datetime(df_filtered['date_mesure']).dt.date
            df_filtered['niveau_nappe_eau'] = pd.to_numeric(df_filtered['niveau_nappe_eau'], errors='coerce')
            df_filtered['profondeur_nappe'] = pd.to_numeric(df_filtered['profondeur_nappe'], errors='coerce')

            all_chunks.append(df_filtered)

            if i % 200 == 0:
                print(f"Progression : {i}/{len(csv_files)} fichiers indexés...")

        except Exception as e:
            print(f"Erreur sur le fichier {file_path} : {e}")
            continue

    # Fusion magique en mémoire et écriture en Parquet (très léger car compressé)
    print("Compilation globale en cours...")
    grand_df = pd.concat(all_chunks, ignore_index=True)
    grand_df.to_parquet(TEMP_PARQUET_FILE, index=False)
    print(f"Fichier compressé créé : {TEMP_PARQUET_FILE} ({len(grand_df)} lignes)")

def upload_to_bigquery():
    print("Début du transfert vers Google Cloud BigQuery...")

    job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            # Remplacer WRITE_APPEND par WRITE_TRUNCATE
            schema=schema,
        )

    with open(TEMP_PARQUET_FILE, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result()  # Attend la fin du transfert
    print(f"Succès ! Les données ont été chargées dans {table_ref}")

    # Nettoyage du fichier temporaire
    if os.path.exists(TEMP_PARQUET_FILE):
        os.remove(TEMP_PARQUET_FILE)


# PIPELINE 2 : DONNÉES MACHINE LEARNING (Parquet -> chroniques_plean)
def upload_chroniques_plean():
    """
    Fusionne les fichiers Parquet locaux préparés
    et les upload directement vers la table 'chroniques_plean' sur BigQuery.
    """
    target_table_id = "chroniques_plean"
    target_table_ref = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{target_table_id}"

    dossier_parquets = os.path.join(LOCAL_DATA_DIR, "processed_pem", "*.parquet")
    cache_global = os.path.join(LOCAL_DATA_DIR, "all_chroniques_plean.parquet")

    print("\n📥 1. Lecture et fusion des fichiers Parquet locaux (Features ML)...")
    df_list = []

    for file_path in glob.glob(dossier_parquets):
        # Extraction du code BSS
        nom_fichier = Path(file_path).stem
        bss_id = nom_fichier.split('_')[-1]

        df = pd.read_parquet(file_path)
        df['bss_id'] = bss_id  # Ajout de la colonne requise
        df_list.append(df)

    if not df_list:
        print("❌ Aucun fichier Parquet trouvé. Avez-vous lancé le pipeline de préparation ?")
        return

    # Fusion globale
    df_global = pd.concat(df_list, ignore_index=True)

    # Réorganisation des colonnes et nettoyage des noms
    df_global.rename(columns={'PC1_hydro': 'PC1', 'PC2_hydro': 'PC2', 'PC3_hydro': 'PC3'}, inplace=True, errors='ignore')
    cols = ['bss_id', 'date_mesure', 'niveau_nappe_eau', 'RR_synth', 'TM_synth', 'FFM_synth', 'PU_synth', 'PC1', 'PC2', 'PC3']
    df_global = df_global[[c for c in cols if c in df_global.columns]]

    print(f"💾 2. Création du cache local : {cache_global}")
    df_global.to_parquet(cache_global, index=False, compression='snappy')

    print(f"🚀 3. Upload de {len(df_global)} lignes vers BigQuery ({target_table_id})...")

    # WRITE_TRUNCATE permet d'écraser l'ancienne table à chaque fois
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

    # BigQuery devine le schéma tout seul grâce à l'objet Pandas
    job = client.load_table_from_dataframe(df_global, target_table_ref, job_config=job_config)
    job.result()

    print(f"✅ Upload terminé avec succès ! La table {target_table_id} est prête pour XGBoost.")

# =====================================================================
# EXECUTION
if __name__ == "__main__":
    # -- Pour les données brutes :
    # create_optimized_table()
    # pipeline_local_to_parquet()
    # upload_to_bigquery()

    # -- Pour les données Machine Learning préparées :
    # upload_chroniques_plean()
    pass
