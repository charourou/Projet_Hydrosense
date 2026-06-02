import os, glob
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import Conflict

# --- CONFIGURATION ---
LOCAL_DATA_DIR = "../raw_data/"
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
DATASET_ID = os.environ.get("BQ_DATASET_ID")
TABLE_ID = "chroniques_piezo"
TEMP_PARQUET_FILE = "all_chroniques.parquet"

client = bigquery.Client(project=PROJECT_ID)
table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

def create_optimized_table():
    """Crée la table BigQuery cible avec Partitionnement et Clustering"""
    schema = [
        bigquery.SchemaField("code_bss", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date_mesure", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("niveau_nappe_eau", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("profondeur_nappe", "FLOAT", mode="REQUIRED"),
    ]

    table = bigquery.Table(table_ref, schema=schema)

    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.YEAR,
        field="date_mesure"
    )

    # Clustering par code de forage
    table.clustering_fields = ["code_bss"]

    try:
        client.create_table(table)
        print(f"Table {table_ref} créée avec succès (Partitionnée + Clustérisée).")
    except Conflict:
        print(f"La table {table_ref} existe déjà. Les données y seront ajoutées.")

def pipeline_local_to_parquet():
    """Combine les 2750 CSV en un seul fichier Parquet compressé"""
    csv_files = glob.glob(os.path.join(LOCAL_DATA_DIR, "*.csv"))
    print(f"Trouvé {len(csv_files)} fichiers CSV à traiter.")

    all_chunks = []

    for i, file_path in enumerate(csv_files, 1):
        try:
            # Lecture du CSV individuel
            # On adapte les noms selon ce que l'API Hub'Eau crache (souvent 'code_bss', 'date_mesure', 'profondeur_nappe')
            df = pd.read_csv(file_path)

            # Uniquement la sélection stricte pour notre modèle en étoile
            # /!\ Ajustez les chaînes de caractères si vos colonnes CSV s'appellent autrement (ex: 'date')
            df_filtered = df[['code_bss', 'date_mesure', 'niveau_nappe']].copy()

            # Nettoyage rapide des types
            df_filtered['date_mesure'] = pd.to_datetime(df_filtered['date_mesure']).dt.date
            df_filtered['niveau_nappe'] = pd.to_numeric(df_filtered['niveau_nappe'], errors='coerce')
            df_filtered = df_filtered.dropna(subset=['niveau_nappe'])

            all_chunks.append(df_filtered)

            if i % 500 == 0:
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
    """Envoie le fichier Parquet vers BigQuery"""
    print("Début du transfert vers Google Cloud BigQuery...")

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND, # Ajoute si déjà existant
    )

    with open(TEMP_PARQUET_FILE, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result()  # Attend la fin du transfert
    print(f"Succès ! Les données ont été chargées dans {table_ref}")

    # Nettoyage du fichier temporaire
    if os.path.exists(TEMP_PARQUET_FILE):
        os.remove(TEMP_PARQUET_FILE)

if __name__ == "__main__":
    create_optimized_table()
    # pipeline_local_to_parquet()
    # upload_to_bigquery()
