import pandas as pd
import requests
from io import StringIO
import os

class CatalogueMeteo:
    """Gestion du catalogue des stations Météo-France (Open Data)."""

    def __init__(self, source_csv):
        """Initialise l'extracteur avec une URL data.gouv ou le chemin d'un fichier local."""
        self.source = source_csv

    def recuperer_donnees_brutes(self) -> pd.DataFrame:
        """Charge le CSV depuis une URL ou un fichier local avec le bon encodage."""
        print(f"Chargement des données Météo depuis : {self.source}")

        if self.source.startswith("http"):
            response = requests.get(self.source)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text), sep=';')
        else:
            if not os.path.exists(self.source):
                raise FileNotFoundError(f"Le fichier {self.source} est introuvable.")
            df = pd.read_csv(self.source, sep=';')
        return df

    def preparer_catalogue_bigquery(self) -> pd.DataFrame:
        """
        Nettoie le DataFrame et crée la colonne geometry (WKT) pour BigQuery.
        """
        df = self.recuperer_donnees_brutes()

        # Normalisation des colonnes
        df.columns = [c.upper().strip() for c in df.columns]

        # Détection adaptative des colonnes clés
        col_id = 'NUM_POSTE' if 'NUM_POSTE' in df.columns else 'POSTE'
        col_nom = 'NOM_USUEL' if 'NOM_USUEL' in df.columns else 'NOM_STATION'
        col_lat = 'LAT' if 'LAT' in df.columns else 'LATITUDE'
        col_lon = 'LON' if 'LON' in df.columns else 'LONGITUDE'

        # 2. Filtrage des lignes sans coordonnées
        df_clean = df.dropna(subset=[col_lat, col_lon]).copy()

        # Conversion numérique (gestion des virgules françaises vs points décimaux)
        df_clean[col_lat] = pd.to_numeric(df_clean[col_lat].astype(str).str.replace(',', '.'), errors='coerce')
        df_clean[col_lon] = pd.to_numeric(df_clean[col_lon].astype(str).str.replace(',', '.'), errors='coerce')
        df_clean = df_clean.dropna(subset=[col_lat, col_lon])

        # 3. Création de la géométrie au format WKT : POINT(Longitude Latitude)
        # CRUCIAL : BigQuery (comme PostGIS) demande TOUJOURS la Longitude en premier !
        df_clean['geometry'] = df_clean.apply(
            lambda row: f"POINT({row[col_lon]} {row[col_lat]})", axis=1
        )

        # 4. Formatage strict du schéma cible
        catalogue_final = pd.DataFrame({
            'num_station': df_clean[col_id].astype(str), # En texte car certains codes commencent par 0
            'nom_station': df_clean[col_nom],
            'geometry': df_clean['geometry']
        })

        print(f"✅ Catalogue Météo prêt : {len(catalogue_final)} stations formatées.")
        return catalogue_final

if __name__ == '__main__':
    meteo_file  = "/home/charourou/projects/Projet_Hydrosense/raw_data/Q_07_previous-1950-2024_RR-T-Vent.csv"

    meteo_agent = CatalogueMeteo(meteo_file)
    # 2. Nettoyage et formatage
    df_catalogue_meteo = meteo_agent.preparer_catalogue_bigquery()

    # 3. Aperçu avant l'envoi sur BigQuery
    print(df_catalogue_meteo.head())



