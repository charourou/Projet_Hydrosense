import os, re, requests
import pandas as pd
import warnings

# Désactivation des alertes de types mixtes fréquentes sur les CSV Météo-France
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

class CatalogueMeteo:
    """Gestionnaire d'extraction des données Météo-France via data.gouv.fr."""

    DATASET_ID = "6569b51ae64326786e4e8e1a"
    API_URL = f"https://www.data.gouv.fr/api/1/datasets/{DATASET_ID}/"

    def __init__(self, dossier_cache="raw_data/meteo"):
        self.dossier_cache = dossier_cache
        os.makedirs(self.dossier_cache, exist_ok=True)

    def _obtenir_urls_dept(self, code_dept: str) -> list:
        """Récupère dynamiquement les URLs des fichiers d'un département."""
        reponse = requests.get(self.API_URL)
        reponse.raise_for_status()

        prefixe = f"QUOT_departement_{code_dept}_"
        return [{'titre': res.get('title'), 'url': res.get('url')}
                for res in reponse.json().get('resources', [])
                if res.get('title', '').startswith(prefixe) and "RR-T-Vent" in res.get('title', '')
                ]

    def _trier_urls_recentes_en_premier(self, urls: list) -> list:
        """Trie les fichiers pour mettre la période la plus récente en premier."""
        def extraire_annee_max(url_info):
            # Cherche toutes les années (4 chiffres) dans le titre du fichier
            annees = re.findall(r'\d{4}', url_info['titre'])
            return max([int(a) for a in annees]) if annees else 0

        return sorted(urls, key=extraire_annee_max, reverse=True)

    def _telecharger_fichier(self, url: str, chemin_local: str):
        if not chemin_local.endswith('.csv.gz'):
            chemin_local += '.csv.gz'

        if not os.path.exists(chemin_local):
            print(f"Téléchargement : {os.path.basename(chemin_local)}...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(chemin_local, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        return chemin_local

    def _nettoyer_donnees(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formatage des dates et typage des variables climatiques."""

        # Extraction de l'objet date pur
        dates_datetime = pd.to_datetime(df['AAAAMMJJ'], format='%Y%m%d', errors='coerce')
        df['date_RR'] = dates_datetime.dt.date

        # LAT/LON :
        for col in ['LAT', 'LON']:
            if col in df.columns:
                # remplace les éventuelles virgules françaises par des points
                df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Variables climatiques cibles
        cibles = ['RR', 'TM', 'FFM']
        for col in cibles:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sélection stricte et tri chronologique
        colonnes_utiles = ['NUM_POSTE','date_RR', 'LAT', 'LON' ] + cibles
        colonnes_existantes = [c for c in colonnes_utiles if c in df.columns]
        return df[colonnes_existantes].sort_values(by='date_RR').reset_index(drop=True)

    def extraire_departement(self, code_dept: str) -> dict:
        """
        Télécharge les données d'un département, identifie les stations actives
        et ne récupère l'historique que pour ces stations.
        Retourne un dictionnaire : { 'code_station': DataFrame_nettoyé }
        """
        urls = self._obtenir_urls_dept(code_dept)
        urls_triees = self._trier_urls_recentes_en_premier(urls)

        if not urls_triees:
            print(f"Aucune URL trouvée pour le département {code_dept}")
            return {}

        donnees_par_station = {}

        #  Le fichier le plus récent définit les stations cibles ---
        fichier_recent = urls_triees[0]
        chemin_recent = os.path.join(self.dossier_cache, fichier_recent['titre']) + '.csv.gz'
        self._telecharger_fichier(fichier_recent['url'], chemin_recent)

        print(f"🔍 Analyse du fichier de référence : {fichier_recent['titre']}")
        df_recent = pd.read_csv(chemin_recent, sep=';', compression='gzip', dtype=str)

        # Extraction des identifiants uniques DANS UN SET
        stations_actives = set(df_recent['NUM_POSTE'].unique())
        print(f"🎯 {len(stations_actives)} stations actives identifiées pour le {code_dept}.")

        # Initialisation du dictionnaire avec les données récentes
        for st in stations_actives:
            donnees_par_station[st] = [df_recent[df_recent['NUM_POSTE'] == st]]

        # --- Traitement de l'historique ---
        for res in urls_triees[1:]:
            if 'avant' in res['titre']:
                continue

            chemin_local = self._telecharger_fichier(res['url'], os.path.join(self.dossier_cache, res['titre']))
            print(f"📖 Récupération de l'historique : {os.path.basename(chemin_local)}")

            # Lire le fichier avec le bon chemin. On ne garde que les lignes dont le NUM_POSTE est dans notre set
            df_historique = pd.read_csv(chemin_local, sep=';', compression='gzip', dtype=str)
            df_historique_filtre = df_historique[df_historique['NUM_POSTE'].isin(stations_actives)]

            # Dispatching des données historiques par station
            for st in df_historique_filtre['NUM_POSTE'].unique():
                donnees_par_station[st].append(df_historique_filtre[df_historique_filtre['NUM_POSTE'] == st])

        # --- Assemblage et nettoyage ---
        print(f"⚙️ Nettoyage et assemblage final pour {len(stations_actives)} stations...")
        resultat_final = {}

        for st, liste_dfs in donnees_par_station.items():
            df_complet = pd.concat(liste_dfs, ignore_index=True)
            resultat_final[st] = self._nettoyer_donnees(df_complet)


        print("✅ Extraction départementale terminée !")
        return resultat_final



if __name__ == '__main__':
    meteo = CatalogueMeteo(dossier_cache="raw_data/meteo")
    db_test = meteo.extraire_departement("64")



    for df in db_test.values():
        print(df['date_RR'].diff().unique())
    # key = list(df_test.keys())[1]
    # print(key)
    # print(df_test[key] )
