import pandas as pd
import requests

class Entite:
    def __init__(self):
        """
        Catalogue pour gérer les métadonnées des stations piézométriques.
        """
        self.url_stations = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations"
        self.donnees_brutes = [] # Stockera le JSON brut (liste de dictionnaires)
        self.catalogue_df = pd.DataFrame() # Stockera le DataFrame structuré


    def rechercher_stations(self, code_dep=None, code_region=None, code_bdlisa=None, taille_max=5000) -> list:
        """
        Retourne la liste les stations et stocke le résultat brut JSON en mémoire.
        """
        parametres = {  "format": "json",
                        "size": taille_max
                        }

        # Ajout des filtres
        if code_dep:
            parametres["code_departement"] = str(code_dep)
        if code_region:
            parametres["code_region"] = str(code_region)
        if code_bdlisa:
            parametres["code_bdlisa"] = str(code_bdlisa)
        print(f"Recherche des stations avec les paramètres : {parametres}...")

        try:
            reponse = requests.get(self.url_stations, params=parametres)
            # print(f"URL appelée : {reponse.url}")
            reponse.raise_for_status()
            donnees = reponse.json()

            # Extraction des bss_id dans une liste.
            liste_stations = []
            if "data" in donnees:
                for station in donnees["data"]:
                    #  Recuperer un code BSS valide
                    if "bss_id" in station and station["bss_id"]:
                        liste_stations.append(station["bss_id"])

            print(f"-> {len(liste_stations)} stations trouvées !")
            self.donnees_brutes = donnees["data"]

            return liste_stations

        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la communication avec l'API : {e}")
            self.donnees_brutes = []
            return []


    def generer_df(self) -> pd.DataFrame:
        """
        Transforme les données brutes JSON en pd.DataFrame
        """
        if not self.donnees_brutes:
            print("Le JSON brut est vide")
            return pd.DataFrame()

        df = pd.DataFrame(self.donnees_brutes)

        # 1. Formatage des dates. TODO : que faire de "date maj" ?
        colonnes_dates = ['date_debut_mesure', 'date_fin_mesure', 'date_maj']
        for col in colonnes_dates:
            if col in df.columns:
                df[col] = df[col].apply(convertir_date_safe)

        # 2. Aplatissement des listes (ex: codes_bdlisa renvoie souvent ["113AC10"])
        # Pour faire un DataFrame propre et exportable, on transforme les listes en chaînes de caractères.
        colonnes_listes = ['codes_bdlisa', 'urns_bdlisa', 'codes_masse_eau_edl', 'noms_masse_eau_edl']
        for col in colonnes_listes:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ",".join(map(str, x)) if isinstance(x, list) else x)

        self.catalogue_df = df
        print(f"-> Catalogue structuré en DataFrame ({df.shape[0]} lignes, {df.shape[1]} colonnes).")
        return self.catalogue_df


def convertir_date_safe(valeur):
    '''
    Fonction pour supprimer les bugs pour les dates
    '''

    if pd.isna(valeur) or valeur == "None":
        return pd.NaT
    try:
        return pd.to_datetime(str(valeur), utc=True)
    except:
        return pd.NaT
