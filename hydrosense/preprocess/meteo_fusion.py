from hydrosense.database.meteo import CatalogueMeteo
from hydrosense.utils.geo import calc_dist, appliquer_idw_df
from hydrosense.database.bigquery import info_piezo
import pandas as pd
import numpy as np


class SynthPrecipitation():
    """
    Classe qui permet la création d'un nouveau pd.Dataframe
    avec de la donnée météorologique concomitant à un dataframe de pizeo.
    """

    def __init__(self, df_piezo: pd.DataFrame, bss_piezo: str):
        """
        Reconnait un dataframe de piezometrie nettoyé qui a un pas de temps régulier et quotidien
        example : le bourdet BSS001QHYH
        """
        self.df_piezo = df_piezo.copy()
        self.bss_id = bss_piezo
        self.info = info_piezo(bss_piezo)

        try:
            self.df_piezo['date_mesure'] = pd.to_datetime(self.df_piezo['date_mesure'])
        except:
            print("Aucune colonne de date trouvée dans df_piezo.")


    def search(self, code_dep:str, n_neighbors = 3):
        """
        Utilise le module database.meteo pour charger un dictionnaire de dataframe avec
        des données météo.
        Si n_neighbors = 1, choisi le plus proche géographiquement. see calc_dist()

        Retourne un dataframe meteo interpolé sur index de dates du dataframe piezo
        """
        print(f"🌍 Recherche des stations météo pour le département {code_dep}...")

        meteo = CatalogueMeteo()
        dict_meteo_dept = meteo.extraire_departement(code_dep)

        if not dict_meteo_dept:
            print("❌ Aucune donnée météo récupérée.")
            return {}

        # Récupération des coordonnées du piézomètre grace à la fonction info_piezo
        x_piezo, y_piezo = self.info.loc[0,'x'], self.info.loc[0,'y']


        # Calcul des distances pour chaque station météo
        distances_stations = []
        for id_station, df_m in dict_meteo_dept.items():
            if df_m.empty:
                continue

            x_station, y_station = df_m['LON'].iloc[0], df_m['LAT'].iloc[0]

            distance_km = calc_dist(x_piezo, y_piezo, x_station, y_station)

            distances_stations.append({
                'id_station': id_station,
                'distance': distance_km,
                'dataframe': df_m
            })

        distances_stations.sort(key=lambda x: x['distance'])
        top_stations = distances_stations[:n_neighbors]

        resultat = {}
        for item in top_stations:
            print(f"🎯 Station météo retenue : {item['id_station']} (à {item['distance']:.2f} km)")

        # On stocke le dataframe ET la distance pour le calcul des poids
            resultat[item['id_station']] = {
                'dataframe': item['dataframe'],
                'distance': item['distance']
            }

        return resultat


    def merge(self, dict_meteo):
        """
        concatenatation de la meteo et du piezo
        """

        df_final = self.df_piezo.copy()

        # Listes pour stocker l'ordre exact des stations et de leurs distances
        distances_ordonnees = []
        ids_stations = []

        for id_station, infos in dict_meteo.items():
            df_m_clean = infos['dataframe'].copy()
            distances_ordonnees.append(infos['distance'])
            ids_stations.append(id_station)

            # On force la date météo en datetime64 pour la jointure
            df_m_clean['date_RR'] = pd.to_datetime(df_m_clean['date_RR'])

            # On ne garde que la date et les variables utiles pour ne pas polluer le tableau final
            colonnes_utiles = ['date_RR', 'RR', 'TM', 'FFM']
            colonnes_existantes = [c for c in colonnes_utiles if c in df_m_clean.columns]
            df_m_clean = df_m_clean[colonnes_existantes]

            # Renommage dynamique pour identifier la station (ex: RR devient RR_17132001)
            renommage = {c: f"{c}_{id_station}" for c in ['RR', 'TM', 'FFM'] if c in df_m_clean.columns}
            df_m_clean.rename(columns=renommage, inplace=True)

            # Jointure à GAUCHE (left merge) : S'il manque de la météo ce jour-là, ça fera un NaN.
            df_final = df_final.merge(
                df_m_clean,
                left_on='date_mesure',
                right_on='date_RR',
                how='left'
            )

            df_final.drop(columns=['date_RR'], inplace=True, errors='ignore')

        # ------
        print("\n⚙️ Calcul des variables synthétiques (Pondération IDW)...")
        variables_climatiques = ['RR', 'TM', 'FFM']

        for var in variables_climatiques:
            # On liste les colonnes fusionnées (ex: ['RR_17132001', 'RR_17154002', ...])
            cols_var = [f"{var}_{id_st}" for id_st in ids_stations if f"{var}_{id_st}" in df_final.columns]

            if cols_var:
                # Application de notre fonction vectorisée et On arrondit à 2 décimales
                df_final[f'{var}_synth'] = appliquer_idw_df(df_final, cols_var, distances_ordonnees)
                df_final[f'{var}_synth'] = df_final[f'{var}_synth'].round(2)

        return df_final


if __name__ == '__main__':

    test_file = '/home/charourou/projects/Projet_Hydrosense/data/piezo_bourdet_clean.csv'
    df_piezo = pd.read_csv(test_file, sep=';')

    synthetiseur = SynthPrecipitation(df_piezo, 'BSS001QHYH')
    departement = '79'
    resultat = synthetiseur.search(departement)
    print(resultat)

    df_final = synthetiseur.merge(resultat)
    print(df_final)
